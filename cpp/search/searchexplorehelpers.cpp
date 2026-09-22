#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

#include <cstdlib>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

//----------------------------------------------------------------------------------------
//Vectorized PUCT child-selection kernels.
//
//The innermost loop of selectBestChildToDescend scores every child with
//  score = exploreScaling * policy / (1 + childWeight) + flip * childUtility
//and takes the argmax. On x86-64 we compute that with AVX-512 (8 doubles/call)
//or AVX2 (4 doubles/call), selected at runtime; everything else keeps the
//original scalar loop.
//
//Bit-exactness with the scalar path in Search::getExploreSelectionValue:
// - all math is IEEE double with the same operation order (mul, then div by
//   (1+w), then add of the negated-or-not utility);
// - no FMA contraction: the intrinsics are explicit, and the scalar form has
//   no multiplies feeding adds, so neither side can contract;
// - ties prefer the lowest index, matching the scalar strict ">" compare;
// - children with policy < 0 (illegal moves) are masked to -infinity so they
//   can never strictly exceed the initial best, matching the scalar behavior
//   where POLICY_ILLEGAL_SELECTION_VALUE never beats the running max.
//----------------------------------------------------------------------------------------
namespace {

constexpr double PUCT_VEC_ILLEGAL_SCORE = -1e50; // == Search::POLICY_ILLEGAL_SELECTION_VALUE

inline double puctVecScore(double exploreScaling, double policy, double childWeight, double childUtility, double flip) {
  return exploreScaling * policy / (1.0 + childWeight) + flip * childUtility;
}

#if defined(__x86_64__)

inline bool puctVecAvx512Supported() {
  static const bool supported = __builtin_cpu_supports("avx512f");
  return supported;
}

inline bool puctVecAvx2Supported() {
  static const bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return supported;
}

__attribute__((target("avx512f")))
int puctVecArgmaxAvx512(const float* policy, const double* weight, const double* utility, int n, double exploreScaling, double flip) {
  const __m512d v_es = _mm512_set1_pd(exploreScaling);
  const __m512d v_flip = _mm512_set1_pd(flip);
  const __m512d v_one = _mm512_set1_pd(1.0);
  const __m512d v_ninf = _mm512_set1_pd(-INFINITY);
  const __m512d v_zero = _mm512_setzero_pd();
  double best = PUCT_VEC_ILLEGAL_SCORE;
  int bestIdx = -1;
  int i = 0;
  for(; i + 8 <= n; i += 8) {
    __m512d p = _mm512_cvtps_pd(_mm256_loadu_ps(&policy[i]));
    __m512d w = _mm512_loadu_pd(&weight[i]);
    __m512d u = _mm512_loadu_pd(&utility[i]);
    __mmask8 valid = _mm512_cmp_pd_mask(p, v_zero, _CMP_GE_OQ);
    __m512d score = _mm512_add_pd(
      _mm512_div_pd(_mm512_mul_pd(v_es, p), _mm512_add_pd(v_one, w)),
      _mm512_mul_pd(v_flip, u)
    );
    score = _mm512_mask_mov_pd(v_ninf, valid, score);
    double bmax = _mm512_reduce_max_pd(score);
    if(bmax > best) {
      __mmask8 eq = _mm512_cmp_pd_mask(score, _mm512_set1_pd(bmax), _CMP_EQ_OQ);
      bestIdx = i + __builtin_ctz((unsigned)eq);
      best = bmax;
    }
  }
  for(; i < n; i++) {
    double p = (double)policy[i];
    if(p < 0)
      continue;
    double score = puctVecScore(exploreScaling, p, weight[i], utility[i], flip);
    if(score > best) {
      best = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

__attribute__((target("avx2,fma")))
int puctVecArgmaxAvx2(const float* policy, const double* weight, const double* utility, int n, double exploreScaling, double flip) {
  const __m256d v_es = _mm256_set1_pd(exploreScaling);
  const __m256d v_flip = _mm256_set1_pd(flip);
  const __m256d v_one = _mm256_set1_pd(1.0);
  const __m256d v_ninf = _mm256_set1_pd(-INFINITY);
  const __m256d v_zero = _mm256_setzero_pd();
  double best = PUCT_VEC_ILLEGAL_SCORE;
  int bestIdx = -1;
  int i = 0;
  for(; i + 4 <= n; i += 4) {
    __m256d p = _mm256_cvtps_pd(_mm_loadu_ps(&policy[i]));
    __m256d w = _mm256_loadu_pd(&weight[i]);
    __m256d u = _mm256_loadu_pd(&utility[i]);
    __m256d valid = _mm256_cmp_pd(p, v_zero, _CMP_GE_OQ);
    __m256d score = _mm256_add_pd(
      _mm256_div_pd(_mm256_mul_pd(v_es, p), _mm256_add_pd(v_one, w)),
      _mm256_mul_pd(v_flip, u)
    );
    score = _mm256_blendv_pd(v_ninf, score, valid);
    __m256d m1 = _mm256_max_pd(score, _mm256_permute2f128_pd(score, score, 0x01));
    __m256d m2 = _mm256_max_pd(m1, _mm256_permute_pd(m1, 0x05));
    double bmax = _mm256_cvtsd_f64(m2);
    if(bmax > best) {
      int eq = _mm256_movemask_pd(_mm256_cmp_pd(score, _mm256_set1_pd(bmax), _CMP_EQ_OQ));
      bestIdx = i + __builtin_ctz((unsigned)eq);
      best = bmax;
    }
  }
  for(; i < n; i++) {
    double p = (double)policy[i];
    if(p < 0)
      continue;
    double score = puctVecScore(exploreScaling, p, weight[i], utility[i], flip);
    if(score > best) {
      best = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}
#endif

//Argmax of the PUCT selection score over n gathered children, or -1 if none is
//a legal move. See the block comment above for the exactness contract.
int puctVecArgmax(const float* policy, const double* weight, const double* utility, int n, double exploreScaling, double flip) {
  if(n <= 0)
    return -1;
#if defined(__x86_64__)
  if(puctVecAvx512Supported())
    return puctVecArgmaxAvx512(policy, weight, utility, n, exploreScaling, flip);
  if(puctVecAvx2Supported())
    return puctVecArgmaxAvx2(policy, weight, utility, n, exploreScaling, flip);
#endif
  double best = PUCT_VEC_ILLEGAL_SCORE;
  int bestIdx = -1;
  for(int i = 0; i < n; i++) {
    double p = (double)policy[i];
    if(p < 0)
      continue;
    double score = puctVecScore(exploreScaling, p, weight[i], utility[i], flip);
    if(score > best) {
      best = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

bool puctVecEnabledByEnv() {
  const char* s = std::getenv("KATAGO_PUCT_VEC");
  return s == NULL || std::strcmp(s, "0") != 0;
}

}

static double cpuctExploration(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.cpuctExploration +
    searchParams.cpuctExplorationLog * log((totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase);
}

static double cpuctExplorationHuman(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.humanSLCpuctExploration + searchParams.humanSLCpuctPermanent * sqrt(totalChildWeight);
}

//Tiny constant to add to numerator of puct formula to make it positive
//even when visits = 0.
static constexpr double TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

double Search::getExploreScaling(
  double totalChildWeight, double parentUtilityStdevFactor
) const {
  return
    cpuctExploration(totalChildWeight, searchParams)
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET)
    * parentUtilityStdevFactor;
}
double Search::getExploreScalingHuman(
  double totalChildWeight
) const {
  return
    cpuctExplorationHuman(totalChildWeight, searchParams)
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET);
}

double Search::getExploreSelectionValue(
  double exploreScaling,
  double nnPolicyProb,
  double childWeight,
  double childUtility,
  Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent = exploreScaling * nnPolicyProb / (1.0 + childWeight);

  //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;
  return exploreComponent + valueComponent;
}

//Return the childWeight that would make Search::getExploreSelectionValue return the given explore selection value.
//Or return 0, if it would be less than 0.
double Search::getExploreSelectionValueInverse(
  double exploreSelectionValue,
  double exploreScaling,
  double nnPolicyProb,
  double childUtility,
  Player pla
) const {
  if(nnPolicyProb < 0)
    return 0;
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;

  double exploreComponent = exploreSelectionValue - valueComponent;
  double exploreComponentScaling = exploreScaling * nnPolicyProb;

  //Guard against float weirdness
  if(exploreComponent <= 0)
    return 1e100;

  double childWeight = exploreComponentScaling / exploreComponent - 1;
  if(childWeight < 0)
    childWeight = 0;
  return childWeight;
}

static void maybeApplyWideRootNoise(
  double& childUtility,
  float& nnPolicyProb,
  const SearchParams& searchParams,
  SearchThread* thread,
  const SearchNode& parent
) {
  //For very large wideRootNoise, go ahead and also smooth out the policy
  nnPolicyProb = (float)pow(nnPolicyProb, 1.0 / (4.0*searchParams.wideRootNoise + 1.0));
  if(thread->rand.nextBool(0.5)) {
    double bonus = searchParams.wideRootNoise * std::fabs(thread->rand.nextGaussian());
    if(parent.nextPla == P_WHITE)
      childUtility += bonus;
    else
      childUtility -= bonus;
  }
}


double Search::getExploreSelectionValueOfChild(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  Loc moveLoc,
  double exploreScaling,
  double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
  double parentUtility, double parentWeightPerVisit,
  bool isDuringSearch, bool antiMirror, double maxChildWeight,
  bool countEdgeVisit,
  SearchThread* thread
) const {
  (void)parentUtility;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double childWeight;
  if(countEdgeVisit)
    childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);
  else
    childWeight = child->stats.weightSum.load(std::memory_order_acquire);

  //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
  //been expanded but its thread not yet finished its first visit.
  //It's also possible that we observe childWeight <= 0 even though childVisits >= due to multithreading, the two could
  //be out of sync briefly since they are separate atomics.
  double childUtility;
  if(childVisits <= 0 || childWeight <= 0.0)
    childUtility = fpuValue;
  else {
    childUtility = utilityAvg;

    //Tiny adjustment for passing
    double endingScoreBonus = getEndingWhiteScoreBonus(parent,moveLoc);
    if(endingScoreBonus != 0)
      childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);
  }

  //Virtual losses to direct threads down different paths
  if(childVirtualLosses > 0) {
    double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;

    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
    double virtualLossUtility = (parent.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
    double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
    childWeight += virtualLossWeight;
  }

  if(isDuringSearch && (&parent == rootNode) && countEdgeVisit) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small, assuming
    //its average weight per visit is maintained.
    //We use childVisits rather than childEdgeVisits for the final estimate since when childEdgeVisits < childVisits, adding new visits is instant.
    if(searchParams.futileVisitsThreshold > 0) {
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = (childEdgeVisits + 1.0) / (childWeight + parentWeightPerVisit);
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(childVisits + thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    //Hack to get the root to funnel more visits down child branches
    if(searchParams.rootDesiredPerChildVisitsCoeff > 0.0) {
      if(nnPolicyProb > 0 && childWeight < sqrt(nnPolicyProb * totalChildWeight * searchParams.rootDesiredPerChildVisitsCoeff)) {
        return 1e20;
      }
    }
    //Hack for hintloc - must search this move almost as often as the most searched move
    if(rootHintLoc != Board::NULL_LOC && moveLoc == rootHintLoc) {
      double averageWeightPerVisit = (childWeight + parentWeightPerVisit) / (childVisits + 1.0);
      ConstSearchNodeChildrenReference children = parent.getChildren();
      int childrenCapacity = children.getCapacity();
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* c = childPointer.getIfAllocated();
        if(c == NULL)
          break;
        int64_t cEdgeVisits = childPointer.getEdgeVisits();
        double cWeight = c->stats.getChildWeight(cEdgeVisits);
        if(childWeight + averageWeightPerVisit < cWeight * 0.8)
          return 1e20;
      }
    }

    if(searchParams.wideRootNoise > 0.0 && nnPolicyProb >= 0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  if(isDuringSearch && antiMirror && nnPolicyProb >= 0 && countEdgeVisit) {
    maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, parentPolicyProbs, parent.nextPla, thread);
    maybeApplyAntiMirrorForcedExplore(childUtility, parentUtility, moveLoc, parentPolicyProbs, childWeight, totalChildWeight, parent.nextPla, thread, parent);
  }

  return getExploreSelectionValue(exploreScaling,nnPolicyProb,childWeight,childUtility,parent.nextPla);
}

double Search::getNewExploreSelectionValue(
  const SearchNode& parent,
  double exploreScaling,
  float nnPolicyProb,
  double fpuValue,
  double parentWeightPerVisit,
  double maxChildWeight,
  bool countEdgeVisit,
  SearchThread* thread
) const {
  double childWeight = 0;
  double childUtility = fpuValue;
  if(&parent == rootNode && countEdgeVisit) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small
    if(searchParams.futileVisitsThreshold > 0) {
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = 1.0 / parentWeightPerVisit;
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    if(searchParams.wideRootNoise > 0.0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  return getExploreSelectionValue(exploreScaling,nnPolicyProb,childWeight,childUtility,parent.nextPla);
}

double Search::getReducedPlaySelectionWeight(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  Loc moveLoc,
  double exploreScaling,
  int64_t childEdgeVisits,
  double bestChildExploreSelectionValue
) const {
  assert(&parent == rootNode);
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);

  //Child visits may be 0 if this function is called in a multithreaded context, such as during live analysis
  //Child weight may also be 0 if it's out of sync.
  if(childVisits <= 0 || childWeight <= 0.0)
    return 0;

  //Tiny adjustment for passing
  double endingScoreBonus = getEndingWhiteScoreBonus(parent,moveLoc);
  double childUtility = utilityAvg;
  if(endingScoreBonus != 0)
    childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);

  double childWeightWeRetrospectivelyWanted = getExploreSelectionValueInverse(
    bestChildExploreSelectionValue, exploreScaling, nnPolicyProb, childUtility, parent.nextPla
  );
  if(childWeight > childWeightWeRetrospectivelyWanted)
    return childWeightWeRetrospectivelyWanted;
  return childWeight;
}

double Search::getFpuValueForChildrenAssumeVisited(
  const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) const {
  int64_t visits = node.stats.visits.load(std::memory_order_acquire);
  double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
  double utilityAvg = node.stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = node.stats.utilitySqAvg.load(std::memory_order_acquire);

  assert(visits > 0);
  assert(weightSum > 0.0);
  parentWeightPerVisit = weightSum / visits;
  parentUtility = utilityAvg;
  double variancePrior = searchParams.cpuctUtilityStdevPrior * searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(visits <= 0 || weightSum <= 1)
    parentUtilityStdev = searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    //Make sure we're robust to numerical precision issues or threading desync of these values, so we don't observe negative variance
    if(utilitySqAvg < utilitySq)
      utilitySqAvg = utilitySq;
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
        / (variancePriorWeight + weightSum - 1.0)
        - utilitySq
      )
    );
  }
  parentUtilityStdevFactor = 1.0 + searchParams.cpuctUtilityStdevScale * (parentUtilityStdev / searchParams.cpuctUtilityStdevPrior - 1.0);

  double parentUtilityForFPU = parentUtility;
  if(searchParams.fpuParentWeightByVisitedPolicy) {
    double avgWeight = std::min(1.0, pow(policyProbMassVisited, searchParams.fpuParentWeightByVisitedPolicyPow));
    parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * getUtilityFromNN(*(node.getNNOutput()));
  }
  else if(searchParams.fpuParentWeight > 0.0) {
    parentUtilityForFPU = searchParams.fpuParentWeight * getUtilityFromNN(*(node.getNNOutput())) + (1.0 - searchParams.fpuParentWeight) * parentUtility;
  }

  double fpuValue;
  {
    double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
    double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;

    double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
    fpuValue = pla == P_WHITE ? parentUtilityForFPU - reduction : parentUtilityForFPU + reduction;
    double lossValue = pla == P_WHITE ? -utilityRadius : utilityRadius;
    fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  }

  return fpuValue;
}


void Search::selectBestChildToDescend(
  SearchThread& thread, const SearchNode& node, SearchNodeState nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc, bool& countEdgeVisit,
  bool isRoot) const
{
  assert(thread.pla == node.nextPla);

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;
  countEdgeVisit = true;

  //A focus playout selects one of the focus moves at random in proportion to its weight. See FocusMoves in search.h.
  //Focus moves that are illegal, avoided, or otherwise not searchable at the root get zero weight, and if none are
  //searchable the playout is a normal one. If symmetry pruning searches an equivalent copy of the chosen move
  //instead, that copy is the target. Analysis output reports the chosen move as a symmetry of that copy, so the
  //focus still shows up on the requested move. The target gets a selection value above every other move, so it is
  //chosen whenever it is selectable. If it is not, the playout falls through to the normal best move, still as a
  //weightless and uncounted playout.
  bool focusPlayout = false;
  Loc focusTarget = Board::NULL_LOC;
  if(isRoot) {
    const FocusMoves* focus = rootFocus.load(std::memory_order_acquire);
    if(focus != nullptr && thread.rand.nextDouble() < focus->prob) {
      const std::vector<int>& rootAvoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
      auto rootFocusTargetOf = [&](Loc loc) {
        if(!rootHistory.isLegal(rootBoard,loc,rootPla))
          return Board::NULL_LOC;
        if(rootAvoidMoveUntilByLoc.size() > 0 && rootAvoidMoveUntilByLoc[loc] > 0)
          return Board::NULL_LOC;
        Loc target = rootSymRepresentativeLoc[loc];
        if(target == Board::NULL_LOC || !isAllowedRootMove(target))
          return Board::NULL_LOC;
        return target;
      };

      std::vector<double>& cumWeights = thread.rootFocusCumWeightsBuf;
      cumWeights.resize(focus->moves.size());
      double cumWeight = 0.0;
      for(size_t i = 0; i<focus->moves.size(); i++) {
        if(rootFocusTargetOf(focus->moves[i]) != Board::NULL_LOC)
          cumWeight += focus->weights[i];
        cumWeights[i] = cumWeight;
      }
      if(cumWeight > 0.0) {
        size_t idx = thread.rand.nextIndexCumulative(cumWeights.data(), cumWeights.size());
        focusPlayout = true;
        focusTarget = rootFocusTargetOf(focus->moves[idx]);
        countEdgeVisit = false;
        thread.shouldCountPlayout = false;
      }
    }
  }

  ConstSearchNodeChildrenReference children = node.getChildren(nodeState);
  int childrenCapacity = children.getCapacity();

  double policyProbMassVisited = 0.0;
  double maxChildWeight = 0.0;
  double totalChildWeight = 0.0;
  int64_t totalChildEdgeVisits = 0;
  int numChildrenPresent = 0;
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    numChildrenPresent++;
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    int movePos = getPos(moveLoc);
    float nnPolicyProb = policyProbs[movePos];
    if(nnPolicyProb < 0)
      continue;
    policyProbMassVisited += nnPolicyProb;

    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
    totalChildEdgeVisits += edgeVisits;
  }

  bool useHumanSL = false;
  if(humanEvaluator != NULL &&
     (searchParams.humanSLProfile.initialized || !humanEvaluator->requiresSGFMetadata())
  ) {
    const NNOutput* humanOutput = node.getHumanOutput();
    if(humanOutput != NULL) {
      double weightlessProb;
      double weightfulProb;
      if(isRoot) {
        weightlessProb = searchParams.humanSLRootExploreProbWeightless;
        weightfulProb = searchParams.humanSLRootExploreProbWeightful;
      }
      else if(thread.pla == rootPla) {
        weightlessProb = searchParams.humanSLPlaExploreProbWeightless;
        weightfulProb = searchParams.humanSLPlaExploreProbWeightful;
      }
      else {
        weightlessProb = searchParams.humanSLOppExploreProbWeightless;
        weightfulProb = searchParams.humanSLOppExploreProbWeightful;
      }

      double totalHumanProb = weightlessProb + weightfulProb;
      if(totalHumanProb > 0.0) {
        double r = thread.rand.nextDouble();
        if(r < weightlessProb) {
          useHumanSL = true;
          countEdgeVisit = false;
        }
        else if(r < totalHumanProb) {
          useHumanSL = true;
        }
      }
    }

    // Swap out policy and also recompute policy prob mass visited
    if(useHumanSL) {
      nnOutput = humanOutput;
      policyProbMassVisited = 0.0;
      assert(nnOutput != NULL);
      policyProbs = nnOutput->getPolicyProbsMaybeNoised();
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* child = childPointer.getIfAllocated();
        if(child == NULL)
          break;
        Loc moveLoc = childPointer.getMoveLocRelaxed();
        int movePos = getPos(moveLoc);
        float nnPolicyProb = policyProbs[movePos];
        if(nnPolicyProb < 0)
          continue;
        policyProbMassVisited += nnPolicyProb;
      }
    }
  }

  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.0001);

  //If we're doing a weightless visit, then we should redo PUCT to operate on child node weight, not child edge weight
  if(!countEdgeVisit) {
    totalChildWeight = 0.0;
    maxChildWeight = 0.0;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      double childWeight = child->stats.weightSum.load(std::memory_order_acquire);
      totalChildWeight += childWeight;
      if(childWeight > maxChildWeight)
        maxChildWeight = childWeight;
    }
  }

  //First play urgency
  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, thread.pla, isRoot, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE] = { }; // Initialize all to false
  bool antiMirror = searchParams.antiMirror && mirroringPla != C_EMPTY && isMirroringSinceSearchStart(thread.history,0);

  double exploreScaling;
  if(useHumanSL)
    exploreScaling = getExploreScalingHuman(totalChildWeight);
  else
    exploreScaling = getExploreScaling(totalChildWeight, parentUtilityStdevFactor);

  //Try all existing children
  //Also count how many children we actually find
  numChildrenFound = 0;

  //Vectorized fast path. For any node that is not the root and has no human-SL
  //policy and no anti-mirror adjustments and no focus playout, each child's
  //selection value is the fixed PUCT expression below over three atomically-loaded
  //stats, so we gather them and argmax with SIMD. Only worth it once the child
  //count is large enough that the arithmetic outweighs the loads - for the young
  //nodes (<=8 children) that most descent steps hit, the gather overhead cancels
  //the win. The root is excluded because it carries extra per-child hacks
  //(futile-visit pruning, hintloc, wide-root noise, endgame score bonuses, focus)
  //with early-return values. countEdgeVisit is always true when !useHumanSL.
  //Set KATAGO_PUCT_VEC=0 to disable (A/B testing).
  static const bool puctVecEnabled = puctVecEnabledByEnv();
  if(puctVecEnabled && !useHumanSL && !antiMirror && &node != rootNode && !focusPlayout && numChildrenPresent >= 32) {
    alignas(64) float vPolicy[NNPos::MAX_NN_POLICY_SIZE];
    alignas(64) double vWeight[NNPos::MAX_NN_POLICY_SIZE];
    alignas(64) double vUtility[NNPos::MAX_NN_POLICY_SIZE];
    const double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
    int n = 0;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      numChildrenFound++;
      int64_t childEdgeVisits = childPointer.getEdgeVisits();
      Loc moveLoc = childPointer.getMoveLocRelaxed();
      int movePos = getPos(moveLoc);
      float nnPolicyProb = policyProbs[movePos];

      if(nnPolicyProb >= 0) {
        //Same loads and adjustments as Search::getExploreSelectionValueOfChild's
        //non-root path (getEndingWhiteScoreBonus is always 0 away from the root).
        int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);
        int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
        double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
        double childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);

        double childUtility;
        if(childVisits <= 0 || childWeight <= 0.0)
          childUtility = fpuValue;
        else
          childUtility = utilityAvg;

        if(childVirtualLosses > 0) {
          double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;
          double virtualLossUtility = (node.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
          double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
          childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
          childWeight += virtualLossWeight;
        }

        vPolicy[n] = nnPolicyProb;
        vWeight[n] = childWeight;
        vUtility[n] = childUtility;
      }
      else {
        //Illegal moves never win the argmax; masked out by negative policy.
        vPolicy[n] = nnPolicyProb;
        vWeight[n] = 0.0;
        vUtility[n] = 0.0;
      }
      n++;

      posesWithChildBuf[movePos] = true;
    }

    double flip = node.nextPla == P_WHITE ? 1.0 : -1.0;
    int vecBest = puctVecArgmax(vPolicy, vWeight, vUtility, n, exploreScaling, flip);
    if(vecBest >= 0) {
      bestChildIdx = vecBest;
      bestChildMoveLoc = children[vecBest].getMoveLocRelaxed();
      maxSelectionValue = puctVecScore(exploreScaling, vPolicy[vecBest], vWeight[vecBest], vUtility[vecBest], flip);
    }
  }
  else {
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      numChildrenFound++;
      int64_t childEdgeVisits = childPointer.getEdgeVisits();

      Loc moveLoc = childPointer.getMoveLocRelaxed();
      bool isDuringSearch = true;
      double selectionValue = getExploreSelectionValueOfChild(
        node,policyProbs,child,
        moveLoc,
        exploreScaling,
        totalChildWeight,childEdgeVisits,fpuValue,
        parentUtility,parentWeightPerVisit,
        isDuringSearch,antiMirror,maxChildWeight,
        countEdgeVisit,
        &thread
      );
      if(focusPlayout && moveLoc == focusTarget && selectionValue > POLICY_ILLEGAL_SELECTION_VALUE)
        selectionValue = ROOT_FOCUS_SELECTION_VALUE;
      if(selectionValue > maxSelectionValue) {
        // if(child->state.load(std::memory_order_seq_cst) == SearchNode::STATE_EVALUATING) {
        //   selectionValue -= EVALUATING_SELECTION_VALUE_PENALTY;
        //   if(isRoot && child->prevMoveLoc == Location::ofString("K4",thread.board)) {
        //     out << "ouch" << "\n";
        //   }
        // }
        maxSelectionValue = selectionValue;
        bestChildIdx = i;
        bestChildMoveLoc = moveLoc;
      }

      posesWithChildBuf[getPos(moveLoc)] = true;
    }
  }

  const std::vector<int>& avoidMoveUntilByLoc = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;

  //Try all the things in the eval cache that are moves we haven't visited yet.
  //Use the normal new explore selection value for them but with their eval cache utility instead of FPU.
  //Might explore things out of descending policy order!
  if(searchParams.useEvalCache && searchParams.useGraphSearch && node.evalCacheEntry != nullptr && mirroringPla == C_EMPTY && !node.forceNonTerminal) {
    for(const auto& pair: node.evalCacheEntry->firstExploreEvals) {
      Loc moveLoc = pair.first;
      int movePos = getPos(moveLoc);
      bool alreadyTried = posesWithChildBuf[movePos];
      if(alreadyTried)
        continue;

      //Quit immediately for illegal moves
      float nnPolicyProb = policyProbs[movePos];
      if(nnPolicyProb < 0)
        continue;

      //Special logic for the root
      if(isRoot) {
        assert(thread.board.pos_hash == rootBoard.pos_hash);
        assert(thread.pla == rootPla);
        if(!isAllowedRootMove(moveLoc))
          continue;
      }
      if(avoidMoveUntilByLoc.size() > 0) {
        assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
        int untilDepth = avoidMoveUntilByLoc[moveLoc];
        if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
          continue;
      }

      FirstExploreEval eval = pair.second;
      double cacheAvgUtility =
        getResultUtility(eval.avgWinLoss,0.0)
        + getScoreUtility(eval.avgScoreMean, eval.avgScoreMean * eval.avgScoreMean);

      double selectionValue = getNewExploreSelectionValue(
        node,
        exploreScaling,
        nnPolicyProb,cacheAvgUtility,
        parentWeightPerVisit,
        maxChildWeight,
        countEdgeVisit,
        &thread
      );
      if(selectionValue > maxSelectionValue) {
        maxSelectionValue = selectionValue;
        bestChildIdx = numChildrenFound;
        bestChildMoveLoc = moveLoc;
      }
    }
  }

  //Try the new child with the best policy value
  Loc bestNewMoveLoc = Board::NULL_LOC;
  float bestNewNNPolicyProb = -1.0f;
  bool focusTargetIsNewMove = false;
  for(int movePos = 0; movePos<policySize; movePos++) {
    bool alreadyTried = posesWithChildBuf[movePos];
    if(alreadyTried)
      continue;

    //Quit immediately for illegal moves
    float nnPolicyProb = policyProbs[movePos];
    if(nnPolicyProb < 0)
      continue;

    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC)
      continue;

    //Special logic for the root
    if(isRoot) {
      assert(thread.board.pos_hash == rootBoard.pos_hash);
      assert(thread.pla == rootPla);
      if(!isAllowedRootMove(moveLoc))
        continue;
    }
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      int untilDepth = avoidMoveUntilByLoc[moveLoc];
      if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
        continue;
    }

    if(antiMirror) {
      maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, policyProbs, node.nextPla, &thread);
    }

    if(focusPlayout && moveLoc == focusTarget)
      focusTargetIsNewMove = true;

    if(nnPolicyProb > bestNewNNPolicyProb) {
      bestNewNNPolicyProb = nnPolicyProb;
      bestNewMoveLoc = moveLoc;
    }
  }
  //A focus target that is a not-yet-visited move beats every existing child and every other new move.
  if(focusTargetIsNewMove) {
    maxSelectionValue = ROOT_FOCUS_SELECTION_VALUE;
    bestChildIdx = numChildrenFound;
    bestChildMoveLoc = focusTarget;
  }
  if(bestNewMoveLoc != Board::NULL_LOC) {
    double selectionValue = getNewExploreSelectionValue(
      node,
      exploreScaling,
      bestNewNNPolicyProb,fpuValue,
      parentWeightPerVisit,
      maxChildWeight,
      countEdgeVisit,
      &thread
    );
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = bestNewMoveLoc;
    }
  }

  if(totalChildEdgeVisits >= 2 &&
     searchParams.enableMorePassingHacks &&
     thread.history.passWouldEndPhase(thread.board,thread.pla) &&
     avoidMoveUntilByLoc.size() == 0 && // Don't force playouts if there's any chance we're specifying specific moves since we don't want to force an avoided move
     !focusPlayout // Focus playouts are already forced to specific moves
  ) {
    bool hasPassMove = false;
    bool hasNonPassMove = false;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      Loc moveLoc = childPointer.getMoveLocRelaxed();
      if(moveLoc == Board::PASS_LOC)
        hasPassMove = true;
      else
        hasNonPassMove = true;
    }
    if(!hasPassMove && bestChildMoveLoc != Board::PASS_LOC && bestChildMoveLoc != Board::NULL_LOC) {
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = Board::PASS_LOC;
      countEdgeVisit = false;
      // Specifically for these special extra-pass search playouts, we don't count them for the purpose of visit/playout limits.
      thread.shouldCountPlayout = false;
    }
    else if(!hasNonPassMove && bestChildMoveLoc == Board::PASS_LOC && bestNewMoveLoc != Board::PASS_LOC && bestNewMoveLoc != Board::NULL_LOC) {
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = bestNewMoveLoc;
      countEdgeVisit = false;
      // Specifically for these special extra-pass search playouts, we don't count them for the purpose of visit/playout limits.
      thread.shouldCountPlayout = false;
    }
  }

}
