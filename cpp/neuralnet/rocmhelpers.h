#ifndef NEURALNET_ROCMHELPERS_H_
#define NEURALNET_ROCMHELPERS_H_

#include "../neuralnet/rocmincludes.h"
#include "../neuralnet/activations.h"

#include "../neuralnet/cudaandrocmhelpers.h"
#include "../neuralnet/rocmcudanames.h"

// Decomposed fp16 attention via rocBLAS batched GEMMs (see rocmhelpers.hip and the kernels in
// cudaandrocmhelpers.inc). Returns false when the shape is unsupported (caller falls back to the
// fused kernel) or KATAGO_ROCM_ATTN_DECOMP=0. scratchMem must hold
//   4*batchSize*seqLen*numHeads*headDim + batchSize*numHeads*seqLen*seqLen  halfs.
bool customRocmAttentionDecomposedHalf(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  cudaStream_t stream, cublasHandle_t blasHandle,
  const void* oneHalf, const void* zeroHalf, void* scratchMem);

#endif  // NEURALNET_ROCMHELPERS_H_
