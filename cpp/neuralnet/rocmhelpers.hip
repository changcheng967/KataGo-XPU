// ROCm/HIP wrapper for the shared CUDA/ROCm GPU kernels.
// All kernel code lives in cudaandrocmhelpers.inc, which is shared with the CUDA backend
// (cudahelpers.cu). See the comment at the top of that file for the macro contract.

#include "../neuralnet/rocmhelpers.h"

// HIP_SUPPORTS_FP16 is defined (or not) by the build system (see CMakeLists.txt) uniformly for
// all target archs at once - unlike CUDA there is no per-arch preprocessor gating here. Every
// arch KataGo's default arch list targets supports packed FP16. If this is absent, the half
// kernels compile to empty stubs, and the backend detects that via customCudaFp16KernelsCompiled()
// and refuses FP16 at runtime rather than silently computing garbage.
#ifdef HIP_SUPPORTS_FP16
#define KATAGO_GPU_SUPPORTS_FP16
#endif

#define KATAGO_GPU_HIP 1
#define KATAGO_GPU_SINCOSF sincosf

#include "../neuralnet/cudaandrocmhelpers.inc"

//--------------------------------------------------------------------------------------------------------------
// Decomposed fp16 attention for the ROCm backend: rocBLAS strided-batched GEMMs for QK^T and AV
// plus the layout transposes and warp-softmax kernels defined in cudaandrocmhelpers.inc. See the
// measured rationale there: rocBLAS reaches 7.6-14.9 TFLOP/s at exactly KataGo's attention shapes
// on gfx906-class DCUs vs ~2.9 TFLOP/s for the fused kernel's scalar-fp32 compute — a clear win
// for head-dim-64 nets (tf3/b11c768), roughly a tie at head dim 32.
//
// Returns false for shapes this path does not handle (the caller falls back to
// customCudaFlashAttention) and when KATAGO_ROCM_ATTN_DECOMP=0. scratchMem must hold
//   4*batchSize*seqLen*numHeads*headDim + batchSize*numHeads*seqLen*((seqLen+7)&~7)
// halfs; the backend sizes it for the evaluator's max batch so it is allocated once.
// Error checks throw rather than using CUBLAS_ERR, honoring the helpers-file contract of not
// pulling in rocmerrorcheck.h (which would redefine CUDA_ERR/CUBLAS_ERR in this TU).
#include "../neuralnet/rocmcudanames.h"
#include <cstdlib>
#include <string>

bool customRocmAttentionDecomposedHalf(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  cudaStream_t stream, cublasHandle_t blasHandle,
  const void* oneHalf, const void* zeroHalf, void* scratchMem
) {
  static const bool disabledByEnv = []() {
    const char* e = std::getenv("KATAGO_ROCM_ATTN_DECOMP");
    return e != NULL && std::string(e) == "0";
  }();
  if(disabledByEnv)
    return false;
  // GQA (numKVHeads < numHeads) needs non-uniform batch strides for the K/V operands; asymmetric
  // head dims complicate the scratch carving. KataGo's transformer nets are all 1:1 and symmetric.
  if(numKVHeads != numHeads || qHeadDim != vHeadDim)
    return false;
  if(qHeadDim != 32 && qHeadDim != 64)
    return false;
  if(seqLen > 384)  // softmax stages <= 6*64 = 384 row floats in registers
    return false;
  if(batchSize * numHeads > 65535)
    return false;
  // The softmax kernel assumes 64-lane wavefronts (true on the gfx906-class DCUs this path
  // targets); on 32-lane RDNA parts the shuffle reduction would be wrong, so fall back.
  // (DTK's HIP calls the attribute WarpSize rather than WaveFrontWidth.)
  static const bool waveIs64 = []() {
    int dev = 0;
    int width = 0;
    if(hipGetDevice(&dev) != hipSuccess ||
       hipDeviceGetAttribute(&width, hipDeviceAttributeWarpSize, dev) != hipSuccess)
      return false;
    return width == 64;
  }();
  if(!waveIs64)
    return false;

  int bh = batchSize * numHeads;
  int td = numHeads * qHeadDim;
  // Pad the score-matrix row stride to a multiple of 8 halves so every row starts
  // 4B-aligned; the softmax then does half2 IO and the GEMMs take it as ldc/stride.
  int rowStride = (seqLen + 7) & ~7;
  size_t densePerHeadElems = (size_t)bh * seqLen * qHeadDim;
  size_t sElems = (size_t)bh * seqLen * rowStride;

  half* Qd = (half*)scratchMem;
  half* Kd = Qd + densePerHeadElems;
  half* Vd = Kd + densePerHeadElems;
  half* Dd = Vd + densePerHeadElems;
  half* S = Dd + densePerHeadElems;
  (void)td;
  (void)sElems;

  int rowsIn = (batchSize * seqLen + 7) / 8;
  int rowsOut = (bh * seqLen + 7) / 8;
  rocmAttnTransposeToDense<<<rowsIn, 256, 0, stream>>>(Q, Qd, seqLen, numHeads, qHeadDim, batchSize * seqLen);
  rocmAttnTransposeToDense<<<rowsIn, 256, 0, stream>>>(K, Kd, seqLen, numKVHeads, qHeadDim, batchSize * seqLen);
  rocmAttnTransposeToDense<<<rowsIn, 256, 0, stream>>>(V, Vd, seqLen, numKVHeads, vHeadDim, batchSize * seqLen);

  // Column-major mapping for the row-major dense tensors (validated against rocBLAS on DTK):
  //   S = Q*K^T (seq x seq)  <=>  S^T = K*Q^T   => A = K dense (opT), B = Q dense (opN)
  //   D = S*V    (seq x hd)  <=>  D^T = V^T*S^T => m = vHeadDim, n = seq, all opN
  const hipblasHalf* alpha = (const hipblasHalf*)oneHalf;
  const hipblasHalf* beta = (const hipblasHalf*)zeroHalf;
  long long strideQKV = (long long)seqLen * qHeadDim;
  hipblasStatus_t st = cublasHgemmStridedBatched(
    blasHandle, HIPBLAS_OP_T, HIPBLAS_OP_N,
    seqLen, seqLen, qHeadDim,
    alpha, (const hipblasHalf*)Kd, qHeadDim, strideQKV,
    (const hipblasHalf*)Qd, qHeadDim, strideQKV,
    beta, (hipblasHalf*)S, rowStride, (long long)seqLen * rowStride, bh);
  if(st != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("customRocmAttentionDecomposedHalf: QK^T hipblasHgemmStridedBatched failed");

  float scale = 1.0f / sqrtf((float)qHeadDim);
  rocmAttnSoftmaxWarpRow<<<bh * seqLen, 64, 0, stream>>>(
    S, mask, bh * seqLen, seqLen, rowStride, numHeads * seqLen, scale);

  st = cublasHgemmStridedBatched(
    blasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N,
    vHeadDim, seqLen, seqLen,
    alpha, (const hipblasHalf*)Vd, vHeadDim, (long long)seqLen * vHeadDim,
    (const hipblasHalf*)S, rowStride, (long long)seqLen * rowStride,
    beta, (hipblasHalf*)Dd, vHeadDim, (long long)seqLen * vHeadDim, bh);
  if(st != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("customRocmAttentionDecomposedHalf: AV hipblasHgemmStridedBatched failed");

  rocmAttnTransposeFromDense<<<rowsOut, 256, 0, stream>>>(Dd, output, seqLen, numHeads, vHeadDim, bh * seqLen);
  return true;
}
