# Hygon DCU support (`USE_BACKEND=ROCM` against DTK) — WORKING

## Status: official backend runs on DCU with ZERO code changes

KataGo's upstream CUDA/ROCm shared backend (`cudaandrocmbackend.inc` via
`rocmbackend.cpp`) configures, builds, and runs against Hygon DTK 26.04 on the
Z200SM_80 (gfx906) with no patches:

```
cmake -S cpp -B build-rocm -DUSE_BACKEND=ROCM \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_PREFIX_PATH=/opt/dtk -DCMAKE_BUILD_TYPE=Release
make -C build-rocm -j6 katago
```

Build notes: source must carry its `.git` dir (git-info step), and git needs
`safe.directory` after kubectl-cp ownership changes. `katago version` reports
"Using ROCm backend, HIP 6.3". fp16 is auto-selected (`useFP16 = true`).

## Device qualification (measured, cpp/scripts/gpubench*.cpp)

| metric | value |
|---|---|
| Device | Z200SM_80, 64 CU, gfx906 (GCN5 vector, no MFMA/BF16), 17.2 GB |
| fp32 GEMM | 10.4 TFLOP/s (96% of the 10.8 ceiling) |
| fp16 GEMM (hipblasHgemm) | 17.4 TFLOP/s (80% of 21.6) |
| HBM | 683 GB/s |

## Engine results (official auto-tuner, 7-core quota + 1 DCU)

Tuner verdict: **numSearchThreads=32 + numNNServerThreadsPerModel=2**
(it found the 2-server-thread +17.3% itself). avgBatch ~10-15.

| model | v/s @100 visits | v/s @800 visits | vs 8-core Zen4 CPU | per-visit Elo anchor |
|---|---|---|---|---|
| tf2-b10c384 | **556** | **713** | 13.2x (54) | 13,712 |
| tf3-b11c768 | 162 | 190 | ~9x | ~14,700 |
| zhizi-b40c768 | **117** | **136** | (unusable on small CPU boxes) | ~14,800 |

During sustained search the DCU runs at **100% utilization, 175W of the 450W
cap, 60°C, VRAM 6%** — the GPU is the worker, thermally and power-wise nowhere
near limits. Deeper searches feed it better (more evals in flight): all models
gain 15-28% from v=100 to v=800; analysis workloads at 1000+ visits will sit at
the top of these ranges.

**Headline: the DCU inverts the CPU model ranking.** On CPU, zhizi-b40 was the
OOM-killer and tf2 was the only sane choice; on the DCU the strongest official
net (zhizi) runs at 117 v/s — 4.75x fewer visits than tf2 gets cannot buy back
~1,100 Elo of per-visit strength, so **strongest-per-wall-clock on DCU is
zhizi-b40c768**, with tf3-b11c768 as the balanced point.

Container logistics that mattered: no internet in the pod, pod→login blocked
(kubectl cp is the only supply path); media.katagotraining.org is directly
fetchable on the login node.

## Theoretical-max accounting (rocprof per-kernel attribution)

Kernel-level profile of tf2 at the tuned config (66,304 dispatches, 7.43s of
GPU kernel time over ~2.5s of wall search — the GPU is ~3x oversubscribed by
kernel queue depth):

| kernel class | GPU time share | calls |
|---|---|---|
| `flashAttentionKernelHalf<32,32,128,32>` (fused QK^T/softmax/AV) | **51.2%** | 5,180 |
| hipBLAS GEMMs (all Cijk_Ailk_Bljk tiles) | ~38% | ~40,000 |
| rmsNorm / swiGLU / cScaleBias-SiLU / RoPE (fused pointwise) | ~9% | ~26,000 |
| fp16↔fp32 copy + misc | ~2% | ~2,200 |

The upstream backend is already well-fused (custom flash-attention, fused
normact/RoPE/SwiGLU — not generic ops). The engine's 604 evals/s = **33% of
the 17.4 TFLOP/s fp16 GEMM peak** is bounded by the attention kernel's
throughput at head_dim 32 / seq 361 shapes, plus the ~10% pointwise tail —
not by missing fusion, CPU feeding (223%/700% CPU), or scheduling gaps.

Remaining headroom, honestly: a flash-attention kernel specialized for
gfx906's vector datapath (the shipped one targets CDNA matrix cores) could
plausibly recover part of the attention share; that is upstream-kernel work,
not backend wiring. Multi-game shared-evaluator mode measured WORSE than
single-search benchmark on this box (492 vs 713 v/s aggregate) — analysis
per-query overhead eats the batch-depth gain at 7-core feeding.

## Dispatch-shape extraction round (post-kernel-patch, all measured)

- **Batch guard on GPU: null** (avgBatch 10.3→11.4 across 0-2000µs, v/s within
  variance) — same sparse-arrival equilibrium as CPU.
- **Demand density: +5% at best** (t=96 × 4 server threads × nnMaxBatchSize=64:
  710-716 v/s; doubling avgBatch 9.7→19.3 moved evals only +13% — per-batch
  constant dominates, deep batching has strongly diminishing returns).
- **Launch overhead ruled out**: mean kernel 199µs, attention 1084µs/call at
  the best config (56,320 dispatches profiled, scripts/rocprof_aggregate.py) —
  µs-scale launch costs are noise; hipGraphs would buy nothing (upstream has
  no graph path anyway).
- **Patched attention kernel confirmed in-engine at ~2.9 TFLOP/s** (was ~2.2):
  the +33% held, but 4 concurrent server streams overlap and absorb it;
  per-batch wall time is the serial 20-layer chain (≈20 × 1.1ms attention).

Verdict: config-level and launch-level levers are exhausted on this box.
The remaining lever is a deeper attention-kernel rewrite (half2-packed LDS
tiles); everything else is at or near its measured ceiling.

## MIGraphX round (DTK 26.04 / MIGraphX 5.2.0, standalone C-API bench)

- **tf2-b10c384 batch=1: 127.8 pos/s** (no offload copy) / **111.6 pos/s**
  (offload_copy=true) vs **78 pos/s** for the ROCm backend on the same GPU —
  MIGraphX graph fusion is worth **+43-64%** even before tuning.
- **Compile cost is the tax**: tf2 = 4.4 min, tf3-b11c768 = 50+ min (400+
  operator shapes autotuned). Startup-time compile is fine for a long-running
  bot; it rules out short-lived processes.
- **Compiled-program save/load is broken on this build**: the saved `.mxr`
  (323 MB) loads in 15 s, but `run` segfaults with a GPU VMFault — the kernel
  dereferences a host pointer (allocation plan not restored); recompiling the
  loaded program fails with a `code_object_op` stride mismatch. A backend must
  compile fresh at every startup; do not ship `.mxr` caching.
- **DTK's C API diverges from stock MIGraphX** (the C++ headers don't compile
  — `shape` ambiguity in `raw_data.hpp`): out-params come first
  (`migraphx_program_run(&out, prog, params)`), names differ
  (`migraphx_onnx_options_create`, `migraphx_target_create(&t, "gpu")`,
  `migraphx_compile_options_create`), status is the `migraphx_status` enum,
  and `migraphx.h` needs `<functional>` included before it. All captured in
  the backend skeleton (`cpp/neuralnet/migraphxbackend.cpp`).
