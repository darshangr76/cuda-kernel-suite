# Profiling CUDA Kernels on a T4: Three Bottleneck Types

## TL;DR
Six CUDA kernels, three bottleneck types, two speedups. Every claim
backed by an Nsight Compute profile.

## The setup
Tesla T4 on Colab. Each kernel goes through a 4-step cycle:
write, run, profile, optimize.

## Bottleneck 1: DRAM-bound
vector_add saturates DRAM at 90% while the SMs sit at 23%.
It is the canonical memory-bound kernel: 12 bytes moved per element,
1 FLOP per element, arithmetic intensity 0.08 FLOPs/byte.
Block size 32 to 1024 does not change bandwidth — DRAM is the wall.

## Bottleneck 2: L1/shared-memory-bound
reduction_naive is 1.6x slower than vector_add despite reading half
the data. The profile shows DRAM at 18% and L1/TEX at 86%. The cause:
a tree reduction with 8 __syncthreads() barriers per block across
65,536 blocks. Switching to a grid-stride loop with 320 blocks and
a shared-memory tree hits DRAM 96% — a 4.75x speedup.

Surprising result: the shared-memory variant slightly beats warp
shuffle on this workload. The lesson: reducing block count matters
more than which reduction primitive you use.

## Bottleneck 3: L1/TEX-bound on matmul
matmul_naive runs at 3.1% of T4 fp32 peak. L1/TEX at 94%, DRAM at 7%.
Every inner-loop iteration issues two redundant global loads.

Tiling with a 32x32 shared-memory block gets 2.75x. But L1/TEX is
still at 88% — because on T4, shared-memory loads share the same
port as global loads. Tiling halved global traffic (57 -> 28 GB/s)
but did not escape the L1/TEX port.

Register blocking is the next step.

## The lesson
Measure, don't guess. Every optimization in this project came from
reading the profiler, not from intuition. Three different kernels,
three different bottlenecks, three different fixes.

## Results
- reduction: 4.75x (naive -> shared)
- matmul: 2.75x (naive -> tiled)
- 6 kernels, 5 ncu profiles

## What I would do next
- Register-blocked matmul (targeting 1500+ GFLOPS)
- FP16 tensor cores (4x theoretical)
- Deploy to Jetson for edge inference
