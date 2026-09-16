# cuda-kernel-suite

A CUDA kernel optimization suite — six kernels across three families
(vector add, reduction, matmul), each profiled with Nsight Compute to
identify its bottleneck and measured for speedup.

Target: Tesla T4 (sm_75) | CUDA 12.8 | Nsight Compute 2025.1.1
Environment: Google Colab (GPU runtime)

## Results at a glance

| Kernel | Time | Best metric | Bottleneck |
|--------|------|-------------|------------|
| vector_add | 0.77 ms | 262 GB/s, 90% DRAM | DRAM |
| reduction_naive | 1.22 ms | 54 GB/s, 86% L1 | L1/shared mem |
| reduction_shared | 0.26 ms | 260 GB/s, 96% DRAM | DRAM (saturated) |
| reduction_warp | 0.29 ms | 235 GB/s, 87% DRAM | DRAM |
| matmul_naive | 8.54 ms | 251 GFLOPS, 94% L1 | L1/TEX |
| matmul_tiled_32 | 3.11 ms | 691 GFLOPS, 88% L1 | L1/TEX (improved) |

## Speedups achieved

- Reduction: **4.75x** (naive 1.22 ms -> shared 0.26 ms)
- Matmul: **2.75x** (naive 8.54 ms -> tiled 3.11 ms)

## Kernels

1. **vector_add** — memory-bound baseline
2. **reduction** — naive tree, shared-memory grid-stride, warp shuffle
3. **matmul** — naive (one thread per output), tiled (shared-memory 32x32)

## Key findings

- Vector add is DRAM-bound at 90% — the canonical memory-bound kernel.
- Naive reduction is L1-bound at 86% due to shared-memory tree
  reduction across 65,536 blocks; the fix is grid-stride + fewer blocks.
- Tiled matmul halved global memory traffic but stayed L1/TEX bound at
  88% because shared-memory loads share the same port as global loads.
  Register blocking would be the next step.
- Three bottleneck types appeared across the project: DRAM-bound,
  L1/shared-memory-bound, and L1/TEX-bound. Every kernel was diagnosed
  by Nsight Compute, not guessed.

## Repository layout

- src/               kernel sources + per-kernel READMEs
- benchmarks/        CSV outputs from runs
- profiles/          Nsight Compute .ncu-rep reports
- plots/             speedup/bandwidth charts
- docs/              methodology and findings

## How to reproduce

See src/*/README.md for per-kernel details.
Compile: nvcc -O3 -arch=sm_75 -lineinfo -o kernel kernel.cu
Profile: ncu --set full -c 1 -o profile ./kernel
