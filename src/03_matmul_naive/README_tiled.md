# Matmul — Naive vs Tiled

Same operation (1024^3 matmul), two implementations, two profiles.

## Results

| Metric | matmul_naive | matmul_tiled (TILE=32) | Change |
|--------|-------------|------------------------|--------|
| Time (host) | 8.5417 ms | 3.1068 ms | 2.75x faster |
| GFLOPS | 251.41 | 691.21 | 2.75x higher |
| Duration (ncu) | 9.19 ms | 5.33 ms | 1.72x faster |
| DRAM Throughput | 7.30% | 8.86% | similar |
| Compute (SM) | 62.46% | 74.20% | higher |
| L1/TEX Throughput | 93.70% | 87.98% | slight drop |
| L2 Throughput | 6.75% | 5.85% | lower |
| Memory Throughput | 57.63 GB/s | 28.31 GB/s | halved |
| Roofline (fp32) | 8% of peak | 13% of peak | +5pts |

## Analysis

**Tiling halved global memory traffic but did NOT fix the L1/TEX bottleneck.**

The naive kernel was L1/TEX bound at 93.7% because every inner-loop
iteration issued two redundant global loads. Tiling replaces those with
two shared-memory loads per FMA — but on the T4 (and all NVIDIA GPUs
since Volta), shared memory and global memory loads share the same
L1/TEX port.

So the bottleneck moved: from "redundant global loads through L1/TEX"
to "necessary shared-memory loads through L1/TEX." Same physical port,
similar utilization (87.98%).

Evidence:
- Memory Throughput dropped from 57.63 GB/s to 28.31 GB/s (halved).
  Tiling reduced global memory traffic as intended.
- But L1/TEX Throughput barely moved: 93.70% -> 87.98%.
- The kernel is still L1/TEX bound.

## What the fix would be

To reduce L1/TEX pressure further, use register blocking: each thread
computes a 2x2 (or larger) sub-tile of outputs, so each shared-memory
load is reused in registers multiple times. This reduces smem reads per
FMA by up to the register-block factor.

That is the matmul_tiled_2x variant (bonus, not implemented here).

## Artifacts
- profiles/matmul_naive.ncu-rep
- profiles/matmul_tiled.ncu-rep
- benchmarks/results/matmul.csv
