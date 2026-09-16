# Findings — cuda-kernel-suite

Running notes on surprising results across the project.

## vector_add
- Achieves 90% DRAM throughput at 0.77 ms. Block size 32-1024 does not
  change bandwidth — DRAM is the wall, not scheduling.
- Occupancy 82% is limited by DRAM saturation, not by register or
  shared-memory pressure.

## reduction_naive
- Slower than vector_add despite reading half the data. Profile shows
  L1/TEX at 86%, DRAM at 18%.
- Cause: tree reduction with 8 __syncthreads() barriers per block
  across 65,536 blocks.

## reduction_shared
- Grid-stride loop + 320 blocks (instead of 65,536) + shared-memory tree
  reaches DRAM 96.06% — essentially peak.
- Interesting: slightly beats warp shuffle on this workload.
- Lesson: reducing block count matters more than the reduction method.

## reduction_warp
- 4.28x faster than naive via warp shuffle.
- DRAM 87% — good, but not as good as shared variant.
- Extra shuffle overhead doesn't pay off at this size.

## matmul_naive
- 3.1% of T4 fp32 peak. L1/TEX bound at 94%.
- Every inner-loop iteration issues two redundant global loads.

## matmul_tiled (TILE=32)
- 2.75x faster than naive.
- Global memory traffic halved (57 -> 28 GB/s).
- L1/TEX still 88% because shared-memory loads share the same port
  as global loads on T4.
- Register blocking is the next optimization.
