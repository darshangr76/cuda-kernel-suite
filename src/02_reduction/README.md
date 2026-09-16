# Reduction — Naive vs Warp-Shuffle

Same operation (sum 16M floats), two implementations, two profiles.

## Results

| Metric | reduction_naive | reduction_warp | Change |
|--------|----------------|----------------|--------|
| Time (host) | 1.2242 ms | 0.2858 ms | 4.3x faster |
| Effective BW | 54.82 GB/s | 234.84 GB/s | 4.3x higher |
| Duration (ncu) | 1.60 ms | 322.46 us | 5.0x faster |
| DRAM Throughput | 18.03% | 87.09% | 4.8x higher |
| Compute (SM) | 76.33% | 14.84% | 5.1x lower |
| L1/TEX Throughput | 85.80% | 29.46% | 2.9x lower |
| L2 Throughput | 4.96% | 23.83% | 4.8x higher |
| Achieved Occupancy | 90.29% | 98.96% | slight rise |
| Registers / thread | 16 | 16 | same |
| Elapsed cycles | 936,003 | 188,622 | 5.0x fewer |
| Grid size | 65,536 blocks | 320 blocks | 205x fewer |

## Analysis

**The bottleneck flipped.** Naive was L1/shared-memory bound at 86%,
with DRAM only at 18%. Warp-shuffle inverts this: DRAM at 87%,
L1 at 29%. The kernel went from spending time shuffling data
around on-chip to streaming data from DRAM at near-peak bandwidth.

Three changes drove this:

1. **Warp shuffle replaces the shared-memory tree.**
   Naive: 8 tree steps, each doing 2 smem reads + 1 write per active
   thread, with a __syncthreads() barrier between steps.
   Warp: 5 __shfl_down_sync instructions per thread, no shared
   memory in the hot loop, no barriers until the very end.

2. **Grid-stride loop cuts the block count 205x.**
   Naive: 65,536 blocks x 8 barriers = 524,288 barrier stalls.
   Warp: 320 blocks x 1 barrier = 320 barrier stalls.
   Also 205x fewer atomicAdd operations.

3. **Coalesced grid-stride reads saturate DRAM.**
   With only 320 blocks, each thread loads multiple consecutive
   elements across the loop, keeping memory access coalesced.

## The larger lesson

The same mathematical operation — summing 16 million floats — can
hit completely different bottlenecks depending on how it is written.
Naive hits the shared-memory port; warp-shuffle hits DRAM. The
profiler shows you which, and that tells you what to optimize.

Warp-shuffle at 87% DRAM is essentially optimal for fp32 input:
each element is read exactly once, so no further speedup is possible
without reducing bytes moved (e.g., fp16 input).

## Artifacts
- profiles/reduction_naive.ncu-rep
- profiles/reduction_warp.ncu-rep
- benchmarks/results/reduction.csv
