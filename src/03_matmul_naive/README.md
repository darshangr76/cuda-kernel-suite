# Matmul — Naive Baseline

## Goal
Establish a baseline for matrix multiplication and expose the cost of
no data reuse: every element of A and B is read O(N) times from memory.

## Kernel

One thread per output cell. For each cell C[row][col], sum over k of
A[row*K + k] * B[k*N + col]. No shared memory, no tiling, no reuse.

## Configuration
- M = N = K = 1024 (1024^3 multiply-adds)
- block = 16x16 = 256 threads, grid = 64x64 = 4096 blocks
- Input: all 1.0 (so expected C[i][j] = 1024.0)
- 3 warmup + 20 timed iterations
- Compile: nvcc -O3 -arch=sm_75 -lineinfo

## Results

| Metric | Value |
|--------|-------|
| Time | 8.5417 ms |
| GFLOPS | 251.41 |
| T4 FP32 peak | ~8,100 GFLOPS |
| % of peak | 3.1% |
| Correctness | PASS (max_err = 0.0) |

### Nsight Compute

| Metric | Value |
|--------|-------|
| Duration | 9.19 ms |
| DRAM Throughput | 7.30% |
| Compute (SM) | 62.46% |
| L1/TEX Throughput | 93.70% |
| L2 Throughput | 6.75% |
| Executed IPC | 0.73 inst/cycle |
| SM Busy | 20.99% |
| Roofline (fp32) | 8% of peak |

## Analysis

**The kernel is L1/TEX bound, not DRAM bound.**

This is counterintuitive: matmul at 1024^3 does 2 GFLOPs, and each
output cell requires a full row of A and column of B. You would expect
DRAM to be saturated.

Instead:
- DRAM: 7.30% (nearly idle)
- L1/TEX: 93.70% (saturated)
- SM: 62.46%

The reason: every inner-loop iteration issues **two global loads**
through the L1/TEX port — one for A[row*K+k], one for B[k*N+col].
Neither benefits from caching (working set of 4 MB per matrix far
exceeds the 128 KB L1 per SM). The L1/TEX port saturates serving
loads that mostly miss, before DRAM can be saturated.

The strided access to B (columns, not rows) also degrades coalescing,
making the L1/TEX pressure worse.

## Roofline

Nsight reports the kernel achieves 8% of the T4's fp32 peak.
Arithmetic intensity for naive matmul is ~0.25 FLOPs/byte — far below
the T4 ridge point (~25 FLOPs/byte). Well within the memory-bound
region of the roofline.

## Why this baseline matters

This is the "before" picture for tiling. The fix is:
1. Load a TILE x TILE block of A and B into shared memory
2. Reuse each loaded value TILE times
3. Reduce global loads from O(N^3) to O(N^2)

Expected improvement: 5-10x (targeting ~1,500 GFLOPS in Week 4).

## Artifacts
- profiles/matmul_naive.ncu-rep
- benchmarks/results/matmul.csv
