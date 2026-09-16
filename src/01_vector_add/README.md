# Vector Add — CUDA Kernel Baseline

## Goal
Establish a baseline streaming kernel and verify it is memory-bandwidth-bound
using Nsight Compute on a Tesla T4 (sm_75).

## Kernel

```cuda
__global__ void vectorAdd(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

## Configuration
- N = 16,777,216 floats (64 MB per array)
- block size = 256, grid = 65,536
- 5 warmup + 50 timed iterations
- Compile: nvcc -O3 -arch=sm_75 -lineinfo

## Results

| Metric | Value | % of Peak |
|--------|-------|-----------|
| Time (host) | 0.7672 ms | - |
| Effective bandwidth | 262.41 GB/s | 82% of 320 GB/s |
| DRAM Throughput (Nsight) | - | 90.38% |
| Compute (SM) Throughput | - | 23.44% |
| Achieved Occupancy | - | 82.25% |
| Registers / thread | 16 | - |

### Block-size sweep

| block size | bandwidth (GB/s) |
|-----------:|----------------:|
| 32 | 174.31 |
| 64 | 261.20 |
| 128 | 258.54 |
| 256 | 259.25 |
| 512 | 259.20 |
| 1024 | 258.33 |

Bandwidth saturates for block size >= 64. At 32 threads per block
(one warp per block), the scheduler cannot hide DRAM latency, and
bandwidth drops to 174 GB/s.

## Analysis

The kernel is memory-bound. Nsight Compute reports 90.38% DRAM
throughput against only 23.44% SM throughput — a ratio of roughly
4:1 characteristic of a streaming kernel with arithmetic intensity
about 0.08 FLOPs/byte.

- 16 registers/thread and 100% theoretical occupancy confirm no
  register or shared-memory pressure limits the kernel.
- Achieved occupancy (82.25%) is below theoretical because the memory
  subsystem is saturated — warps stall on DRAM loads faster than
  the scheduler can hide the latency.
- Conclusion: further optimization is not possible without reducing
  bytes moved. This is the correct baseline for all subsequent kernels.

## Artifacts
- profiles/vecadd_256.ncu-rep — full Nsight Compute report
- benchmarks/results/vector_add.csv — block-size sweep
- plots/vector_add_bw.png — bandwidth vs block size
