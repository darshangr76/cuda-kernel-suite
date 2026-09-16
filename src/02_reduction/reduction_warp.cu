#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Warp-shuffle reduction.
//
// Strategy:
//   1. Each thread loads one element (grid-stride so N can exceed grid*block)
//   2. Each WARP reduces its 32 values using __shfl_down_sync — no shared mem
//   3. Warp leaders write partial sums to shared memory (only 8 per block)
//   4. One warp finishes the final 8-value reduction with shuffle
//   5. Thread 0 does one atomicAdd per block
//
// Compared to naive:
//   - Shared memory: 8 floats/block instead of 256
//   - Barriers: 1 __syncthreads() instead of 8
//   - No divergent tree steps
// ---------------------------------------------------------------------------

__inline__ __device__ float warpReduceSum(float val) {
    // Reduce 32 lanes to 1 using shuffle; final value in lane 0
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void reduceWarp(const float* __restrict__ in,
                           float* __restrict__ out,
                           int N) {
    __shared__ float warpSums[32];   // up to 32 warps per block

    int tid      = threadIdx.x;
    int lane     = tid & 31;         // lane within warp
    int warpId   = tid >> 5;         // warp index within block
    int nWarps   = blockDim.x >> 5;

    // 1. Grid-stride load + accumulate per-thread partial
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N;
         i += gridDim.x * blockDim.x) {
        sum += in[i];
    }

    // 2. Intra-warp reduction via shuffle
    sum = warpReduceSum(sum);

    // 3. Lane 0 of each warp writes partial to shared memory
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // 4. First warp reduces the per-warp partials
    if (warpId == 0) {
        sum = (lane < nWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);

        // 5. Thread 0 of block does one atomicAdd
        if (lane == 0) {
            atomicAdd(out, sum);
        }
    }
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
    const size_t bytes = (size_t)N * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const int block = 256;
    // Use fewer blocks than naive — grid-stride handles the work.
    // Pick enough to saturate: T4 has 40 SMs, aim for ~8 blocks/SM.
    int grid = 320;
    if ((long)grid * block > N) grid = (N + block - 1) / block;

    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_out, 0, sizeof(float));
        reduceWarp<<<grid, block>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 50; ++i) {
        cudaMemset(d_out, 0, sizeof(float));
        reduceWarp<<<grid, block>>>(d_in, d_out, N);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    ms /= 50.0f;

    float h_out = 0;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    double gbps = (double)bytes / (ms * 1e-3) / 1e9;
    float expected = (float)N;

    printf("N=%d block=%d grid=%d\n", N, block, grid);
    printf("time=%.4f ms  BW=%.2f GB/s\n", ms, gbps);
    printf("result=%.0f  expected=%.0f  %s\n",
           h_out, expected,
           (fabsf(h_out - expected) < 1e-2f * expected) ? "PASS" : "FAIL");

    FILE* f = fopen("benchmarks/results/reduction.csv", "a");
    if (f) {
        fprintf(f, "reduction_warp,%d,%d,%.4f,%.2f\n", N, block, ms, gbps);
        fclose(f);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
