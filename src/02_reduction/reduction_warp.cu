#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>



__inline__ __device__ float warpReduceSum(float val) {
    
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

   
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N;
         i += gridDim.x * blockDim.x) {
        sum += in[i];
    }

    
    sum = warpReduceSum(sum);

    
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

   
    if (warpId == 0) {
        sum = (lane < nWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);

     
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
