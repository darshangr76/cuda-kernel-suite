#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void reduceNaive(const float* __restrict__ in,
                            float* __restrict__ out,
                            int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
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
    const int grid  = (N + block - 1) / block;

    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_out, 0, sizeof(float));
        reduceNaive<<<grid, block, block * sizeof(float)>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 50; ++i) {
        cudaMemset(d_out, 0, sizeof(float));
        reduceNaive<<<grid, block, block * sizeof(float)>>>(d_in, d_out, N);
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
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        if (sz == 0) {
            fprintf(f, "kernel,N,block_size,time_ms,bandwidth_gbps\n");
        }
        fprintf(f, "reduction_naive,%d,%d,%.4f,%.2f\n", N, block, ms, gbps);
        fclose(f);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
