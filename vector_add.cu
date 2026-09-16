#include <cstdio>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = (size_t)N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes), *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int block = 256, grid = (N + block - 1) / block;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 5; i++) vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int i = 0; i < 50; i++) vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);

    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 50.0f;
    double gbps = (3.0 * bytes) / (ms * 1e-3) / 1e9;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("N=%d block=%d time=%.4f ms BW=%.2f GB/s sample=%.1f\n",
           N, block, ms, gbps, h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
