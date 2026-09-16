#include <cstdio>
#include <cstdlib>
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

    int blocks[] = {32, 64, 128, 256, 512, 1024};
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    // print header for CSV
    printf("kernel,N,block_size,time_ms,bandwidth_gbps\n");

    for (int bi = 0; bi < 6; bi++) {
        int block = blocks[bi];
        int grid = (N + block - 1) / block;

        for (int i = 0; i < 5; i++) vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaEventRecord(s);
        for (int i = 0; i < 50; i++) vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e); cudaEventSynchronize(e);

        float ms; cudaEventElapsedTime(&ms, s, e); ms /= 50.0f;
        double gbps = (3.0 * bytes) / (ms * 1e-3) / 1e9;

        printf("vector_add,%d,%d,%.4f,%.2f\n", N, block, ms, gbps);
        fflush(stdout);
    }

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i += N/1000) if (fabsf(h_C[i]-3.0f) > 1e-5f) { ok = false; break; }
    fprintf(stderr, "correctness: %s\n", ok ? "PASS" : "FAIL");
    return 0;
}
