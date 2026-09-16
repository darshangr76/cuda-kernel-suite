#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Naive matmul: one thread per output element.
//
// C[M][N] = A[M][K] * B[K][N]
//
// Deliberately unoptimized:
//   - Each thread reads its A row and B column from DRAM
//   - No shared memory, no data reuse
//   - O(N^3) DRAM reads for an N*N matmul
//
// This is the baseline that motivates tiling.
// ---------------------------------------------------------------------------

#define TILE 16   // unused in naive; placeholder for later

__global__ void matmulNaive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// --- Host-side CPU reference for correctness -------------------------------
void matmulCPU(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    const int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int N = M;
    const int K = M;

    printf("Matmul: M=%d N=%d K=%d\n", M, N, K);

    // Host buffers
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);
    float* h_Cref = (float*)malloc(bytesC);

    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = 1.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = 1.0f;

    // Device buffers
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    // Launch config
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        matmulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Timed
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    const int iters = 20;
    for (int i = 0; i < iters; ++i) {
        matmulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    ms /= iters;

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    // CPU reference
    matmulCPU(h_A, h_B, h_Cref, M, N, K);

    // Compare (sample)
    bool ok = true;
    float max_err = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float err = fabsf(h_C[i] - h_Cref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) { ok = false; break; }
    }

    // Compute GFLOPS: 2 * M * N * K per matmul
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / (ms * 1e-3) / 1e9;

    printf("time=%.4f ms  GFLOPS=%.2f  max_err=%.6f  %s\n",
           ms, gflops, max_err, ok ? "PASS" : "FAIL");

    // CSV row
    FILE* f = fopen("benchmarks/results/matmul.csv", "a");
    if (f) {
        fseek(f, 0, SEEK_END);
        if (ftell(f) == 0)
            fprintf(f, "kernel,M,N,K,time_ms,GFLOPS\n");
        fprintf(f, "matmul_naive,%d,%d,%d,%.4f,%.2f\n",
                M, N, K, ms, gflops);
        fclose(f);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_Cref);
    return 0;
}
