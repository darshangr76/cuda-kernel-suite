#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Tiled matmul: block loads a TILE x TILE chunk of A and B into shared
// memory, then each thread computes one output cell reusing those values.
//
// C[M][N] = A[M][K] * B[K][N]
//
// Fixes the naive bottleneck: global loads drop from O(N^3) to O(N^2)
// ---------------------------------------------------------------------------

#define TILE 32   // 32x32 tile = 4 KB per tile (fits 48 KB smem easily)

__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int tiledK = t * TILE;

        // Cooperative load of A[row][tiledK + threadIdx.x] into As
        if (row < M && tiledK + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledK + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Cooperative load of B[tiledK + threadIdx.y][col] into Bs
        if (tiledK + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(tiledK + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// --- Host CPU reference ----------------------------------------------------
void matmulCPU(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main(int argc, char** argv) {
    const int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int N = M;
    const int K = M;

    printf("Matmul tiled: M=%d N=%d K=%d TILE=%d\n", M, N, K, TILE);

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);
    float* h_Cref = (float*)malloc(bytesC);

    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = 1.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    for (int i = 0; i < 3; ++i)
        matmulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    const int iters = 50;
    for (int i = 0; i < iters; ++i)
        matmulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    ms /= iters;

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    matmulCPU(h_A, h_B, h_Cref, M, N, K);

    bool ok = true;
    float max_err = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float err = fabsf(h_C[i] - h_Cref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) { ok = false; break; }
    }

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e-3) / 1e9;

    printf("time=%.4f ms  GFLOPS=%.2f  max_err=%.6f  %s\n",
           ms, gflops, max_err, ok ? "PASS" : "FAIL");

    FILE* f = fopen("benchmarks/results/matmul.csv", "a");
    if (f) {
        fprintf(f, "matmul_tiled_%d,%d,%d,%d,%.4f,%.2f\n",
                TILE, M, N, K, ms, gflops);
        fclose(f);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_Cref);
    return 0;
}
