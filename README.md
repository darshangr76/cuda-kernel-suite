# 🚀 CUDA Kernel Optimization Suite

A from-scratch implementation of custom CUDA kernels (Vector Addition, Reduction, Matrix Multiplication) designed to benchmark and optimize GPU performance against standard CPU baselines.

## 📊 Key Achievements
*   **4.75x Speedup** on Reduction using Shared-Memory Tiling.
*   **2.75x Speedup** on Matrix Multiplication.
*   Profiled memory bottlenecks using **Nsight Compute** (identified DRAM-bound, L1/Shared-Memory bound, and L1/TEX-bound bottlenecks).

## 🛠️ Tech Stack
*   **Language:** C++ / CUDA C++
*   **Tools:** Nsight Compute (for profiling), NVIDIA GPU (Tesla T4 / L40S)

## 📂 Kernels Implemented
1.  **Vector Addition:** Standard element-wise addition.
2.  **Reduction:** Summing an array (optimized with shared memory to avoid warp divergence).
3.  **Matrix Multiplication (MatMul):** Optimized using shared-memory tiling to reduce global memory accesses.

## 🚀 How to Run
[Insert the exact commands you use to compile and run these kernels, e.g., `nvcc vector_add.cu -o vector_add`, etc.]

## 📈 Profiling Results
[Include screenshots or links to your Nsight Compute profiles here].
