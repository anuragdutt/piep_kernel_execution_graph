#include "cublas_kernels.hpp"
#include "benchmark_utils.hpp"
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace tier2 {

static cublasHandle_t g_cublas_handle = nullptr;

bool init_cublas() {
    if (g_cublas_handle != nullptr) {
        return true;  // Already initialized
    }
    
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle" << std::endl;
        return false;
    }
    
    std::cout << "cuBLAS initialized successfully" << std::endl;
    return true;
}

void cleanup_cublas() {
    if (g_cublas_handle != nullptr) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

cublasHandle_t get_cublas_handle() {
    if (g_cublas_handle == nullptr) {
        throw std::runtime_error("cuBLAS not initialized. Call init_cublas() first.");
    }
    return g_cublas_handle;
}

double benchmark_gemm_fp16(int M, int N, int K, bool transpose_a, bool transpose_b, int warmup_runs = 10, int timed_runs = 1) {
    cublasHandle_t handle = get_cublas_handle();
    
    // Allocate device memory
    // A is M×K (or K×M if transposed), B is K×N (or N×K if transposed), C is M×N
    __half *A_d, *B_d, *C_d;
    size_t size_A = M * K * sizeof(__half);
    size_t size_B = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(__half);
    
    CUDA_CHECK(cudaMalloc(&A_d, size_A));
    CUDA_CHECK(cudaMalloc(&B_d, size_B));
    CUDA_CHECK(cudaMalloc(&C_d, size_C));
    
    cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // Leading dimensions:
    // - For column-major (cuBLAS default): lda is the number of rows in the stored matrix
    // - If op_a = N (not transposed): A is M×K, lda = M
    // - If op_a = T (transposed): A is stored as K×M, lda = K
    // Same logic for B
    int lda = transpose_a ? K : M;
    int ldb = transpose_b ? N : K;
    int ldc = M;  // C is always M×N, ldc = M
    
    auto bench_func = [&]() {
        // Use FP32 compute for Pascal GPUs (which have poor native FP16)
        float alpha_f = 1.0f;
        float beta_f = 0.0f;
        
        cublasStatus_t status = cublasGemmEx(
            handle,
            op_a, op_b,
            M, N, K,
            &alpha_f,
            A_d, CUDA_R_16F, lda,
            B_d, CUDA_R_16F, ldb,
            &beta_f,
            C_d, CUDA_R_16F, ldc,
            CUBLAS_COMPUTE_32F,  // Use FP32 compute (faster on Pascal)
            CUBLAS_GEMM_DEFAULT
        );
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasGemmEx failed with status " + std::to_string(status));
        }
    };
    
    // Warmup + benchmark
    double time_us = benchmark::benchmark_us(bench_func, warmup_runs, timed_runs);
    
    // Cleanup
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    return time_us;
}

double benchmark_gemv_fp16(int M, int K, bool transpose, int warmup_runs = 10, int timed_runs = 1) {
    // GEMV is just GEMM with N=1
    return benchmark_gemm_fp16(M, 1, K, transpose, false, warmup_runs, timed_runs);
}

double run_tier2_kernel(const kernel::KernelSignature& sig, int num_runs) {
    const std::string& name = sig.name;
    
    // Extract M, N, K from params or use defaults
    int M = sig.get_M();
    int N = sig.get_N();
    int K = sig.get_K();
    
    // If not specified, try to infer from grid dimensions
    if (M == 0 || N == 0 || K == 0) {
        // Default BLOOM-560M dimensions for common layer sizes
        if (name.find("gemv") != std::string::npos) {
            M = 1024;
            K = 1024;
            N = 1;
        } else {
            M = 1024;
            N = 1024;
            K = 1024;
        }
    }
    
    // Parse transpose information from kernel name
    // Common patterns:
    //   _tn or _TN: first matrix Transposed, second Normal
    //   _nt or _NT: first matrix Normal, second Transposed
    //   _nn or _NN: both Normal (not transposed)
    //   _tt or _TT: both Transposed
    //   gemv2T: Transposed GEMV (matrix^T × vector)
    //   gemv2N: Normal GEMV (matrix × vector)
    bool transpose_a = false;
    bool transpose_b = false;
    
    std::string name_lower = name;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
    
    if (name_lower.find("gemv2t") != std::string::npos) {
        // GEMV with transposed matrix
        transpose_a = true;
    } else if (name_lower.find("_tn") != std::string::npos) {
        // GEMM with first matrix transposed, second normal
        transpose_a = true;
        transpose_b = false;
    } else if (name_lower.find("_nt") != std::string::npos) {
        // GEMM with first matrix normal, second transposed
        transpose_a = false;
        transpose_b = true;
    } else if (name_lower.find("_tt") != std::string::npos) {
        // GEMM with both matrices transposed
        transpose_a = true;
        transpose_b = true;
    }
    // _nn or gemv2n cases: default transpose_a=false, transpose_b=false (already set)
    
    // Warmup once (10 iters), then time num_runs iterations
    if (name.find("gemv") != std::string::npos) {
        return benchmark_gemv_fp16(M, K, transpose_a, 10, num_runs);
    } else {
        return benchmark_gemm_fp16(M, N, K, transpose_a, transpose_b, 10, num_runs);
    }
}

} // namespace tier2
