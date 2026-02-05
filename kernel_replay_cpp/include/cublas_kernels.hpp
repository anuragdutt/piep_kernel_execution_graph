#pragma once

#include "kernel_registry.hpp"
#include <cublas_v2.h>

namespace tier2 {

/**
 * Tier 2: Direct cuBLAS API calls
 * 
 * Handles: cublasGemmEx, cublasSgemv (23 unique kernels, ~3700 invocations)
 */

/**
 * Initialize cuBLAS handle (call once at startup)
 */
bool init_cublas();

/**
 * Cleanup cuBLAS handle (call at shutdown)
 */
void cleanup_cublas();

/**
 * Get cuBLAS handle (initialized by init_cublas)
 */
cublasHandle_t get_cublas_handle();

/**
 * Benchmark FP16 GEMM: C = alpha*A*B + beta*C
 * 
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A, rows of B
 * @param transpose_a Whether to transpose A (default: false)
 * @param transpose_b Whether to transpose B (default: false)
 * @return Time in microseconds
 */
double benchmark_gemm_fp16(int M, int N, int K, 
                          bool transpose_a = false, 
                          bool transpose_b = false);

/**
 * Benchmark FP16 GEMV: y = alpha*A*x + beta*y
 * 
 * Implemented as GEMM with N=1
 * 
 * @param M Rows of A
 * @param K Columns of A (length of x)
 * @param transpose Whether to transpose A (default: false)
 * @return Time in microseconds
 */
double benchmark_gemv_fp16(int M, int K, bool transpose = false);

/**
 * Dispatch and benchmark a Tier 2 kernel based on its signature
 * 
 * Extracts M, N, K from signature and calls appropriate cuBLAS function
 * 
 * @param sig Kernel signature from registry
 * @return Time in microseconds
 */
double run_tier2_kernel(const kernel::KernelSignature& sig, int num_runs = 1);

} // namespace tier2
