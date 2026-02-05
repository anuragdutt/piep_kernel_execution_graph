#pragma once

#include "kernel_registry.hpp"
#include <cuda_runtime.h>

namespace tier1 {

/**
 * Tier 1: Direct CUDA Runtime API calls
 * 
 * Handles: cudaMemcpy, cudaMemset (68 unique kernels, ~1300 invocations)
 */

/**
 * Benchmark cudaMemcpy with specified bytes and kind
 * 
 * @param bytes Number of bytes to copy
 * @param kind cudaMemcpyKind (HtoD, DtoH, DtoD)
 * @return Time in microseconds
 */
double benchmark_memcpy(size_t bytes, cudaMemcpyKind kind);

/**
 * Benchmark cudaMemset with specified bytes
 * 
 * @param bytes Number of bytes to set
 * @param value Value to set (default: 0)
 * @return Time in microseconds
 */
double benchmark_memset(size_t bytes, int value = 0);

/**
 * Dispatch and benchmark a Tier 1 kernel based on its signature
 * 
 * @param sig Kernel signature from registry
 * @return Time in microseconds
 */
double run_tier1_kernel(const kernel::KernelSignature& sig, int num_runs = 1);

/**
 * Parse memcpy kind string to cudaMemcpyKind enum
 */
cudaMemcpyKind parse_memcpy_kind(const std::string& kind);

} // namespace tier1
