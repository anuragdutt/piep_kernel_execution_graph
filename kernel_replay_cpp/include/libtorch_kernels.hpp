#pragma once

#include "kernel_registry.hpp"
#include <torch/torch.h>

namespace tier3 {

/**
 * Tier 3: libtorch Fallback
 * 
 * Handles: torch::layer_norm, torch::softmax, torch::add, etc.
 * (53 unique kernels, ~12000 invocations)
 */

/**
 * Benchmark layer_norm operation
 * 
 * @param shape Input tensor shape
 * @param norm_dim Dimension to normalize over
 * @return Time in microseconds
 */
double benchmark_layer_norm(const std::vector<int64_t>& shape, int64_t norm_dim);

/**
 * Benchmark softmax operation
 * 
 * @param shape Input tensor shape
 * @param dim Dimension to apply softmax
 * @return Time in microseconds
 */
double benchmark_softmax(const std::vector<int64_t>& shape, int dim);

/**
 * Benchmark elementwise add operation
 * 
 * @param shape Tensor shape
 * @return Time in microseconds
 */
double benchmark_add(const std::vector<int64_t>& shape);

/**
 * Benchmark elementwise mul operation
 * 
 * @param shape Tensor shape
 * @return Time in microseconds
 */
double benchmark_mul(const std::vector<int64_t>& shape);

/**
 * Benchmark fill operation
 * 
 * @param shape Tensor shape
 * @param value Value to fill
 * @return Time in microseconds
 */
double benchmark_fill(const std::vector<int64_t>& shape, float value = 0.0);

/**
 * Benchmark index_select operation
 * 
 * @param shape Input tensor shape
 * @param dim Dimension to select from
 * @param index_size Number of indices
 * @return Time in microseconds
 */
double benchmark_index_select(const std::vector<int64_t>& shape, int dim, int index_size);

/**
 * Benchmark gelu activation
 * 
 * @param shape Tensor shape
 * @return Time in microseconds
 */
double benchmark_gelu(const std::vector<int64_t>& shape);

/**
 * Benchmark reduce operation (sum)
 * 
 * @param shape Input tensor shape
 * @param dim Dimension to reduce
 * @return Time in microseconds
 */
double benchmark_reduce(const std::vector<int64_t>& shape, int dim);

/**
 * Benchmark generic elementwise operation
 * 
 * @param shape Tensor shape
 * @return Time in microseconds
 */
double benchmark_elementwise(const std::vector<int64_t>& shape);

/**
 * Dispatch and benchmark a Tier 3 kernel based on its signature
 * 
 * Uses pattern matching on kernel name to determine operation type,
 * then calls appropriate libtorch function
 * 
 * @param sig Kernel signature from registry
 * @return Time in microseconds
 */
double run_tier3_kernel(const kernel::KernelSignature& sig, int num_runs = 1);

/**
 * Infer tensor shape from grid/block dimensions (approximate)
 * 
 * Many kernels don't have explicit shape info, so we estimate
 * based on grid dimensions
 */
std::vector<int64_t> infer_shape_from_grid(
    const std::vector<int>& grid,
    const std::vector<int>& block
);

} // namespace tier3
