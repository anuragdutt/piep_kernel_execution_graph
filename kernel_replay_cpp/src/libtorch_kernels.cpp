#include "libtorch_kernels.hpp"
#include "benchmark_utils.hpp"
#include <iostream>

namespace tier3 {

std::vector<int64_t> infer_shape_from_grid(
    const std::vector<int>& grid,
    const std::vector<int>& block
) {
    // Approximate shape inference from grid dimensions
    // This is a rough heuristic - actual shapes should come from bloom_shapes.jsonl
    int64_t total_threads = (int64_t)grid[0] * grid[1] * grid[2] * 
                           block[0] * block[1] * block[2];
    
    // Default to 2D shape for most operations
    // BLOOM-560M uses [batch, seq_len, hidden_size] = [1, 5, 1024]
    return {1, 5, 1024};  // Default BLOOM shape
}

double benchmark_layer_norm(const std::vector<int64_t>& shape, int64_t norm_dim, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto x = torch::randn(shape, opts);
    auto gamma = torch::ones({norm_dim}, opts);
    auto beta = torch::zeros({norm_dim}, opts);
    
    auto bench_func = [&]() {
        torch::layer_norm(x, {norm_dim}, gamma, beta);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_softmax(const std::vector<int64_t>& shape, int dim, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto x = torch::randn(shape, opts);
    
    auto bench_func = [&]() {
        torch::softmax(x, dim);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_add(const std::vector<int64_t>& shape, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto a = torch::randn(shape, opts);
    auto b = torch::randn(shape, opts);
    
    auto bench_func = [&]() {
        torch::add(a, b);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_mul(const std::vector<int64_t>& shape, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto a = torch::randn(shape, opts);
    auto b = torch::randn(shape, opts);
    
    auto bench_func = [&]() {
        torch::mul(a, b);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_fill(const std::vector<int64_t>& shape, float value, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto x = torch::randn(shape, opts);
    
    auto bench_func = [&]() {
        x.fill_(value);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_index_select(const std::vector<int64_t>& shape, int dim, int index_size, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto x = torch::randn(shape, opts);
    auto indices = torch::randint(0, shape[dim], {index_size}, 
                                 torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    
    auto bench_func = [&]() {
        torch::index_select(x, dim, indices);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_gelu(const std::vector<int64_t>& shape, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto x = torch::randn(shape, opts);
    
    auto bench_func = [&]() {
        torch::gelu(x);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_reduce(const std::vector<int64_t>& shape, int dim, int num_iters) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto x = torch::randn(shape, opts);
    
    auto bench_func = [&]() {
        torch::sum(x, dim);
    };
    
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_scan(const std::vector<int64_t>& shape, int num_iters) {
    // CUB exclusive scan - approximate with cumsum
    auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
    auto x = torch::randint(0, 10, shape, opts);
    auto bench_func = [&]() {
        torch::cumsum(x, -1);
    };
    return benchmark::benchmark_us(bench_func, 10, num_iters);
}

double benchmark_elementwise(const std::vector<int64_t>& shape, int num_iters) {
    // Generic elementwise - use add as default
    return benchmark_add(shape, num_iters);
}

double run_tier3_kernel(const kernel::KernelSignature& sig, int num_runs) {
    const std::string& name = sig.name;
    std::string operation = sig.get_operation();
    
    // Run many iterations so power logger (nvidia-smi ~25 Hz) gets 2+ samples per kernel.
    // Cap at 100k to avoid excessive runtime; 100k * ~10us = ~1s per kernel.
    int iters = std::min(std::max(num_runs, 1000), 100000);
    
    // Infer shape from grid/block
    std::vector<int64_t> shape = infer_shape_from_grid(sig.grid, sig.block);
    
    // Dispatch based on operation type
    if (operation == "layer_norm") {
        return benchmark_layer_norm(shape, 1024, iters);  // BLOOM norm_dim
    } 
    else if (operation == "softmax") {
        return benchmark_softmax(shape, -1, iters);  // Last dimension
    } 
    else if (operation == "add") {
        return benchmark_add(shape, iters);
    } 
    else if (operation == "mul") {
        return benchmark_mul(shape, iters);
    } 
    else if (operation == "fill") {
        return benchmark_fill(shape, 0.0, iters);
    } 
    else if (operation == "index_select") {
        return benchmark_index_select(shape, 0, 5, iters);  // BLOOM seq_len
    } 
    else if (operation == "gelu") {
        return benchmark_gelu(shape, iters);
    } 
    else if (operation == "reduce") {
        return benchmark_reduce(shape, -1, iters);
    } 
    else if (operation == "scan") {
        return benchmark_scan(shape, iters);
    } 
    else if (operation == "elementwise") {
        return benchmark_elementwise(shape, iters);
    }
    else {
        // Unknown operation - use generic elementwise as fallback
        static int warning_count = 0;
        if (warning_count < 3) {
            std::cerr << "Warning: Unknown operation '" << operation 
                      << "' (using elementwise fallback)" << std::endl;
            warning_count++;
            if (warning_count == 3) {
                std::cerr << "  (further warnings suppressed)" << std::endl;
            }
        }
        return benchmark_elementwise(shape, iters);
    }
}

} // namespace tier3
