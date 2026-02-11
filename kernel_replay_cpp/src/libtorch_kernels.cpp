#include "libtorch_kernels.hpp"
#include "benchmark_utils.hpp"
#include <iostream>

namespace tier3 {

std::vector<int64_t> infer_shape_from_grid(
    const std::vector<int>& grid,
    const std::vector<int>& block,
    const std::string& operation
) {
    // Infer approximate tensor shape from grid/block dimensions
    // Different operations have different shape requirements
    
    int64_t total_threads = (int64_t)grid[0] * grid[1] * grid[2] * 
                           block[0] * block[1] * block[2];
    
    // BLOOM-560M common tensor sizes:
    // - [1, 5, 1024]:  input embeddings (batch=1, seq=5, hidden=1024)
    // - [1, 5, 4096]:  FFN intermediate (batch=1, seq=5, ffn=4096)
    // - [1, 5, 3072]:  QKV projection (batch=1, seq=5, 3*hidden)
    
    int batch = 1;
    int seq_len = 5;
    int64_t feature_dim = 1024;  // Default to hidden size
    
    // For layer_norm, softmax, etc.: must use proper feature dimensions
    if (operation == "layer_norm") {
        // Layer norm always normalizes over hidden_dim=1024
        return {batch, seq_len, 1024};
    } else if (operation == "softmax") {
        // Softmax over attention scores: [batch, num_heads, seq, seq] or similar
        // For BLOOM-560M with 16 heads: simplified to [1, 5, 1024]
        return {batch, seq_len, 1024};
    } else {
        // For elementwise operations, infer from grid size
        if (total_threads <= 5120) {
            // Small kernels: ~1×5×1024 or smaller, but keep multiples of common sizes
            if (total_threads <= 512) {
                feature_dim = 128;
            } else if (total_threads <= 2560) {
                feature_dim = 512;
            } else {
                feature_dim = 1024;
            }
        } else if (total_threads <= 15360) {
            // Medium kernels: ~1×5×3072
            feature_dim = 3072;
        } else if (total_threads <= 20480) {
            // Large kernels: ~1×5×4096
            feature_dim = 4096;
        } else {
            // Very large kernels: estimate but cap at reasonable size
            feature_dim = std::min((int64_t)8192, total_threads / (batch * seq_len));
        }
    }
    
    return {batch, seq_len, feature_dim};
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
    
    // Use the num_runs passed from aggregation (calibrated for system power meter at ~1 Hz)
    // No cap - we need long durations for accurate power measurement
    int iters = num_runs;
    
    // Infer shape from grid/block, considering operation type
    std::vector<int64_t> shape = infer_shape_from_grid(sig.grid, sig.block, operation);
    
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
