#include "cuda_kernels.hpp"
#include "benchmark_utils.hpp"
#include <iostream>

namespace tier1 {

cudaMemcpyKind parse_memcpy_kind(const std::string& kind) {
    if (kind == "HtoD") return cudaMemcpyHostToDevice;
    if (kind == "DtoH") return cudaMemcpyDeviceToHost;
    if (kind == "DtoD") return cudaMemcpyDeviceToDevice;
    
    // Default to DtoD
    return cudaMemcpyDeviceToDevice;
}

double benchmark_memcpy(size_t bytes, cudaMemcpyKind kind, int num_runs = 1) {
    void *src = nullptr, *dst = nullptr;
    void *src_host = nullptr, *dst_host = nullptr;
    
    // Allocate based on kind
    if (kind == cudaMemcpyHostToDevice) {
        src_host = malloc(bytes);
        CUDA_CHECK(cudaMalloc(&dst, bytes));
    } else if (kind == cudaMemcpyDeviceToHost) {
        CUDA_CHECK(cudaMalloc(&src, bytes));
        dst_host = malloc(bytes);
    } else {  // DtoD
        CUDA_CHECK(cudaMalloc(&src, bytes));
        CUDA_CHECK(cudaMalloc(&dst, bytes));
    }
    
    auto bench_func = [&]() {
        if (kind == cudaMemcpyHostToDevice) {
            CUDA_CHECK(cudaMemcpy(dst, src_host, bytes, kind));
        } else if (kind == cudaMemcpyDeviceToHost) {
            CUDA_CHECK(cudaMemcpy(dst_host, src, bytes, kind));
        } else {
            CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
        }
    };
    
    // Warmup once, then time num_runs iterations
    double time_us = benchmark::benchmark_us(bench_func, 10, num_runs);
    
    // Cleanup
    if (src) cudaFree(src);
    if (dst) cudaFree(dst);
    if (src_host) free(src_host);
    if (dst_host) free(dst_host);
    
    return time_us;
}

double benchmark_memset(size_t bytes, int value, int num_runs = 1) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    
    auto bench_func = [&]() {
        CUDA_CHECK(cudaMemset(ptr, value, bytes));
    };
    
    // Warmup once, then time num_runs iterations
    double time_us = benchmark::benchmark_us(bench_func, 10, num_runs);
    
    // Cleanup
    cudaFree(ptr);
    
    return time_us;
}

double run_tier1_kernel(const kernel::KernelSignature& sig, int num_runs) {
    const std::string& name = sig.name;
    
    // Determine operation type
    if (name.find("Memcpy") != std::string::npos || 
        name.find("memcpy") != std::string::npos) {
        
        size_t bytes = sig.get_bytes();
        if (bytes == 0) bytes = 1024;  // Default if not specified
        
        cudaMemcpyKind kind = parse_memcpy_kind(sig.get_memcpy_kind());
        return benchmark_memcpy(bytes, kind, num_runs);
        
    } else if (name.find("Memset") != std::string::npos || 
               name.find("memset") != std::string::npos) {
        
        size_t bytes = sig.get_bytes();
        if (bytes == 0) bytes = 1024;  // Default if not specified
        
        return benchmark_memset(bytes, 0, num_runs);
    }
    
    static int warning_count = 0;
    if (warning_count == 0) {
        std::cerr << "Warning: Unknown Tier 1 kernel type (further warnings suppressed)" << std::endl;
        warning_count++;
    }
    return -1.0;
}

} // namespace tier1
