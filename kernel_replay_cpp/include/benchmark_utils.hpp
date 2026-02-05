#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>
#include <string>

// CUDA error checking macro (must be outside namespace)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " - " + \
                                   cudaGetErrorString(err)); \
        } \
    } while(0)

namespace benchmark {

/**
 * RAII wrapper for CUDA events
 */
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }
    
    // Returns elapsed time in microseconds
    double elapsed_us() const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms * 1000.0;
    }
    
    // Returns elapsed time in milliseconds
    double elapsed_ms() const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

/**
 * CPU timer for host-side measurements
 */
class CpuTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_us() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return duration.count();
    }
    
    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_ - start_);
        return duration.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_, stop_;
};

/**
 * Warmup helper - runs a lambda multiple times to warm up GPU
 */
template<typename Func>
void warmup(Func&& func, int iterations = 10) {
    for (int i = 0; i < iterations; i++) {
        func();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Benchmark helper - times a lambda with warmup
 */
template<typename Func>
double benchmark_us(Func&& func, int warmup_iters = 10, int timed_iters = 1) {
    // Warmup
    warmup(func, warmup_iters);
    
    // Timed run
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < timed_iters; i++) {
        func();
    }
    timer.stop();
    
    return timer.elapsed_us() / timed_iters;
}

} // namespace benchmark
