/**
 * Real CUDA Runtime API replay for cuda_api kernels.
 * Runs the actual API (cudaEventRecord, cudaEventQuery, cudaStreamWaitEvent,
 * cudaLaunchKernel, etc.) in a loop so system power captures CPU/driver cost.
 * Uses libtorch for minimal kernel launches (fill) where needed; no .cu file.
 */
#include "cuda_api_replay.hpp"
#include "benchmark_utils.hpp"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <unordered_set>
#include <iostream>

namespace {

using Clock = std::chrono::high_resolution_clock;

/** Wall-clock time in µs for a lambda (used so CPU/driver time is measured). */
template<typename Func>
double benchmark_cpu_us(Func&& func, int warmup_iters, int timed_iters) {
    for (int i = 0; i < warmup_iters; i++) func();
    auto start = Clock::now();
    for (int i = 0; i < timed_iters; i++) func();
    auto end = Clock::now();
    return 1e-6 * std::chrono::duration<double, std::micro>(end - start).count() / timed_iters;
}

/** Run lambda once and return total wall-clock µs. */
template<typename Func>
double run_once_us(Func&& func) {
    auto start = Clock::now();
    func();
    auto end = Clock::now();
    return 1e-6 * std::chrono::duration<double, std::micro>(end - start).count();
}

/** Minimal kernel launch via libtorch (real cudaLaunchKernel under the hood). */
void launch_fill(int count) {
    auto t = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    for (int i = 0; i < count; i++) {
        t.fill_(0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

bool is_known_cuda_api(const std::string& api_name) {
    static const std::unordered_set<std::string> known = {
        "cudaEventQuery",
        "cudaEventRecord",
        "cudaStreamWaitEvent",
        "cudaStreamSynchronize",
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cudaPeekAtLastError",
        "cudaDeviceGetAttribute",
        "cudaStreamIsCapturing",
        "cudaStreamGetCaptureInfo_v2",
        "cudaDeviceSynchronize",
        // These need a kernel pointer; we fall back to benchmark_fill
        "cudaFuncSetAttribute",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    };
    return known.count(api_name) != 0;
}

double run_cuda_api_benchmark(const std::string& api_name, int num_runs) {
    const int warmup = 10;
    int actual = num_runs;

    try {
        if (api_name == "cudaLaunchKernel" || api_name == "cudaLaunchKernelExC") {
            launch_fill(actual);
            double total_us = run_once_us([&]() { launch_fill(actual); });
            return total_us / actual;
        }
        if (api_name == "cudaPeekAtLastError") {
            return benchmark_cpu_us([&]() { (void)cudaPeekAtLastError(); }, warmup, actual) / actual;
        }
        if (api_name == "cudaDeviceGetAttribute") {
            int val = 0;
            return benchmark_cpu_us([&]() {
                CUDA_CHECK(cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount, 0));
            }, warmup, actual) / actual;
        }
        if (api_name == "cudaFuncSetAttribute" ||
            api_name == "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
            api_name == "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") {
            return -1.0;
        }
        if (api_name == "cudaDeviceSynchronize") {
            auto t = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            double total_us = run_once_us([&]() {
                for (int i = 0; i < actual; i++) {
                    t.fill_(0);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            });
            return total_us / actual;
        }
    } catch (const std::exception& e) {
        std::cerr << "cuda_api_replay " << api_name << ": " << e.what() << std::endl;
        return -1.0;
    }

    cudaStream_t stream = nullptr;
    cudaStream_t stream2 = nullptr;
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (api_name == "cudaStreamWaitEvent") {
        CUDA_CHECK(cudaStreamCreate(&stream2));
        CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(event, stream2));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
    } else if (api_name == "cudaEventRecord" || api_name == "cudaEventQuery") {
        CUDA_CHECK(cudaEventCreate(&event));
    }

    try {
        double avg_us = -1.0;
        if (api_name == "cudaEventQuery") {
            CUDA_CHECK(cudaEventRecord(event, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            avg_us = benchmark_cpu_us([&]() {
                for (int i = 0; i < actual; i++) (void)cudaEventQuery(event);
            }, warmup, 1) / actual;
        } else if (api_name == "cudaEventRecord") {
            auto t = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            c10::cuda::CUDAGuard guard(0);
            c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromPool(false);
            c10::cuda::setCurrentCUDAStream(torch_stream);
            cudaStream_t raw_stream = torch_stream.stream();
            double total_us = run_once_us([&]() {
                for (int i = 0; i < actual; i++) {
                    t.fill_(0);
                    CUDA_CHECK(cudaEventRecord(event, raw_stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(raw_stream));
            });
            avg_us = total_us / actual;
        } else if (api_name == "cudaStreamWaitEvent") {
            double total_us = run_once_us([&]() {
                for (int i = 0; i < actual; i++) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            avg_us = total_us / actual;
        } else if (api_name == "cudaStreamSynchronize") {
            auto t = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            c10::cuda::CUDAGuard guard(0);
            c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromPool(false);
            c10::cuda::setCurrentCUDAStream(torch_stream);
            cudaStream_t raw_stream = torch_stream.stream();
            double total_us = run_once_us([&]() {
                for (int i = 0; i < actual; i++) {
                    t.fill_(0);
                    CUDA_CHECK(cudaStreamSynchronize(raw_stream));
                }
            });
            avg_us = total_us / actual;
        } else if (api_name == "cudaStreamIsCapturing") {
            cudaStreamCaptureStatus status;
            avg_us = benchmark_cpu_us([&]() {
                for (int i = 0; i < actual; i++) {
                    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
                }
            }, warmup, 1) / actual;
        } else if (api_name == "cudaStreamGetCaptureInfo_v2") {
            cudaStreamCaptureStatus status;
            unsigned long long id = 0;
            avg_us = benchmark_cpu_us([&]() {
                for (int i = 0; i < actual; i++) {
                    (void)cudaStreamGetCaptureInfo_v2(stream, &status, &id, nullptr, nullptr, nullptr);
                }
            }, warmup, 1) / actual;
        } else {
            if (stream) cudaStreamDestroy(stream);
            if (stream2) cudaStreamDestroy(stream2);
            if (event) cudaEventDestroy(event);
            return -1.0;
        }
        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
        if (stream2) CUDA_CHECK(cudaStreamDestroy(stream2));
        if (event) CUDA_CHECK(cudaEventDestroy(event));
        return avg_us;
    } catch (...) {
        if (stream) cudaStreamDestroy(stream);
        if (stream2) cudaStreamDestroy(stream2);
        if (event) cudaEventDestroy(event);
        throw;
    }
}
