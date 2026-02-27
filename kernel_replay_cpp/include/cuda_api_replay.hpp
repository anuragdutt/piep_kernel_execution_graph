#pragma once

#include <string>

/**
 * Replay real CUDA Runtime API calls (cudaEventRecord, cudaEventQuery,
 * cudaStreamWaitEvent, cudaLaunchKernel, etc.) so that system power
 * reflects actual CPU/driver overhead instead of a proxy kernel.
 *
 * Each API is run in a loop; wall-clock time is used so the power
 * meter sees the real duration. Returns average time per call (µs).
 *
 * @param api_name   Exact name from trace (e.g. "cudaEventQuery", "cudaLaunchKernel")
 * @param num_runs   Number of API calls to execute in the timed loop
 * @return Average time per call in microseconds, or -1 if unknown API
 */
double run_cuda_api_benchmark(const std::string& api_name, int num_runs);

/**
 * Returns true if api_name is a known cuda_api that we replay with real API calls.
 */
bool is_known_cuda_api(const std::string& api_name);
