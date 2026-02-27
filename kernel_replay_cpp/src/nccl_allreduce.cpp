#include "nccl_allreduce.hpp"
#include <iostream>
#include <cstring>
#include <chrono>

#ifdef HAVE_NCCL
#include "nccl.h"
#include <cuda_runtime.h>
#include <vector>

#define NCCLCHECK(c) do { \
    ncclResult_t r = (c); \
    if (r != ncclSuccess) { \
        std::cerr << "NCCL error " << (int)r << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return { false, 0.0, 0.0, 0, 0, 0, 0, "NCCL error " + std::to_string((int)r) }; \
    } \
} while(0)

#define CUDACHECK(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return { false, 0.0, 0.0, 0, 0, 0, 0, std::string("CUDA: ") + cudaGetErrorString(e) }; \
    } \
} while(0)

namespace nccl_replay {

namespace {
    constexpr size_t kDefaultNelems = 94208;
    constexpr int kDefaultNumGpus = 2;
}

bool has_nccl_support() { return true; }

AllReduceResult run_allreduce_replay(int total_calls, size_t nelems, int group_size, int warmup) {
    AllReduceResult result;
    result.nelems = nelems;
    result.nbytes = nelems * sizeof(__half);
    result.num_gpus = group_size;
    result.num_calls = total_calls;

    int device_count = 0;
    CUDACHECK(cudaGetDeviceCount(&device_count));
    if (device_count < group_size) {
        result.error_message = "Need at least " + std::to_string(group_size) + " GPUs, found " + std::to_string(device_count);
        return result;
    }

    std::vector<int> devs(group_size);
    for (int i = 0; i < group_size; i++) devs[i] = i;

    std::vector<ncclComm_t> comms(group_size);
    std::vector<void*> sendbuf(group_size);
    std::vector<void*> recvbuf(group_size);
    std::vector<cudaStream_t> streams(group_size);

    NCCLCHECK(ncclCommInitAll(comms.data(), group_size, devs.data()));

    for (int i = 0; i < group_size; i++) {
        CUDACHECK(cudaSetDevice(devs[i]));
        CUDACHECK(cudaMalloc(&sendbuf[i], result.nbytes));
        CUDACHECK(cudaMalloc(&recvbuf[i], result.nbytes));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        CUDACHECK(cudaMemset(sendbuf[i], 0, result.nbytes));
    }

    // Warmup
    for (int r = 0; r < warmup; r++) {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < group_size; i++)
            NCCLCHECK(ncclAllReduce(sendbuf[i], recvbuf[i], nelems, ncclHalf, ncclSum, comms[i], streams[i]));
        NCCLCHECK(ncclGroupEnd());
        for (int i = 0; i < group_size; i++) {
            CUDACHECK(cudaSetDevice(devs[i]));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    // Timed run
    cudaEvent_t start_ev, stop_ev;
    CUDACHECK(cudaSetDevice(devs[0]));
    CUDACHECK(cudaEventCreate(&start_ev));
    CUDACHECK(cudaEventCreate(&stop_ev));

    CUDACHECK(cudaEventRecord(start_ev, streams[0]));
    for (int r = 0; r < total_calls; r++) {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < group_size; i++)
            NCCLCHECK(ncclAllReduce(sendbuf[i], recvbuf[i], nelems, ncclHalf, ncclSum, comms[i], streams[i]));
        NCCLCHECK(ncclGroupEnd());
    }
    CUDACHECK(cudaEventRecord(stop_ev, streams[0]));
    CUDACHECK(cudaStreamSynchronize(streams[0]));
    CUDACHECK(cudaEventSynchronize(stop_ev));

    float ms = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
    result.total_time_us = ms * 1000.0;
    result.avg_time_us = (total_calls > 0) ? (result.total_time_us / total_calls) : 0.0;
    result.success = true;

    // Cleanup
    CUDACHECK(cudaEventDestroy(start_ev));
    CUDACHECK(cudaEventDestroy(stop_ev));
    for (int i = 0; i < group_size; i++) {
        CUDACHECK(cudaSetDevice(devs[i]));
        CUDACHECK(cudaFree(sendbuf[i]));
        CUDACHECK(cudaFree(recvbuf[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }

    return result;
}

}  // namespace nccl_replay

#else  // !HAVE_NCCL

namespace nccl_replay {

bool has_nccl_support() { return false; }

AllReduceResult run_allreduce_replay(int /* total_calls */, size_t /* nelems */, int /* group_size */, int /* warmup */) {
    AllReduceResult result;
    result.success = false;
    result.error_message = "Binary built without NCCL. Set NCCL_ROOT and rebuild with NCCL found.";
    return result;
}

}  // namespace nccl_replay

#endif  // HAVE_NCCL
