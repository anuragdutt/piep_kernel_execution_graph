#ifndef NCCL_ALLREDUCE_HPP
#define NCCL_ALLREDUCE_HPP

#include <string>

namespace nccl_replay {

/// Result of AllReduce replay (timing and success).
struct AllReduceResult {
    bool success = false;
    double total_time_us = 0.0;   // total time for all AllReduce calls
    double avg_time_us = 0.0;     // per-call average
    int num_calls = 0;
    size_t nelems = 0;
    size_t nbytes = 0;
    int num_gpus = 0;
    std::string error_message;
};

/**
 * Run AllReduce replay with the given message size and group size (TP width).
 * Requires at least group_size GPUs. Uses NCCL AllReduce (Sum, fp16).
 *
 * Runs total_calls AllReduce operations in the timed section (like other tiers:
 * run many times, then caller divides energy by total_calls and multiplies by
 * invocation_count for per-inference energy).
 *
 * @param total_calls  Number of AllReduce operations to run in timed section (e.g. 30000)
 * @param nelems       Message size in elements (fp16); must match trace (e.g. 94208 or 4096)
 * @param group_size   Number of GPUs (2 for TP=2)
 * @param warmup       Warmup rounds before timed run (default 2)
 * @return             Timing and success; num_calls = total_calls.
 */
AllReduceResult run_allreduce_replay(int total_calls, size_t nelems, int group_size = 2, int warmup = 2);

/// Returns true if this binary was built with NCCL support.
bool has_nccl_support();

}  // namespace nccl_replay

#endif  // NCCL_ALLREDUCE_HPP
