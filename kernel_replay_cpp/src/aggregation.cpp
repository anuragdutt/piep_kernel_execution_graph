#include "kernel_registry.hpp"
#include "cuda_kernels.hpp"
#include "cublas_kernels.hpp"
#include "libtorch_kernels.hpp"
#include "nccl_allreduce.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

using json = nlohmann::json;

namespace aggregation {

// Helper to get ISO timestamp with milliseconds
std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Helper to start metrics collection subprocess
// Returns PID of the metrics collector process, or -1 on failure
pid_t start_metrics_collector(const std::string& output_file, double sample_interval = 0.1) {
    pid_t pid = fork();
    
    if (pid < 0) {
        std::cerr << "Failed to fork metrics collector process" << std::endl;
        return -1;
    }
    
    if (pid == 0) {
        // Child process: exec the metrics collector
        std::string interval_str = std::to_string(sample_interval);
        
        // Use absolute path to script
        const char* python_path = "/home/adutt/wasiq/piep_kernel_execution_graph/kernel_replay_cpp/scripts/collect_system_metrics.py";
        
        // Keep stderr for debugging, only redirect stdout
        freopen("/dev/null", "w", stdout);
        // DO NOT redirect stderr - we want to see errors!
        
        execl("/usr/bin/python3", "python3", python_path, 
              "--output", output_file.c_str(),
              "--interval", interval_str.c_str(),
              nullptr);
        
        // If execl returns, there was an error
        std::cerr << "Failed to exec metrics collector: " << python_path << std::endl;
        exit(1);
    }
    
    // Parent process: return child PID
    // Give the metrics collector a moment to start up
    usleep(200000);  // 200ms
    return pid;
}

// Helper to stop metrics collection subprocess
bool stop_metrics_collector(pid_t pid, int timeout_ms = 5000) {
    if (pid <= 0) {
        return false;
    }
    
    // Send SIGINT to trigger graceful shutdown
    if (kill(pid, SIGINT) != 0) {
        std::cerr << "Failed to send SIGINT to metrics collector (PID " << pid << ")" << std::endl;
        return false;
    }
    
    // Wait for process to exit
    int status;
    int wait_iterations = timeout_ms / 100;
    for (int i = 0; i < wait_iterations; i++) {
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            // Process exited
            return WIFEXITED(status) && WEXITSTATUS(status) == 0;
        } else if (result < 0) {
            std::cerr << "waitpid failed for metrics collector" << std::endl;
            return false;
        }
        usleep(100000);  // 100ms
    }
    
    // Timeout: forcefully kill
    std::cerr << "Metrics collector didn't exit gracefully, sending SIGKILL" << std::endl;
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return false;
}

// Helper to load metrics from JSON file
json load_metrics_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open metrics file: " << filepath << std::endl;
        return json::object();
    }
    
    try {
        json metrics;
        file >> metrics;
        return metrics;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to parse metrics file " << filepath << ": " << e.what() << std::endl;
        return json::object();
    }
}

struct KernelTiming {
    std::string name;
    kernel::Tier tier;
    int count;
    double single_time_us;
    double total_time_us;
    std::string start_timestamp;  // For per-kernel energy calculation
    std::string end_timestamp;
    int benchmark_runs;  // Number of runs used for this kernel
    
    // System metrics collected during benchmarking
    std::string metrics_file;  // Path to metrics JSON file
    bool has_metrics;
    json system_metrics;  // Loaded metrics data
};

struct AggregatedResults {
    std::vector<KernelTiming> kernel_timings;
    double predicted_total_us;
    double tier1_total_us;
    double tier2_total_us;
    double tier3_total_us;
    double tier4_total_us;
    int tier1_count;
    int tier2_count;
    int tier3_count;
    int tier4_count;
    int num_runs;
    std::string start_timestamp;
    std::string end_timestamp;
};

AggregatedResults run_isolated_kernels(const kernel::KernelRegistry& registry, int num_runs) {
    std::cout << "\n=== Running Isolated Kernel Benchmarks ===" << std::endl;
    std::cout << "Note: Each kernel runs long enough to get good power samples (~10s minimum)" << std::endl;
    
    // For energy measurements, we need LONG runs per kernel (at least 10 seconds)
    // to get enough power samples (meter samples at ~1 Hz)
    // Adaptive runs based on kernel type:
    // - Small/fast kernels (memcpy, elementwise): 1M runs → ~10s
    // - Medium kernels (GEMV, small ops): 100K runs → ~10s
    // - Large kernels (GEMM): 10K runs → ~10s+
    
    AggregatedResults results;
    results.predicted_total_us = 0.0;
    results.tier1_total_us = 0.0;
    results.tier2_total_us = 0.0;
    results.tier3_total_us = 0.0;
    results.tier4_total_us = 0.0;
    results.tier1_count = 0;
    results.tier2_count = 0;
    results.tier3_count = 0;
    results.tier4_count = 0;
    results.num_runs = num_runs;  // Record user's requested runs (for reference only)
    
    const auto& kernels = registry.get_all_kernels();
    int total = kernels.size();
    int completed = 0;

    // One run per unique signature; aggregate as (single_time × invocation_count) per kernel.
    // No cache/reuse: each signature is run once; predicted total = sum over signatures.
    // Tier 4 (AllReduce) is no different: each Tier 4 entry has its own nelems/group_size/count.
    
    // Record start timestamp for energy correlation
    results.start_timestamp = get_iso_timestamp();
    std::cout << "Start timestamp: " << results.start_timestamp << std::endl;
    std::cout << "\nNote: Run counts chosen so each kernel runs 2–3+ s for system power meter (~1 Hz)." << std::endl;

    for (const auto& sig : kernels) {
        double avg_single_time_us = -1.0;
        std::string kernel_start_ts, kernel_end_ts;
        std::string metrics_file;
        pid_t metrics_pid = -1;
        int benchmark_runs = 0;

        try {
            // Target: each kernel runs long enough for system power meter (WattsUp at ~1 Hz).
            int adaptive_runs;
            if (sig.tier == kernel::Tier::CUBLAS) {
                adaptive_runs = 1000000;   // GEMM/GEMV: 100K was ~0.2–0.6 s; 500K → ~1–3 s
            } else if (sig.tier == kernel::Tier::CUDA_RUNTIME) {
                adaptive_runs = 2000000;  // Memcpy/memset: 1M was ~1.4 s for one; 2M → ~2–3 s 
            } else {
                adaptive_runs = 5000000;  // Tier 3 libtorch (all ops, including cuda_api)
            }
            
            // Create metrics file path for this kernel (absolute path)
            std::stringstream metrics_path;
            metrics_path << "/home/adutt/wasiq/piep_kernel_execution_graph/kernel_replay_cpp/results/metrics/kernel_" << completed << ".json";
            metrics_file = metrics_path.str();
            
            // Start metrics collector subprocess
            metrics_pid = start_metrics_collector(metrics_file, 0.1);
            if (metrics_pid < 0) {
                std::cerr << "Warning: Failed to start metrics collector for kernel " << sig.name << std::endl;
            }
            
            // Record start timestamp for THIS kernel
            kernel_start_ts = get_iso_timestamp();
            
            benchmark_runs = adaptive_runs;
            switch (sig.tier) {
                case kernel::Tier::CUDA_RUNTIME:
                    avg_single_time_us = tier1::run_tier1_kernel(sig, adaptive_runs);
                    results.tier1_count++;
                    break;
                case kernel::Tier::CUBLAS:
                    avg_single_time_us = tier2::run_tier2_kernel(sig, adaptive_runs);
                    results.tier2_count++;
                    break;
                case kernel::Tier::LIBTORCH:
                    avg_single_time_us = tier3::run_tier3_kernel(sig, adaptive_runs);
                    results.tier3_count++;
                    break;
                case kernel::Tier::COMMUNICATION:
                    if (nccl_replay::has_nccl_support()) {
                        size_t nelems = sig.get_allreduce_nelems();
                        int group_size = sig.get_allreduce_group_size();
                        auto ar_res = nccl_replay::run_allreduce_replay(adaptive_runs, nelems, group_size, 2);
                        if (ar_res.success && ar_res.num_calls > 0)
                            avg_single_time_us = ar_res.avg_time_us;
                        else
                            avg_single_time_us = -1.0;
                    } else {
                        avg_single_time_us = -1.0;  // Skip if no NCCL
                    }
                    results.tier4_count++;
                    break;
            }

            // Record end timestamp for THIS kernel
            kernel_end_ts = get_iso_timestamp();

            // Stop metrics collector
            if (metrics_pid > 0) {
                bool stopped = stop_metrics_collector(metrics_pid, 5000);
                if (!stopped) {
                    std::cerr << "Warning: Metrics collector may not have exited cleanly" << std::endl;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "Error benchmarking kernel '" << sig.name << "': " 
                      << e.what() << std::endl;
            avg_single_time_us = 0.0;  // Skip this kernel
            kernel_start_ts = "";
            kernel_end_ts = "";
            
            // Clean up metrics collector if it's still running
            if (metrics_pid > 0) {
                kill(metrics_pid, SIGKILL);
                waitpid(metrics_pid, nullptr, 0);
            }
        }

        if (avg_single_time_us >= 0) {
            double total_time_us = avg_single_time_us * sig.count;
            
            KernelTiming timing;
            timing.name = sig.name;
            timing.tier = sig.tier;
            timing.count = sig.count;
            timing.single_time_us = avg_single_time_us;
            timing.total_time_us = total_time_us;
            timing.start_timestamp = kernel_start_ts;
            timing.end_timestamp = kernel_end_ts;
            timing.benchmark_runs = benchmark_runs;
            
            // Load system metrics if available
            timing.metrics_file = metrics_file;
            timing.has_metrics = false;
            if (!metrics_file.empty()) {
                timing.system_metrics = load_metrics_file(metrics_file);
                timing.has_metrics = !timing.system_metrics.empty();
                
                // Delete temporary metrics file after loading
                if (timing.has_metrics) {
                    std::remove(metrics_file.c_str());
                }
            }
            
            results.kernel_timings.push_back(timing);
            results.predicted_total_us += total_time_us;
            
            switch (sig.tier) {
                case kernel::Tier::CUDA_RUNTIME:
                    results.tier1_total_us += total_time_us;
                    break;
                case kernel::Tier::CUBLAS:
                    results.tier2_total_us += total_time_us;
                    break;
                case kernel::Tier::LIBTORCH:
                    results.tier3_total_us += total_time_us;
                    break;
                case kernel::Tier::COMMUNICATION:
                    results.tier4_total_us += total_time_us;
                    break;
            }
        }
        
        completed++;
        if (completed % 10 == 0 || completed == total) {
            std::cout << "Progress: " << completed << "/" << total 
                      << " kernels benchmarked" << std::endl;
        }
    }
    
    // Record end timestamp for energy correlation
    results.end_timestamp = get_iso_timestamp();
    std::cout << "End timestamp:   " << results.end_timestamp << std::endl;
    
    std::cout << "\n=== Isolated Kernels Summary ===" << std::endl;
    std::cout << "Tier 1 (CUDA Runtime): " << results.tier1_count << " kernels, "
              << results.tier1_total_us << " us total" << std::endl;
    std::cout << "Tier 2 (cuBLAS):       " << results.tier2_count << " kernels, "
              << results.tier2_total_us << " us total" << std::endl;
    std::cout << "Tier 3 (libtorch):     " << results.tier3_count << " kernels, "
              << results.tier3_total_us << " us total" << std::endl;
    std::cout << "Tier 4 (Communication): " << results.tier4_count << " kernels, "
              << results.tier4_total_us << " us total" << std::endl;
    std::cout << "\nPredicted total:      " << results.predicted_total_us 
              << " us (" << (results.predicted_total_us / 1000.0) << " ms)" << std::endl;
    
    return results;
}

bool save_isolated_results(const AggregatedResults& results, const std::string& output_path) {
    json j;
    j["predicted_total_us"] = results.predicted_total_us;
    j["predicted_total_ms"] = results.predicted_total_us / 1000.0;
    j["num_runs"] = results.num_runs;
    j["start_timestamp"] = results.start_timestamp;
    j["end_timestamp"] = results.end_timestamp;
    j["tier1_total_us"] = results.tier1_total_us;
    j["tier2_total_us"] = results.tier2_total_us;
    j["tier3_total_us"] = results.tier3_total_us;
    j["tier4_total_us"] = results.tier4_total_us;
    j["tier1_count"] = results.tier1_count;
    j["tier2_count"] = results.tier2_count;
    j["tier3_count"] = results.tier3_count;
    j["tier4_count"] = results.tier4_count;
    
    // Save per-kernel timing and timestamp data for per-kernel energy calculation
    json kernels_array = json::array();
    for (const auto& kt : results.kernel_timings) {
        json kernel_obj;
        kernel_obj["name"] = kt.name;
        kernel_obj["tier"] = static_cast<int>(kt.tier);
        kernel_obj["invocation_count"] = kt.count;  // How many times this kernel is called per inference
        kernel_obj["single_time_us"] = kt.single_time_us;  // Average time for one execution
        kernel_obj["total_time_us"] = kt.total_time_us;  // single_time * invocation_count
        kernel_obj["start_timestamp"] = kt.start_timestamp;
        kernel_obj["end_timestamp"] = kt.end_timestamp;
        kernel_obj["benchmark_runs"] = kt.benchmark_runs;  // How many times we ran it for measurement
        
        // Add system metrics if available
        kernel_obj["has_metrics"] = kt.has_metrics;
        if (kt.has_metrics) {
            kernel_obj["system_metrics"] = kt.system_metrics;
        }
        
        kernels_array.push_back(kernel_obj);
    }
    j["kernels"] = kernels_array;
    
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << output_path << " for writing" << std::endl;
        return false;
    }
    
    file << j.dump(2) << std::endl;
    return true;
}

void generate_comparison_report(const AggregatedResults& isolated_results,
                               double actual_total_us,
                               const std::string& output_path) {
    json report;
    
    // Top-level comparison
    double error_us = std::abs(isolated_results.predicted_total_us - actual_total_us);
    double error_pct = (error_us / actual_total_us) * 100.0;
    
    report["full_model_inference_us"] = actual_total_us;
    report["full_model_inference_ms"] = actual_total_us / 1000.0;
    report["predicted_from_kernels_us"] = isolated_results.predicted_total_us;
    report["predicted_from_kernels_ms"] = isolated_results.predicted_total_us / 1000.0;
    report["error_us"] = error_us;
    report["error_percent"] = error_pct;
    
    // Tier breakdown
    report["tier_breakdown"]["tier1_cuda_runtime"] = {
        {"method", "cudaMemcpy/cudaMemset"},
        {"unique_kernels", isolated_results.tier1_count},
        {"total_us", isolated_results.tier1_total_us},
        {"percentage_of_predicted", (isolated_results.tier1_total_us / isolated_results.predicted_total_us) * 100.0}
    };
    
    report["tier_breakdown"]["tier2_cublas"] = {
        {"method", "cublasGemmEx"},
        {"unique_kernels", isolated_results.tier2_count},
        {"total_us", isolated_results.tier2_total_us},
        {"percentage_of_predicted", (isolated_results.tier2_total_us / isolated_results.predicted_total_us) * 100.0}
    };
    
    report["tier_breakdown"]["tier3_libtorch"] = {
        {"method", "torch::layer_norm/add/etc"},
        {"unique_kernels", isolated_results.tier3_count},
        {"total_us", isolated_results.tier3_total_us},
        {"percentage_of_predicted", isolated_results.predicted_total_us > 0 ? (isolated_results.tier3_total_us / isolated_results.predicted_total_us) * 100.0 : 0.0}
    };
    report["tier_breakdown"]["tier4_communication"] = {
        {"method", "NCCL AllReduce"},
        {"unique_kernels", isolated_results.tier4_count},
        {"total_us", isolated_results.tier4_total_us},
        {"percentage_of_predicted", isolated_results.predicted_total_us > 0 ? (isolated_results.tier4_total_us / isolated_results.predicted_total_us) * 100.0 : 0.0}
    };
    
    // Individual kernel details (top 20 by total time)
    auto sorted_timings = isolated_results.kernel_timings;
    std::sort(sorted_timings.begin(), sorted_timings.end(),
              [](const KernelTiming& a, const KernelTiming& b) {
                  return a.total_time_us > b.total_time_us;
              });
    
    json kernel_details = json::array();
    for (size_t i = 0; i < std::min(size_t(20), sorted_timings.size()); i++) {
        const auto& k = sorted_timings[i];
        kernel_details.push_back({
            {"name", k.name},
            {"tier", static_cast<int>(k.tier)},
            {"count", k.count},
            {"single_us", k.single_time_us},
            {"total_us", k.total_time_us},
            {"percentage_of_predicted", (k.total_time_us / isolated_results.predicted_total_us) * 100.0}
        });
    }
    report["top_kernels"] = kernel_details;
    
    // Write to file
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << output_path << " for writing" << std::endl;
        return;
    }
    
    file << report.dump(2) << std::endl;
    std::cout << "\nComparison report saved to: " << output_path << std::endl;
    
    // Print summary to console
    std::cout << "\n=== Comparison Summary ===" << std::endl;
    std::cout << "Actual (full model):      " << actual_total_us << " us ("
              << (actual_total_us / 1000.0) << " ms)" << std::endl;
    std::cout << "Predicted (sum kernels):  " << isolated_results.predicted_total_us << " us ("
              << (isolated_results.predicted_total_us / 1000.0) << " ms)" << std::endl;
    std::cout << "Error:                    " << error_us << " us (" 
              << error_pct << "%)" << std::endl;
}

} // namespace aggregation
