#include "kernel_registry.hpp"
#include "cuda_kernels.hpp"
#include "cublas_kernels.hpp"
#include "libtorch_kernels.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>

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

struct KernelTiming {
    std::string name;
    kernel::Tier tier;
    int count;
    double single_time_us;
    double total_time_us;
    std::string start_timestamp;  // For per-kernel energy calculation
    std::string end_timestamp;
    int benchmark_runs;  // Number of runs used for this kernel
};

struct AggregatedResults {
    std::vector<KernelTiming> kernel_timings;
    double predicted_total_us;
    double tier1_total_us;
    double tier2_total_us;
    double tier3_total_us;
    int tier1_count;
    int tier2_count;
    int tier3_count;
    int num_runs;
    std::string start_timestamp;
    std::string end_timestamp;
};

AggregatedResults run_isolated_kernels(const kernel::KernelRegistry& registry, int num_runs) {
    std::cout << "\n=== Running Isolated Kernel Benchmarks ===" << std::endl;
    std::cout << "Running each kernel " << num_runs << " times for accurate energy measurement" << std::endl;
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
    results.tier1_count = 0;
    results.tier2_count = 0;
    results.tier3_count = 0;
    results.num_runs = num_runs;  // Record user's requested runs (for reference only)
    
    const auto& kernels = registry.get_all_kernels();
    int total = kernels.size();
    int completed = 0;
    
    // Record start timestamp for energy correlation
    results.start_timestamp = get_iso_timestamp();
    std::cout << "Start timestamp: " << results.start_timestamp << std::endl;
    std::cout << "\nNote: Run counts chosen so each kernel runs 100ms+ for 2–3+ nvidia-smi samples (~25 Hz)." << std::endl;
    
    for (const auto& sig : kernels) {
        double avg_single_time_us = -1.0;
        std::string kernel_start_ts, kernel_end_ts;
        int actual_runs = 0;
        
        try {
            // Target: each kernel runs long enough for several nvidia-smi samples (~25 Hz => 1 sample/40ms).
            // Aim for ~100ms+ per kernel so we get 2–3+ power samples and energy is measured not estimated.
            int adaptive_runs;
            if (sig.tier == kernel::Tier::CUBLAS) {
                adaptive_runs = 50000;   // ~1–2.5s for typical GEMM/GEMV => 25–60+ samples at 25 Hz
            } else if (sig.tier == kernel::Tier::CUDA_RUNTIME) {
                adaptive_runs = 600000;  // ~2–3s for memcpy => 50+ samples at 25 Hz
            } else {
                adaptive_runs = 5000000; // 5M: even 0.02us kernels => 100ms window => 2–3 samples at 25 Hz
            }
            
            // Record start timestamp for THIS kernel
            kernel_start_ts = get_iso_timestamp();
            
            // Each tier benchmark function does warmup once, then times adaptive_runs iterations
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
            }
            
            actual_runs = adaptive_runs;
            
            // Record end timestamp for THIS kernel
            kernel_end_ts = get_iso_timestamp();
            
        } catch (const std::exception& e) {
            std::cerr << "Error benchmarking kernel '" << sig.name << "': " 
                      << e.what() << std::endl;
            avg_single_time_us = 0.0;  // Skip this kernel
            kernel_start_ts = "";
            kernel_end_ts = "";
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
            timing.benchmark_runs = actual_runs;  // Save actual adaptive run count
            
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
    j["tier1_count"] = results.tier1_count;
    j["tier2_count"] = results.tier2_count;
    j["tier3_count"] = results.tier3_count;
    
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
        {"percentage_of_predicted", (isolated_results.tier3_total_us / isolated_results.predicted_total_us) * 100.0}
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
