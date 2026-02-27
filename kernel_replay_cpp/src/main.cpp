#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <unistd.h>  // For _exit()
#include "kernel_registry.hpp"
#include "cuda_kernels.hpp"
#include "cublas_kernels.hpp"
#include "libtorch_kernels.hpp"
#include "benchmark_utils.hpp"
#include "nccl_allreduce.hpp"

// Forward declarations from other source files
namespace full_model {
    struct BenchmarkResult {
        double avg_time_us;
        double min_time_us;
        double max_time_us;
        double energy_joules;
        int num_runs;
        std::string start_timestamp;
        std::string end_timestamp;
    };
    BenchmarkResult benchmark_model(const std::string& model_path, 
                                   int seq_len, int warmup_runs, int timed_runs);
    bool save_results(const BenchmarkResult& result, const std::string& output_path);
}

namespace aggregation {
    struct KernelTiming {
        std::string name;
        kernel::Tier tier;
        int count;
        double single_time_us;
        double total_time_us;
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
    
    AggregatedResults run_isolated_kernels(const kernel::KernelRegistry& registry, int num_runs);
    bool save_isolated_results(const AggregatedResults& results, const std::string& output_path);
    void generate_comparison_report(const AggregatedResults& isolated_results,
                                   double actual_total_us,
                                   const std::string& output_path);
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <mode> [options]" << std::endl;
    std::cout << "\nModes:" << std::endl;
    std::cout << "  full      - Run full model benchmark only" << std::endl;
    std::cout << "  isolated  - Run isolated kernel benchmarks only" << std::endl;
    std::cout << "  compare   - Run both and compare (default)" << std::endl;
    std::cout << "  allreduce - Run NCCL AllReduce replay (2 GPUs, 94208 elem x 64; requires NCCL)" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --model <path>        Path to TorchScript model (required for full/compare)" << std::endl;
    std::cout << "  --kernels <path>      Path to kernel_signatures.json (default: data/kernel_signatures.json)" << std::endl;
    std::cout << "  --seq-len <n>         Sequence length for model input (default: 5)" << std::endl;
    std::cout << "  --warmup <n>          Number of warmup runs (default: 10)" << std::endl;
    std::cout << "  --runs <n>            Number of timed runs (default: 100)" << std::endl;
    std::cout << "  --output-dir <path>   Output directory for results (default: results/)" << std::endl;
    std::cout << "  --allreduce-repeats <n>  AllReduce replay: number of AllReduce steps (default: 64)" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << prog_name << " compare --model bloom_560m_traced.pt --kernels data/kernel_signatures.json" << std::endl;
    std::cout << "  " << prog_name << " compare --model bloom_560m_traced.pt --runs 1000" << std::endl;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string mode = "compare";
    std::string model_path = "";
    std::string kernels_path = "data/kernel_signatures.json";
    std::string output_dir = "results/";
    int seq_len = 5;
    int warmup_runs = 10;
    int timed_runs = 100;
    int allreduce_repeats = 64;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--kernels" && i + 1 < argc) {
            kernels_path = argv[++i];
        }
        else if (arg == "--seq-len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        }
        else if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::atoi(argv[++i]);
        }
        else if (arg == "--runs" && i + 1 < argc) {
            timed_runs = std::atoi(argv[++i]);
        }
        else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if (arg == "--allreduce-repeats" && i + 1 < argc) {
            allreduce_repeats = std::atoi(argv[++i]);
        }
        else if (arg == "full" || arg == "isolated" || arg == "compare" || arg == "allreduce") {
            mode = arg;
        }
    }
    
    std::cout << "=== BLOOM Kernel Replay Benchmark ===" << std::endl;
    std::cout << "Mode: " << mode << std::endl;
    
    // Check CUDA device
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    // AllReduce mode: run NCCL AllReduce replay and exit
    if (mode == "allreduce") {
        if (!nccl_replay::has_nccl_support()) {
            std::cerr << "AllReduce mode requires NCCL. Set NCCL_ROOT (e.g. to python nvidia.nccl path) and rebuild." << std::endl;
            return 1;
        }
        if (device_count < 2) {
            std::cerr << "AllReduce replay requires at least 2 GPUs (found " << device_count << ")." << std::endl;
            return 1;
        }
        std::cout << "Running NCCL AllReduce replay: " << allreduce_repeats << " steps, 94208 elem (fp16), GPUs 0,1" << std::endl;
        auto res = nccl_replay::run_allreduce_replay(allreduce_repeats, 94208, 2, 2);
        if (!res.success) {
            std::cerr << "AllReduce replay failed: " << res.error_message << std::endl;
            return 1;
        }
        std::cout << "AllReduce replay OK: total " << (res.total_time_us / 1000.0) << " ms, "
                  << res.avg_time_us << " us/call, " << res.nelems << " elem, " << res.nbytes << " B" << std::endl;
        _exit(0);
    }
    
    // Initialize cuBLAS (needed for Tier 2)
    if (!tier2::init_cublas()) {
        std::cerr << "Error: Failed to initialize cuBLAS" << std::endl;
        return 1;
    }
    
    bool success = true;
    full_model::BenchmarkResult full_result{0, 0, 0, 0, 0};
    aggregation::AggregatedResults isolated_results;
    
    // Run full model benchmark
    if (mode == "full" || mode == "compare") {
        if (model_path.empty()) {
            std::cerr << "Error: --model is required for full/compare mode" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
        full_result = full_model::benchmark_model(model_path, seq_len, warmup_runs, timed_runs);
        if (full_result.avg_time_us < 0) {
            std::cerr << "Error: Full model benchmark failed" << std::endl;
            success = false;
        } else {
            full_model::save_results(full_result, output_dir + "full_model_timing.json");
        }
    }
    
    // Run isolated kernel benchmarks
    if (mode == "isolated" || mode == "compare") {
        kernel::KernelRegistry registry;
        if (!registry.load_from_file(kernels_path)) {
            std::cerr << "Error: Failed to load kernel registry from " << kernels_path << std::endl;
            return 1;
        }
        
        registry.print_summary();
        
        isolated_results = aggregation::run_isolated_kernels(registry, timed_runs);
        aggregation::save_isolated_results(isolated_results, output_dir + "isolated_kernels_timing.json");
    }
    
    // Generate comparison report
    if (mode == "compare" && success) {
        try {
            aggregation::generate_comparison_report(
                isolated_results,
                full_result.avg_time_us,
                output_dir + "comparison_report.json"
            );
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to generate comparison report: " << e.what() << std::endl;
        }
    }
    
    std::cout.flush();
    std::cerr.flush();
    
    _exit(success ? 0 : 1);
}
