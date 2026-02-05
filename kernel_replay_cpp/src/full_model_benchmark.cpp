#include <torch/torch.h>
#include <torch/script.h>
#include "benchmark_utils.hpp"
#include "energy_hooks.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace full_model {

struct BenchmarkResult {
    double avg_time_us;
    double min_time_us;
    double max_time_us;
    double energy_joules;  // Placeholder for energy measurement
    int num_runs;
    std::string start_timestamp;
    std::string end_timestamp;
};

// Helper function to get ISO timestamp
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

BenchmarkResult benchmark_model(const std::string& model_path, 
                                int seq_len = 5,
                                int warmup_runs = 10,
                                int timed_runs = 100,
                                bool disable_fusion = false) {
    std::cout << "\n=== Full Model Benchmark ===" << std::endl;
    std::cout << "Loading model from: " << model_path << std::endl;
    
    // Disable JIT fusion if requested (for fair comparison with isolated kernels)
    if (disable_fusion) {
        std::cout << "Disabling JIT kernel fusion for fair comparison..." << std::endl;
        
        // Method 1: Disable graph executor optimizations
        torch::jit::setGraphExecutorOptimize(false);
        
        // Method 2: Disable fusion group inlining
        torch::jit::debugSetFusionGroupInlining(false);
        
        // Method 3: Set fusion strategy to disable both static and dynamic fusion
        torch::jit::FusionStrategy strategy = {{torch::jit::FusionBehavior::STATIC, 0}, 
                                                {torch::jit::FusionBehavior::DYNAMIC, 0}};
        torch::jit::setFusionStrategy(strategy);
        
        std::cout << "  Graph executor optimization: OFF" << std::endl;
        std::cout << "  Fusion group inlining: OFF" << std::endl;
        std::cout << "  Fusion strategy: STATIC=0, DYNAMIC=0" << std::endl;
    } else {
        std::cout << "JIT kernel fusion: ENABLED (default)" << std::endl;
    }
    
    // Load TorchScript model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.eval();
        model.to(torch::kCUDA);
        std::cout << "Model loaded and moved to CUDA" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return {-1.0, -1.0, -1.0, 0.0, 0};
    }
    
    // Create input tensor (example tokens)
    auto input_ids = torch::randint(0, 50000, {1, seq_len}, 
                                   torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    
    std::cout << "Input shape: [" << input_ids.size(0) << ", " << input_ids.size(1) << "]" << std::endl;
    
    // Warmup
    std::cout << "Warming up (" << warmup_runs << " runs)..." << std::endl;
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < warmup_runs; i++) {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_ids);
            auto output = model.forward(inputs);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Warmup complete" << std::endl;
    
    // Timed runs
    std::cout << "Running benchmark (" << timed_runs << " runs)..." << std::endl;
    std::vector<double> times_us;
    
    // Energy probe (placeholder)
    energy::EnergyProbe* probe = energy::create_probe();
    probe->start_measurement();
    
    // Record start timestamp (for power meter correlation)
    std::string start_timestamp = get_iso_timestamp();
    std::cout << "Start timestamp: " << start_timestamp << std::endl;
    
    {
        torch::NoGradGuard no_grad;
        
        for (int i = 0; i < timed_runs; i++) {
            benchmark::CudaTimer timer;
            
            timer.start();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_ids);
            auto output = model.forward(inputs);
            timer.stop();
            
            times_us.push_back(timer.elapsed_us());
            
            if ((i + 1) % 10 == 0) {
                std::cout << "  Progress: " << (i + 1) << "/" << timed_runs << std::endl;
            }
        }
    }
    
    // Record end timestamp
    std::string end_timestamp = get_iso_timestamp();
    std::cout << "End timestamp:   " << end_timestamp << std::endl;
    
    double energy_j = probe->read_energy_joules();
    delete probe;
    
    // Calculate statistics
    double sum = 0.0;
    double min_time = times_us[0];
    double max_time = times_us[0];
    
    for (double t : times_us) {
        sum += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    
    double avg_time = sum / times_us.size();
    
    BenchmarkResult result;
    result.avg_time_us = avg_time;
    result.min_time_us = min_time;
    result.max_time_us = max_time;
    result.energy_joules = energy_j;
    result.num_runs = timed_runs;
    result.start_timestamp = start_timestamp;
    result.end_timestamp = end_timestamp;
    
    std::cout << "\n=== Full Model Results ===" << std::endl;
    std::cout << "Average time: " << avg_time << " us (" << (avg_time / 1000.0) << " ms)" << std::endl;
    std::cout << "Min time:     " << min_time << " us" << std::endl;
    std::cout << "Max time:     " << max_time << " us" << std::endl;
    std::cout << "Num runs:     " << timed_runs << std::endl;
    std::cout << "Energy:       " << energy_j << " J (placeholder)" << std::endl;
    std::cout << "Start time:   " << start_timestamp << std::endl;
    std::cout << "End time:     " << end_timestamp << std::endl;
    
    return result;
}

bool save_results(const BenchmarkResult& result, const std::string& output_path) {
    json j;
    j["full_model_inference_us"] = result.avg_time_us;
    j["full_model_inference_ms"] = result.avg_time_us / 1000.0;
    j["min_time_us"] = result.min_time_us;
    j["max_time_us"] = result.max_time_us;
    j["energy_joules"] = result.energy_joules;
    j["num_runs"] = result.num_runs;
    j["start_timestamp"] = result.start_timestamp;
    j["end_timestamp"] = result.end_timestamp;
    
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << output_path << " for writing" << std::endl;
        return false;
    }
    
    file << j.dump(2) << std::endl;
    std::cout << "Results saved to: " << output_path << std::endl;
    return true;
}

} // namespace full_model
