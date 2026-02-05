#include "kernel_registry.hpp"
#include <fstream>
#include <iostream>

namespace kernel {

bool KernelRegistry::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << path << std::endl;
        return false;
    }
    
    json data;
    file >> data;
    
    // Clear existing data
    kernels_.clear();
    tier_indices_.clear();
    
    // Parse kernels array
    if (!data.contains("kernels")) {
        std::cerr << "Error: JSON file missing 'kernels' array" << std::endl;
        return false;
    }
    
    const auto& kernels_json = data["kernels"];
    for (size_t i = 0; i < kernels_json.size(); i++) {
        const auto& k = kernels_json[i];
        
        KernelSignature sig;
        sig.name = k["name"];
        sig.tier = static_cast<Tier>(k["tier"].get<int>());
        sig.count = k["count"];
        sig.params = k["params"];
        
        // Extract signature info
        if (k.contains("signature")) {
            const auto& signature = k["signature"];
            if (signature.contains("grid")) {
                sig.grid = signature["grid"].get<std::vector<int>>();
            }
            if (signature.contains("block")) {
                sig.block = signature["block"].get<std::vector<int>>();
            }
            if (signature.contains("shared memory")) {
                sig.shared_memory = signature["shared memory"];
            }
        }
        
        kernels_.push_back(sig);
        tier_indices_[sig.tier].push_back(i);
    }
    
    std::cout << "Loaded " << kernels_.size() << " kernels from " << path << std::endl;
    return true;
}

std::vector<KernelSignature> KernelRegistry::get_tier_kernels(Tier tier) const {
    std::vector<KernelSignature> result;
    
    auto it = tier_indices_.find(tier);
    if (it != tier_indices_.end()) {
        for (size_t idx : it->second) {
            result.push_back(kernels_[idx]);
        }
    }
    
    return result;
}

KernelRegistry::Summary KernelRegistry::get_summary() const {
    Summary summary;
    summary.total_kernels = kernels_.size();
    
    for (const auto& kernel : kernels_) {
        switch (kernel.tier) {
            case Tier::CUDA_RUNTIME:
                summary.tier1_unique++;
                summary.tier1_invocations += kernel.count;
                break;
            case Tier::CUBLAS:
                summary.tier2_unique++;
                summary.tier2_invocations += kernel.count;
                break;
            case Tier::LIBTORCH:
                summary.tier3_unique++;
                summary.tier3_invocations += kernel.count;
                break;
        }
    }
    
    return summary;
}

void KernelRegistry::print_summary() const {
    auto summary = get_summary();
    int total_invocations = summary.tier1_invocations + 
                           summary.tier2_invocations + 
                           summary.tier3_invocations;
    
    std::cout << "\nKernel Registry Summary:" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Total unique kernels: " << summary.total_kernels << std::endl;
    std::cout << "Total invocations: " << total_invocations << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tier 1 (CUDA Runtime):" << std::endl;
    std::cout << "  Unique kernels: " << summary.tier1_unique << std::endl;
    std::cout << "  Invocations: " << summary.tier1_invocations 
              << " (" << (100.0 * summary.tier1_invocations / total_invocations) 
              << "%)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tier 2 (cuBLAS):" << std::endl;
    std::cout << "  Unique kernels: " << summary.tier2_unique << std::endl;
    std::cout << "  Invocations: " << summary.tier2_invocations 
              << " (" << (100.0 * summary.tier2_invocations / total_invocations) 
              << "%)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tier 3 (libtorch):" << std::endl;
    std::cout << "  Unique kernels: " << summary.tier3_unique << std::endl;
    std::cout << "  Invocations: " << summary.tier3_invocations 
              << " (" << (100.0 * summary.tier3_invocations / total_invocations) 
              << "%)" << std::endl;
}

} // namespace kernel
