#pragma once

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace kernel {

/**
 * Kernel classification tiers
 */
enum class Tier {
    CUDA_RUNTIME = 1,  // cudaMemcpy, cudaMemset
    CUBLAS = 2,        // cublasGemmEx, cublasGemv
    LIBTORCH = 3       // torch::layer_norm, torch::add, etc.
};

/**
 * Kernel signature with all metadata needed for replay
 */
struct KernelSignature {
    std::string name;
    Tier tier;
    int count;  // Number of invocations in trace
    
    // Grid/block dimensions (from trace)
    std::vector<int> grid{1, 1, 1};
    std::vector<int> block{1, 1, 1};
    int shared_memory = 0;
    
    // Tier-specific parameters
    json params;  // Flexible storage for tier-specific data
    
    // Helper methods to extract tier-specific params
    
    // Tier 1 (CUDA Runtime) - memcpy/memset
    size_t get_bytes() const {
        return params.value("bytes", 0);
    }
    
    std::string get_memcpy_kind() const {
        return params.value("kind", "unknown");
    }
    
    // Tier 2 (cuBLAS) - GEMM dimensions
    int get_M() const {
        // Try "M" first (from bloom_shapes), then fall back to "M_estimate"
        if (params.contains("M")) return params["M"].get<int>();
        return params.value("M_estimate", 1024);
    }
    
    int get_N() const {
        if (params.contains("N")) return params["N"].get<int>();
        return params.value("N_estimate", 1024);
    }
    
    int get_K() const {
        if (params.contains("K")) return params["K"].get<int>();
        return params.value("K_estimate", 1024);
    }
    
    // Tier 3 (libtorch) - operation type
    std::string get_operation() const {
        return params.value("operation", "unknown");
    }
};

/**
 * Kernel registry - loads and manages kernel signatures
 */
class KernelRegistry {
public:
    /**
     * Load kernel signatures from JSON file
     */
    bool load_from_file(const std::string& path);
    
    /**
     * Get all kernels of a specific tier
     */
    std::vector<KernelSignature> get_tier_kernels(Tier tier) const;
    
    /**
     * Get all kernels
     */
    const std::vector<KernelSignature>& get_all_kernels() const { return kernels_; }
    
    /**
     * Get summary statistics
     */
    struct Summary {
        int total_kernels = 0;
        int tier1_unique = 0;
        int tier1_invocations = 0;
        int tier2_unique = 0;
        int tier2_invocations = 0;
        int tier3_unique = 0;
        int tier3_invocations = 0;
    };
    
    Summary get_summary() const;
    
    /**
     * Print summary to stdout
     */
    void print_summary() const;

private:
    std::vector<KernelSignature> kernels_;
    std::map<Tier, std::vector<size_t>> tier_indices_;  // Indices grouped by tier
};

} // namespace kernel
