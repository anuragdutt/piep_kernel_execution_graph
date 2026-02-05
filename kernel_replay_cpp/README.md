# BLOOM-560M Kernel Replay Benchmark

Hybrid C++/CUDA project that benchmarks BLOOM-560M kernels using:
- **Tier 1 (29% of invocations):** Direct CUDA Runtime APIs (`cudaMemcpy`, `cudaMemset`)
- **Tier 2 (22% of invocations):** Direct cuBLAS APIs (`cublasGemmEx`)
- **Tier 3 (71% of invocations):** libtorch fallback for PyTorch-native kernels

## Prerequisites

### Required
- **CUDA Toolkit** 11.6+ (tested with 12.1/12.2)
- **CMake** 3.18+
- **C++ Compiler** with C++17 support (g++ 7+, clang++ 5+)
- **libtorch** 2.0+ (download from https://pytorch.org/get-started/locally/)
- **nlohmann/json** library (header-only, included via package manager or manually)
- **Python 3.8+** with `transformers`, `torch` (for model export and kernel classification)

### Hardware
- NVIDIA GPU with compute capability 6.0+ (tested on GTX 1080 Ti and A6000)

## Setup

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt install cmake build-essential nlohmann-json3-dev

# Or manually clone nlohmann/json
cd /tmp
git clone https://github.com/nlohmann/json.git
sudo cp -r json/include/nlohmann /usr/local/include/
```

### 2. Download libtorch

```bash
# For CUDA 12.1
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
# This creates ./libtorch/ directory
```

### 3. Update CMakeLists.txt

Edit `CMakeLists.txt` and set the libtorch path:

```cmake
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/path/to/libtorch")
```

Replace `/path/to/libtorch` with your actual libtorch path (e.g., `/home/user/libtorch`).

### 4. Export BLOOM Model to TorchScript

From the parent directory:

```bash
cd ..  # Go to piep_kernel_execution_graph/
python kernel_replay_cpp/scripts/export_model.py \
    --model bigscience/bloom-560m \
    --output bloom_560m_traced.pt
```

This creates `bloom_560m_traced.pt` (~1.1 GB).

### 5. Classify Kernels

```bash
python kernel_replay_cpp/scripts/classify_kernels.py \
    --input bloom_unique_kernels_compute.jsonl \
    --output kernel_replay_cpp/data/kernel_signatures.json
```

This reads the unique kernels from your BLOOM trace and classifies them into 3 tiers.

## Build

```bash
cd kernel_replay_cpp
mkdir build && cd build
cmake ..
make -j8
```

This creates the `kernel_benchmark` executable.

## Usage

### Run Complete Benchmark (Recommended)

```bash
./kernel_benchmark compare \
    --model ../../bloom_560m_traced.pt \
    --kernels ../data/kernel_signatures.json \
    --seq-len 5 \
    --warmup 10 \
    --runs 100 \
    --output-dir ../results/
```

This will:
1. Run full BLOOM model inference (100 times)
2. Run isolated benchmarks for all 144 unique kernels
3. Compare predicted vs actual total latency
4. Generate `results/comparison_report.json`

### Run Full Model Only

```bash
./kernel_benchmark full \
    --model ../../bloom_560m_traced.pt \
    --seq-len 5 \
    --runs 100
```

### Run Isolated Kernels Only

```bash
./kernel_benchmark isolated \
    --kernels ../data/kernel_signatures.json
```

## Output Files

| File | Description |
|------|-------------|
| `results/full_model_timing.json` | Full BLOOM inference timing |
| `results/comparison_report.json` | Predicted vs actual comparison with tier breakdown |

### Example `comparison_report.json`

```json
{
  "full_model_inference_us": 12500.0,
  "predicted_from_kernels_us": 11800.0,
  "error_percent": 5.6,
  "tier_breakdown": {
    "tier1_cuda_runtime": {
      "method": "cudaMemcpy/cudaMemset",
      "unique_kernels": 68,
      "total_us": 450.0
    },
    "tier2_cublas": {
      "method": "cublasGemmEx",
      "unique_kernels": 23,
      "total_us": 7800.0
    },
    "tier3_libtorch": {
      "method": "torch::layer_norm/add/etc",
      "unique_kernels": 53,
      "total_us": 3550.0
    }
  }
}
```

## Architecture

```
kernel_replay_cpp/
├── CMakeLists.txt              # Build configuration
├── include/                    # Headers
│   ├── kernel_registry.hpp     # Kernel classification & dispatch
│   ├── cuda_kernels.hpp        # Tier 1: CUDA Runtime
│   ├── cublas_kernels.hpp      # Tier 2: cuBLAS
│   ├── libtorch_kernels.hpp    # Tier 3: libtorch
│   ├── benchmark_utils.hpp     # Timing utilities
│   └── energy_hooks.hpp        # Energy probe interface (placeholder)
├── src/                        # Implementation
│   ├── main.cpp                # CLI entry point
│   ├── kernel_registry.cpp     # Load & manage kernel signatures
│   ├── cuda_kernels.cpp        # Tier 1 implementation
│   ├── cublas_kernels.cpp      # Tier 2 implementation
│   ├── libtorch_kernels.cpp    # Tier 3 implementation
│   ├── full_model_benchmark.cpp# Full BLOOM inference
│   └── aggregation.cpp         # Compare sum vs total
├── scripts/                    # Python helpers
│   ├── export_model.py         # Export BLOOM to TorchScript
│   └── classify_kernels.py     # Classify kernels into tiers
├── data/                       # Generated data
│   └── kernel_signatures.json  # Classified kernels (from classify_kernels.py)
└── results/                    # Benchmark outputs
```

## Troubleshooting

### CMake can't find libtorch

Make sure `CMAKE_PREFIX_PATH` in `CMakeLists.txt` points to your libtorch directory. You can also set it via command line:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```

### Missing nlohmann/json

Install via package manager or manually:

```bash
# Manual install
cd /tmp
git clone https://github.com/nlohmann/json.git
sudo cp -r json/include/nlohmann /usr/local/include/
```

### CUDA architecture mismatch

Edit `CMakeLists.txt` and set the correct compute capability for your GPU:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)  # Adjust for your GPU
```

Common values:
- `61` = GTX 1080 Ti (Pascal)
- `70` = V100 (Volta)
- `75` = RTX 2080 (Turing)
- `80` = A100 (Ampere)
- `86` = RTX 3090, A6000 (Ampere)

### libtorch runtime errors

Make sure libtorch shared libraries are in your `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
./kernel_benchmark compare ...
```

## Next Steps

### Energy Measurement

The `energy_hooks.hpp` interface is currently a placeholder. To integrate a physical energy probe:

1. Implement the `EnergyProbe` interface in `include/energy_hooks.hpp`
2. Update `create_probe()` to return your probe implementation
3. Rebuild the project

The benchmark will automatically use the energy measurements in the output reports.

### Tuning Kernel Parameters

If the predicted time differs significantly from actual time, tune the kernel parameters:

1. Check `data/kernel_signatures.json` for extracted parameters
2. Update `scripts/classify_kernels.py` to improve M/N/K extraction for Tier 2 kernels
3. Cross-reference with `bloom_shapes.jsonl` for accurate tensor dimensions
4. Rebuild kernel signatures: `python scripts/classify_kernels.py ...`

## References

- Plan document: `../BLOOM_DIRECT_REPLAY_ANALYSIS.md`
- Original trace: `../trace.json`
- Unique kernels: `../bloom_unique_kernels_compute.jsonl`
- Kernel analysis: `../BLOOM_KERNEL_ANALYSIS.md`
