# Quick Start Guide

Get the kernel replay benchmark running in ~15 minutes.

## Prerequisites Checklist

- [ ] NVIDIA GPU with CUDA 11.6+ installed
- [ ] Python 3.8+ with PyTorch and transformers
- [ ] CMake 3.18+
- [ ] C++17 compiler (g++ 7+ or clang++ 5+)

## Step 1: Download libtorch (5 min)

```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
sudo mv libtorch /opt/libtorch  # Or any location you prefer
```

## Step 2: Install nlohmann/json (1 min)

```bash
# Option A: Package manager
sudo apt install nlohmann-json3-dev

# Option B: Manual
cd /tmp
wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
sudo mkdir -p /usr/local/include/nlohmann
sudo cp json.hpp /usr/local/include/nlohmann/
```

## Step 3: Build the Project (2 min)

```bash
cd kernel_replay_cpp

# Update CMakeLists.txt with your libtorch path
sed -i 's|/path/to/libtorch|/opt/libtorch|g' CMakeLists.txt

mkdir build && cd build
cmake ..
make -j$(nproc)
```

If build succeeds, you should see: `kernel_benchmark` executable in the build directory.

## Step 4: Prepare Data (5 min)

From the parent directory (`piep_kernel_execution_graph/`):

```bash
cd ..  # Go back to piep_kernel_execution_graph/

# Export BLOOM model (if not already done)
python kernel_replay_cpp/scripts/export_model.py \
    --model bigscience/bloom-560m \
    --output bloom_560m_traced.pt

# Classify kernels
python kernel_replay_cpp/scripts/classify_kernels.py \
    --input bloom_unique_kernels_compute.jsonl \
    --output kernel_replay_cpp/data/kernel_signatures.json
```

## Step 5: Run Benchmark (2 min)

```bash
cd kernel_replay_cpp/build

# Set library path
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Run complete benchmark
./kernel_benchmark compare \
    --model ../../bloom_560m_traced.pt \
    --kernels ../data/kernel_signatures.json \
    --seq-len 5 \
    --warmup 10 \
    --runs 50  # Use 50 runs for quick test
```

## Expected Output

```
=== BLOOM Kernel Replay Benchmark ===
Mode: compare
GPU: NVIDIA GeForce GTX 1080 Ti
Compute capability: 6.1
cuBLAS initialized successfully

=== Full Model Benchmark ===
Loading model from: ../../bloom_560m_traced.pt
Model loaded and moved to CUDA
...
Average time: 8234.5 us (8.23 ms)

=== Running Isolated Kernel Benchmarks ===
Progress: 144/144 kernels benchmarked

Tier 1 (CUDA Runtime): 68 kernels, 1303 invocations
Tier 2 (cuBLAS):       23 kernels, 3721 invocations  
Tier 3 (libtorch):     53 kernels, 12162 invocations

=== Comparison Summary ===
Actual (full model):      8234.5 us (8.23 ms)
Predicted (sum kernels):  7892.3 us (7.89 ms)
Error:                    342.2 us (4.2%)
```

## View Results

```bash
# View comparison report
cat ../results/comparison_report.json | jq .

# Key fields:
# - full_model_inference_us: Actual full model time
# - predicted_from_kernels_us: Sum of isolated kernel times
# - error_percent: Difference percentage
# - tier_breakdown: Time contribution by tier
```

## Troubleshooting

### "CUDA error: no kernel image available"

Your GPU compute capability may not be compiled. Edit `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 61)  # GTX 1080 Ti
# or 70, 75, 80, 86 for other GPUs
```

Then rebuild: `cd build && make clean && cmake .. && make -j$(nproc)`

### "error while loading shared libraries: libtorch.so"

Add libtorch to library path:

```bash
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

Or permanently add to `~/.bashrc`:

```bash
echo 'export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### "nlohmann/json.hpp: No such file or directory"

Install nlohmann-json3-dev or manually download json.hpp (see Step 2).

## Next Steps

- Read `README.md` for detailed documentation
- Tune kernel parameters in `data/kernel_signatures.json`
- Integrate energy probe in `include/energy_hooks.hpp`
- Run with more iterations for accurate measurements: `--runs 1000`
