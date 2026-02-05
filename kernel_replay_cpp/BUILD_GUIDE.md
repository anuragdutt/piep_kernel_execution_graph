# Build Guide - Quick Reference

## Build Status: ✅ Fixed Compilation Issues

The initial build had macro namespace issues which have been fixed.

## Quick Build

```bash
cd kernel_replay_cpp

# Update libtorch path in CMakeLists.txt (line 14)
# Set to your libtorch installation path
sed -i 's|/path/to/libtorch|/opt/libtorch|g' CMakeLists.txt

# Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Build Issues Fixed

### Issue 1: CUDA_CHECK Macro Namespace Error ✅ Fixed

**Error:**
```
error: expected unqualified-id before 'do'
note: in expansion of macro 'CUDA_CHECK'
   benchmark::CUDA_CHECK(cudaMalloc(&dst, bytes));
```

**Root cause:** Macros don't have namespaces. `benchmark::CUDA_CHECK` is invalid.

**Fix:** Moved `CUDA_CHECK` macro outside the `benchmark` namespace in `benchmark_utils.hpp`

### Issue 2: Missing Headers ✅ Fixed

Added missing includes:
- `<algorithm>` in `aggregation.cpp` (for `std::sort`)
- `<cstdlib>` in `main.cpp` (for `std::atoi`)
- `<chrono>`, `<iomanip>`, `<sstream>` in `full_model_benchmark.cpp` (for timestamps)

## Dependencies

### System Packages

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y cmake build-essential nlohmann-json3-dev

# If nlohmann-json3-dev not available, install manually:
wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
sudo mkdir -p /usr/local/include/nlohmann
sudo cp json.hpp /usr/local/include/nlohmann/
```

### libtorch

Download pre-built binary:

```bash
# For CUDA 12.1 (adjust for your CUDA version)
cd /tmp
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
sudo mv libtorch /opt/libtorch

# Update CMakeLists.txt to point to /opt/libtorch
```

### Python Dependencies

For the helper scripts:

```bash
# From parent directory
cd ..
source .venv/bin/activate
pip install pandas  # For energy calculation script
```

## Verify Build

After successful build:

```bash
cd build

# Check executable exists
ls -lh kernel_benchmark

# Check dependencies
ldd kernel_benchmark | grep torch
# Should show libtorch.so, libc10.so, etc.

# Quick test (without running full benchmark)
./kernel_benchmark --help
```

Expected output:
```
Usage: ./kernel_benchmark <mode> [options]

Modes:
  full      - Run full model benchmark only
  isolated  - Run isolated kernel benchmarks only
  compare   - Run both and compare (default)
...
```

## Common Build Errors

### "Could not find Torch"

**Fix:** Set CMAKE_PREFIX_PATH in CMakeLists.txt or via command line:

```bash
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
```

### "nlohmann/json.hpp: No such file"

**Fix:** Install nlohmann-json3-dev or download manually (see Dependencies section above)

### "undefined reference to cublasCreate"

**Fix:** Make sure CUDA Toolkit is installed and CMake found it:

```bash
# Check CUDA installation
nvcc --version

# Reinstall if needed
sudo apt install nvidia-cuda-toolkit

# Rebuild
cd build && rm -rf * && cmake .. && make -j$(nproc)
```

### "error: 'cudaMemcpyKind' has not been declared"

**Fix:** Make sure CUDA headers are included. Check that `cuda_runtime.h` is in your include path.

## Build from Scratch

If you want to start fresh:

```bash
cd kernel_replay_cpp
rm -rf build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
make -j$(nproc) VERBOSE=1  # Verbose mode shows full commands
```

## Next Steps After Successful Build

1. **Classify kernels:**
   ```bash
   cd ..  # Go to kernel_replay_cpp/
   python scripts/classify_kernels.py \
       --input ../bloom_unique_kernels_compute.jsonl \
       --output data/kernel_signatures.json
   ```

2. **Export model (if not done):**
   ```bash
   python scripts/export_model.py \
       --model bigscience/bloom-560m \
       --output ../bloom_560m_traced.pt
   ```

3. **Run benchmark:**
   ```bash
   cd build
   ./kernel_benchmark compare \
       --model ../../bloom_560m_traced.pt \
       --kernels ../data/kernel_signatures.json \
       --runs 100
   ```

4. **With power logging:**
   ```bash
   ../scripts/run_with_power_logging.sh \
       --model ../../bloom_560m_traced.pt \
       --kernels ../data/kernel_signatures.json \
       --runs 1000
   ```

## Build Time

Expected build times:
- **First build:** 2-5 minutes (compiles all sources)
- **Incremental builds:** 10-30 seconds (only changed files)
- **Clean rebuild:** 2-5 minutes

## Verification

After build, verify all components:

```bash
cd build

# 1. Check executable
file kernel_benchmark
# Output: kernel_benchmark: ELF 64-bit LSB executable, x86-64...

# 2. Check libtorch linkage
ldd kernel_benchmark | grep libtorch
# Should show: libtorch.so => /opt/libtorch/lib/libtorch.so

# 3. Check CUDA linkage
ldd kernel_benchmark | grep cuda
# Should show: libcudart.so, libcublas.so, etc.

# 4. Quick help test
./kernel_benchmark --help
# Should show usage information
```

All checks passing? You're ready to run benchmarks!
