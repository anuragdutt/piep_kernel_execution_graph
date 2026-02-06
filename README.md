# Kernel execution profiling and replay

Profile LLM inference (BLOOM, Vicuna), extract CUDA kernel traces, and replay them in C++ with **GPU power/energy measurement** via nvidia-smi.

## Project layout

| Directory | Contents |
|-----------|----------|
| **bloom/** | BLOOM-560M profiling (PyTorch profiler), trace extraction, unique kernels JSONL, gantt plots |
| **vicuna/** | Vicuna-7B tensor-parallel profiling (2-GPU) |
| **kernel_replay_cpp/** | C++ benchmark: load BLOOM TorchScript, run full model + isolated kernel replay, correlate with GPU power log → per-kernel and full-model energy (J) |

## Environment

Use a single Python environment (venv) so that all scripts and workers see the same PyTorch/CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS

# PyTorch with CUDA (match your driver; nvidia-smi shows CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # or cu124

pip install -r requirements.txt
```

Requirements: `transformers`, `sentencepiece`, `accelerate`, `protobuf`, `matplotlib`, `numpy<2`.

## BLOOM workflow: profile → replay → energy

### 1. Profile BLOOM and extract kernels

```bash
source .venv/bin/activate
cd bloom
./run_bloom_profile.sh
```

Produces PyTorch trace and shape logs. Then extract unique compute kernels:

```bash
# From bloom/ or repo root
python bloom/extract_unique_kernels.py ...   # see bloom scripts; outputs bloom_unique_kernels_compute.jsonl
```

### 2. Export BLOOM to TorchScript (for C++)

```bash
source .venv/bin/activate
python kernel_replay_cpp/scripts/export_model.py --output bloom_560m_traced.pt
```

Uses HuggingFace `bigscience/bloom-560m`, traces with `torch.jit.trace`, saves a `.pt` file (~1.1 GB). The C++ benchmark loads this via libtorch.

### 3. Classify kernels into tiers

```bash
python kernel_replay_cpp/scripts/classify_kernels.py \
  --input bloom/bloom_unique_kernels_compute.jsonl \
  --output kernel_replay_cpp/data/kernel_signatures.json
```

Classifies each kernel as Tier 1 (CUDA memcpy/memset), Tier 2 (cuBLAS GEMM), or Tier 3 (libtorch ops).

### 4. Build C++ benchmark

See **kernel_replay_cpp/BUILD_GUIDE.md** and **kernel_replay_cpp/README.md**. Summary:

```bash
cd kernel_replay_cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j
```

### 5. Run benchmark with GPU power and get energy report

From `kernel_replay_cpp/scripts/` (or pass paths accordingly):

```bash
./run_with_gpu_power.sh --model ../bloom_560m_traced.pt --runs 1000
```

This script:

1. Starts a background **GPU power logger** (nvidia-smi, CSV log).
2. Runs **kernel_benchmark compare**: full BLOOM inference then isolated replay of each unique kernel (with timestamps).
3. Stops the logger.
4. Runs **calculate_per_kernel_energy.py**: integrates power over each kernel’s time window → per-kernel energy (measured); **predicted energy = measured kernels only** (no estimated).
5. Writes **results/gpu_energy_report.json** and prints an energy comparison summary (full model vs predicted from measured kernels).

Optional: `--gpu 0`, `--interval 0.04` (polling interval in seconds; ~25 Hz typical for nvidia-smi).

Full-model energy is also computed from the power log and shown in the C++ “Full Model Results” block when `POWER_LOG_PATH` is set (the script exports it).

## Vicuna (2-GPU tensor parallel)

```bash
source .venv/bin/activate
cd vicuna
./run_vicuna_2gpu.sh
```

Requires 2 GPUs and NCCL. See `vicuna/` for options.

## References

- **kernel_replay_cpp/README.md** – C++ project structure, tiers, usage.
- **kernel_replay_cpp/BUILD_GUIDE.md** – libtorch, CMake, dependencies.
- **kernel_replay_cpp/QUICKSTART.md** – Short runbook.
