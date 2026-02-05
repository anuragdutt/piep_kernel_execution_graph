# Kernel execution profiling and replay

Project layout:
- **vicuna/** – Vicuna-7B TP profiling scripts and data (trace, unique kernels, etc.)
- **bloom/** – BLOOM-560M profiling scripts and data (trace, shapes, unique kernels, gantt)
- **kernel_replay_cpp/** – C++ kernel replay benchmark and GPU power/energy measurement

## Environment (use one env only)

Scripts require **one** Python environment (e.g. a venv) so that `torchrun` and the worker processes use the same Python and PyTorch. If you run with another env active, workers will use that env and you can get "no GPUs found" or wrong packages.

### Setup with pip (no conda)

1. **Create and activate a venv:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   ```

2. **Install PyTorch with CUDA** (match index to your driver; `nvidia-smi` shows your CUDA version):
   - **CUDA 12.2** → use `cu121`:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - **CUDA 12.4+** → use `cu124`:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
     ```

3. **Install the rest:**
   ```bash
   pip install -r requirements.txt
   ```

### Required packages (summary)

| Script | Packages |
|--------|----------|
| **vicuna_tp_profile.py** | PyTorch ≥2.1 (CUDA 12.x + NCCL), transformers, sentencepiece, accelerate. Requires 2 GPUs with TP. |
| **bloom_profile.py** | PyTorch ≥2.0 (CUDA), transformers. Runs single GPU (fits 11GB) or 2-GPU TP. |
| **plot_kernel_gantt.py** | PyTorch traces (input); matplotlib |
| **extract_unique_kernels.py** | stdlib only (reads trace JSON/CSV) |

## Running the profiler

1. **Activate the same env in the shell you use for torchrun:**
   ```bash
   source .venv/bin/activate
   ```
2. **Confirm Python and GPUs:**
   ```bash
   which python torchrun   # should be .../piep_kernel_execution_graph/.venv/bin/...
   nvidia-smi              # should list GPUs
   ```

### Option A: BLOOM-560M (fits 11GB single GPU, e.g. GTX 1080 Ti)

From repo root, activate venv then run from `bloom/`:
```bash
source .venv/bin/activate
cd bloom && ./run_bloom_profile.sh
```
Or run the Python commands inside `bloom/` with `--trace trace.json --shape-log bloom_shapes.jsonl` (outputs stay in `bloom/`).

### Option B: Vicuna-7B (requires 2× A6000 with TP, ~14GB per GPU)

From repo root:
```bash
source .venv/bin/activate
cd vicuna && ./run_vicuna_2gpu.sh
```
Or:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  vicuna/vicuna_tp_profile.py \
  --model lmsys/vicuna-7b-v1.5 \
  --prompt "Explain tensor parallelism in one paragraph." \
  --max-new-tokens 64 \
  --trace trace.json \
  --record-shapes \
  --shape-log linear_shapes.jsonl
```

If you see **"ProcessGroupNCCL is only supported with GPUs, no GPUs found"** or **"MPS client failed to connect"** (CUDA error 805):

1. **Same shell, correct env:** Use your venv in the shell that runs torchrun (`source .venv/bin/activate`), then run the command again.
2. **Must be on a GPU node:** The machine you run on must have GPUs. In the same shell, run:
   ```bash
   nvidia-smi
   ```
   If that shows no devices or fails, you are on a **login/head node** with no GPUs. Run the script **on a compute node that has GPUs**:
   - **SLURM:** Get a session on a GPU node, then run the torchrun command there:
     ```bash
     srun --gres=gpu:2 --pty bash
     # now you're on a GPU node
     source .venv/bin/activate
     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...
     ```
   - Or submit a batch job with `#SBATCH --gres=gpu:2` and run the same command in the job script.
3. **Env vars:** Set `CUDA_VISIBLE_DEVICES=0,1` (or your GPU indices) in the same shell as torchrun so worker processes inherit it.

### Pascal (GTX 1080 Ti) with PyTorch 1.12

If you use PyTorch 1.12+cu116 for Pascal, **transformers** from PyPI may require PyTorch ≥2.2 and disable the PyTorch backend. Install an older transformers that works with 1.12:

```bash
pip install 'transformers>=4.30,<4.36'
```

Then install the rest of `requirements.txt` (sentencepiece, accelerate, etc.). The script will run in single-GPU mode (no tensor parallel) and still produce kernel traces.

**VRAM (single-GPU / 1080 Ti):** Vicuna-7B in fp16 needs ~14GB. A 1080 Ti has 11GB. **Do not** `pip install bitsandbytes` when using PyTorch 1.12 (Pascal): current bitsandbytes requires PyTorch ≥2.3 and will upgrade your install, breaking 1080 Ti support. On Pascal without bitsandbytes the script falls back to fp16 and will OOM on 11GB—use a smaller model, or run on a machine with TP (e.g. 2× A6000). On PyTorch 2.3+ (e.g. A6000), `pip install bitsandbytes` enables 8-bit in single-GPU mode. Use `--no-load-in-8bit` to force fp16 when you have enough VRAM.

**If bitsandbytes already upgraded PyTorch:** Restore Pascal-compatible PyTorch with:
```bash
pip uninstall -y bitsandbytes torch torchvision torchaudio
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu116
```
Then do not install bitsandbytes again in this env.

### GPU compatibility (PyTorch wheels)

Current PyTorch pip wheels support only **compute capability 7.0 and above** (Volta, Turing, Ampere, Ada, etc.). They do **not** support Pascal (e.g. **GTX 1080 Ti**, sm_61).

If you see *"NVIDIA GeForce GTX 1080 Ti with CUDA capability sm_61 is not compatible with the current PyTorch installation"*:

- **Option A (recommended):** Run on a machine with GPUs that have compute capability ≥ 7.0 (e.g. RTX 20xx, 30xx, 40xx, A100, V100, etc.).
- **Option B:** Build PyTorch from source with `TORCH_CUDA_ARCH_LIST="6.1"` so Pascal is included (see [PyTorch docs](https://pytorch.org/get-started/locally/)).

Install **protobuf** if the tokenizer warns: `pip install protobuf` (it is already in `requirements.txt`).

---

## Kernel Replay Analysis

After generating traces, see the analysis documents for kernel replay strategies:

### BLOOM-560M Analysis

- **`BLOOM_KERNEL_ANALYSIS.md`** - Breakdown of 144 unique kernels found in BLOOM-560M trace
- **`BLOOM_DIRECT_REPLAY_ANALYSIS.md`** - How to replay kernels in C++/CUDA (with/without PyTorch)
- **`LIBTORCH_REPLAY_GUIDE.md`** - Step-by-step guide to replay BLOOM kernels using libtorch C++ API
- **`minimal_replay_example.cpp`** - Minimal working C++ example

### Quick Summary

**With libtorch C++ API (RECOMMENDED):**
- ✅ 100% kernel coverage (all 144 unique kernels replay exactly)
- ✅ ~2-4 hours of work (mostly API translation)
- ✅ Same performance as Python PyTorch
- See `LIBTORCH_REPLAY_GUIDE.md` for full walkthrough

**Without PyTorch (pure CUDA/cuBLAS):**
- ⚠️ ~29% can use standard APIs (cudaMemcpy, cublasGemmEx)
- ⚠️ ~67% need custom CUDA kernels (elementwise, LayerNorm, etc.)
- ⚠️ ~2-4 weeks of work
- See `BLOOM_DIRECT_REPLAY_ANALYSIS.md` for details
