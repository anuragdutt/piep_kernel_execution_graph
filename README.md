# unique_kernels – Vicuna TP profiling and kernel extraction

## Environment (use one env only)

Scripts require **one** conda environment so that `torchrun` and the worker processes use the same Python and PyTorch. If you run with another env active (e.g. `irene-parallel`), workers will use that env and you can get "no GPUs found" or wrong packages.

### Option A: Create from spec

```bash
cd /home/adutt/kernel_profiling/unique_kernels
conda env create -f environment.yml
```

### Option B: Manual setup (existing kernel-prof)

Match `pytorch-cuda` to your driver (e.g. 12.4): `nvidia-smi` shows driver/CUDA version.

```bash
conda activate kernel-prof
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y "mkl=2023.1.0" mkl-service intel-openmp "numpy<2"
pip install -r requirements.txt
```

### Required packages (summary)

| Script | Packages |
|--------|----------|
| **vicuna_tp_profile.py** | PyTorch ≥2.1 (with CUDA 12.x + NCCL to match host), transformers, sentencepiece, accelerate |
| **plot_kernel_gantt.py** | PyTorch traces (input); matplotlib |
| **extract_unique_kernels.py** | stdlib only (reads trace JSON/CSV) |

## Running the profiler

1. **Activate the env in the same shell you use for torchrun:**
   ```bash
   source /home/adutt/anaconda3/etc/profile.d/conda.sh
   conda activate kernel-prof
   ```
2. **Confirm Python and GPUs:**
   ```bash
   which python torchrun   # should be .../envs/kernel-prof/bin/...
   nvidia-smi              # should list GPUs
   ```
3. **Run (on a machine with GPUs, e.g. 2x A6000):**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
     /home/adutt/kernel_profiling/unique_kernels/vicuna_tp_profile.py \
     --model lmsys/vicuna-7b-v1.5 \
     --prompt "Explain tensor parallelism in one paragraph." \
     --max-new-tokens 64 \
     --trace trace.json \
     --record-shapes \
     --shape-log linear_shapes.jsonl
   ```

If you see **"ProcessGroupNCCL is only supported with GPUs, no GPUs found"** or **"MPS client failed to connect"** (CUDA error 805):

1. **Same shell, correct env:** Use `kernel-prof` in the shell that runs torchrun (`conda activate kernel-prof`), then run the command again.
2. **Must be on a GPU node:** The machine you run on must have GPUs. In the same shell, run:
   ```bash
   nvidia-smi
   ```
   If that shows no devices or fails, you are on a **login/head node** (e.g. `guppy`) with no GPUs. You need to run the script **on a compute node that has GPUs**:
   - **SLURM:** Get a session on a GPU node, then run the torchrun command there:
     ```bash
     srun --gres=gpu:2 --pty bash
     # now you're on a GPU node
     conda activate kernel-prof
     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...
     ```
   - Or submit a batch job with `#SBATCH --gres=gpu:2` and run the same command in the job script.
3. **Env vars:** Set `CUDA_VISIBLE_DEVICES=0,1` (or your GPU indices) in the same shell as torchrun so worker processes inherit it.
