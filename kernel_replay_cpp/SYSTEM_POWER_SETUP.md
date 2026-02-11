# System Power Measurement Setup

## Overview

This project now supports **total system power measurement** in addition to GPU-only power measurement. The WattsUp power probes are physically swapped between etracker1 and etracker2:

- **etracker1's probe** (`/dev/ttyUSB0`) → measures **etracker2's** system power
- **etracker2's probe** (`/dev/ttyUSB0`) → measures **etracker1's** system power

To measure etracker2's power while running benchmarks on etracker2, we SSH into etracker1 and read from its WattsUp probe.

## Architecture

```
etracker2 (this machine, 130.245.127.109)
├── Runs BLOOM benchmark + CUDA kernels
├── SSH → etracker1 to start/stop power logging
└── Fetches power log back periodically

etracker1 (130.245.127.111)
├── WattsUp probe /dev/ttyUSB0 → measures etracker2's power
└── Runs ppm.py to log power samples to CSV
```

## Files Created/Modified

### New Files

#### 1. `scripts/remote_power_logger.py`
Python script that:
- SSHes into etracker1
- Starts `ppm.py` on etracker1 to log power from `/dev/ttyUSB0`
- Periodically fetches the log file back to etracker2
- Handles cleanup on termination

**Usage:**
```bash
python3 remote_power_logger.py -o system_power.csv -i 1.0 --fetch-interval 5.0
```

#### 2. `scripts/run_with_system_power.sh`
Orchestration script (analogous to `run_with_gpu_power.sh`) that:
1. Starts remote power logger on etracker1
2. Runs the C++ benchmark (`kernel_benchmark compare`)
3. Stops the remote power logger
4. Calculates energy using `calculate_per_kernel_energy.py`

**Usage:**
```bash
cd kernel_replay_cpp/scripts
./run_with_system_power.sh --model ../bloom_560m_traced.pt --runs 1000
./run_with_system_power.sh --model ../bloom_560m_traced.pt --runs 1000 --idle-power 110.0
```

**Options:**
- `--interval <seconds>`: Sampling interval (default: 1.0s, WattsUp is ~1Hz)
- `--idle-power <watts>`: Idle/baseline power to subtract from measurements
- All other args passed to `kernel_benchmark`

#### 3. `scripts/measure_idle_power.py`
Utility to measure the system's idle/baseline power. Run this when the system is idle (no workloads) to establish a baseline for subtraction.

**Usage:**
```bash
python3 measure_idle_power.py --duration 60
python3 measure_idle_power.py --duration 120 --output idle_baseline.csv
```

**Output:**
```
Average power: 110.28W
Min power: 108.70W
Max power: 119.80W
Std deviation: 3.02W
```

Use the average power as the `--idle-power` argument for benchmarking.

### Modified Files

#### 4. `scripts/calculate_per_kernel_energy.py`
Added `--idle-power` argument to subtract baseline power from system power measurements.

**Rationale:** Total system power includes idle components (CPU idle, memory, fans, etc.). To isolate GPU/workload power:
```
workload_power = measured_power - idle_power
```

### Deprecated/Removed Files

The following files were removed as they used the **incorrect** local WattsUp probe:
- ❌ `scripts/run_with_power_logging.sh` - Used local probe (measures etracker1, not etracker2)
- ❌ `scripts/power_logger_headless.py` - Local probe reader
- ❌ `scripts/wattsup_logger.py` - Local probe reader

These scripts are **obsolete** due to the probe swap. Use `run_with_system_power.sh` instead.

## Quick Start

### Step 1: Measure Idle Power (one-time calibration)
```bash
cd kernel_replay_cpp/scripts
/home/pace/piep_kernel_execution_graph/.venv/bin/python3 measure_idle_power.py --duration 60
```

**Example output:**
```
Average power: 110.28W
```

### Step 2: Run Benchmark with System Power Measurement
```bash
./run_with_system_power.sh \
    --idle-power 110.28 \
    --model ../bloom_560m_traced.pt \
    --runs 1000
```

### Step 3: View Results
```bash
cat ../results/system_energy_report.json
```

**Example output:**
```json
{
  "full_model": {
    "energy_per_inference_j": 2.42
  },
  "isolated_kernels": {
    "predicted_energy_per_inference_j": 3.08
  },
  "comparison": {
    "error_percent": 27.1
  }
}
```

## Comparison: GPU Power vs System Power

| Method | Measurement | Sampling Rate | Baseline Subtraction |
|--------|-------------|---------------|---------------------|
| **GPU Power** (`run_with_gpu_power.sh`) | GPU only via nvidia-smi | ~25 Hz | Not needed (GPU-only) |
| **System Power** (`run_with_system_power.sh`) | Total system via WattsUp | ~1 Hz | Recommended (subtract idle) |

### When to Use Each:

**GPU Power:**
- More accurate for GPU-only workloads
- Higher sampling rate (25 Hz vs 1 Hz)
- Previous result: 27.1% over-prediction

**System Power:**
- Captures full system energy (GPU + CPU + memory + everything)
- Better for total energy accounting
- Requires idle power calibration for accuracy

## SSH Setup

The scripts use SSH keys for passwordless access to etracker1:

**SSH Key:** `/home/pace/piep_kernel_execution_graph/etracker1_key` (already in project)

**Connection:**
- Host: `pace@130.245.127.111`
- Key permissions: `chmod 600` (handled automatically)

## CSV Format

All power logs use compatible CSV formats:

**WattsUp format (system power):**
```csv
timestamp,id,power
2026-02-10 17:44:24.430999,0,110.5
2026-02-10 17:44:25.320786,1,111.2
```

**nvidia-smi format (GPU power):**
```csv
timestamp,gpu,power_w
2026-02-10 17:44:24.430,0,85.23
2026-02-10 17:44:24.470,0,86.15
```

Both formats are normalized by `calculate_per_kernel_energy.py` to a common `time,sum` format.

## Troubleshooting

### "Remote power logger failed to start"
**Cause:** Serial port contention (another process accessing `/dev/ttyUSB0` on etracker1)

**Solution:**
```bash
ssh -i /home/pace/piep_kernel_execution_graph/etracker1_key pace@130.245.127.111 \
    "pkill -f 'python.*ppm.py'"
```

### "No samples collected" in idle power measurement
**Cause:** Serial port busy or SSH connection issue

**Solution:** Wait 10 seconds between measurements for the serial port to settle.

### Large idle power (>120W)
**Cause:** System not actually idle (background processes, SSH tests, etc.)

**Solution:** 
1. Stop all workloads
2. Wait 30 seconds
3. Measure again

**Expected idle power:** 105-115W for etracker2

## Implementation Notes

### Python Environment
All scripts use the project's venv:
```
#!/home/pace/piep_kernel_execution_graph/.venv/bin/python3
```

### Error Handling
- Remote logger handles SSH disconnection gracefully
- Performs final log fetch on termination
- Cleans up remote processes via `pkill`

### Idle Power Subtraction Logic
```python
if args.idle_power is not None:
    power_df['sum'] = power_df['sum'] - args.idle_power
    # Clamp to zero (no negative power)
    power_df.loc[power_df['sum'] < 0, 'sum'] = 0.0
```

This ensures workload-only energy is calculated when total system power is measured.

## Expected Results

### Current Status (GPU Power - nvidia-smi):
- **Full model energy:** 2.42 J/inference
- **Predicted energy:** 3.08 J/inference  
- **Error:** 27.1% over-prediction

### Next: System Power (WattsUp):
Run with system power to compare against GPU-only measurements. The system power approach should capture:
- GPU power
- CPU power (for data loading, kernel launches)
- Memory power
- Other system components

Expected improvement in accuracy if CPU/memory power is significant during inference.

## Future Work

1. **Compare GPU-only vs System power:** Understand the contribution of non-GPU components
2. **Optimize idle subtraction:** Consider time-varying idle power (thermal effects)
3. **Multi-GPU support:** Extend to tensor-parallel benchmarks (Vicuna-7B on 2 GPUs)
4. **Direct RAPL integration:** Bypass external probes for microsecond-level CPU power measurement

---

**Last Updated:** 2026-02-10  
**Tested On:** etracker2 (Ubuntu, CUDA 12.x, BLOOM-560M)
