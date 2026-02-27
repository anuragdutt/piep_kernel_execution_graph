#!/usr/bin/env python3
"""
Unified Power Logger - Combines WattsUp System Power + GPU Power + CPU Power

Logs power from:
1. Two WattsUp power monitors (/dev/ttyUSB0, /dev/ttyUSB1) - main + auxiliary system power
2. All NVIDIA GPUs via nvidia-smi
3. CPU via Intel RAPL (Running Average Power Limit)

Output CSV format:
timestamp,pm1_watts,pm2_watts,system_total_watts,cpu_watts,gpu0_watts,gpu1_watts,...,gpu_total_watts

All sources are polled in one loop at the same interval (e.g. 1s). system_total_watts = pm1 + pm2
(only written when BOTH WattsUp meters report in that poll; otherwise N/A). Serial reads often
return only one meter per cycle, so system_total has more N/As than GPU. Energy scripts integrate
using only valid system samples and interpolate across gaps (see ENERGY_AND_POWER_FLOW.md).

Usage:
    python unified_power_logger.py -o power_log.csv
    python unified_power_logger.py -o power_log.csv --interval 0.04  # 25 Hz sampling
"""

import argparse
import datetime
import signal
import subprocess
import sys
import time
import serial
from typing import List, Tuple, Optional

# WattsUp constants
EXTERNAL_MODE = "E"
INTERNAL_MODE = "I"
FULLHANDLING = 2

running = True


def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM to gracefully stop logging"""
    global running
    running = False
    print("\nStopping power logger...")


class WattsUpReader:
    """Read from WattsUp power meters via USB serial"""

    def __init__(self, ports: List[str], interval: float):
        """
        Initialize WattsUp readers

        Args:
            ports: List of serial ports (e.g., ['/dev/ttyUSB0', '/dev/ttyUSB1'])
            interval: Sampling interval in seconds
        """
        self.ports = ports
        self.interval = interval
        self.sockets = []

        for port in ports:
            try:
                s = serial.Serial(port, 115200, timeout=2)
                self.sockets.append(s)
                print(f"  Connected to WattsUp on {port}")
            except Exception as e:
                print(f"  Warning: Failed to open {port}: {e}")
                self.sockets.append(None)

    def configure(self):
        """Configure WattsUp meters to external logging mode"""
        interval_deciseconds = int(self.interval * 10)  # Convert to deciseconds

        for i, s in enumerate(self.sockets):
            if s is None:
                continue
            try:
                # Set external logging mode with specified interval
                cmd = f"#L,W,3,{EXTERNAL_MODE},,{interval_deciseconds};".encode()
                s.write(cmd)
                s.write(f"#O,W,1,{FULLHANDLING}".encode())
                time.sleep(0.1)
                print(f"  Configured WattsUp on {self.ports[i]}")
            except Exception as e:
                print(f"  Warning: Failed to configure {self.ports[i]}: {e}")

    def read_power(self) -> List[Optional[float]]:
        """
        Read power from all WattsUp meters

        Returns:
            List of power readings in watts (None if read failed)
        """
        powers = []

        for i, s in enumerate(self.sockets):
            if s is None:
                powers.append(None)
                continue

            try:
                line = s.readline()
                if line.startswith(b"#d"):
                    fields = line.split(b",")
                    if len(fields) > 5:
                        # Field 3 contains power in deciseconds (0.1W)
                        w = float(fields[3]) / 10.0
                        powers.append(w)
                    else:
                        powers.append(None)
                else:
                    powers.append(None)
            except Exception as e:
                powers.append(None)

        return powers

    def close(self):
        """Close all serial connections"""
        for s in self.sockets:
            if s is not None:
                try:
                    s.close()
                except:
                    pass


class GPUPowerReader:
    """Read GPU power via nvidia-smi"""

    @staticmethod
    def read_gpu_power() -> List[Tuple[int, Optional[float]]]:
        """
        Read power draw for all GPUs via nvidia-smi

        Returns:
            List of (gpu_index, power_watts) tuples
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                powers = []
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue

                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            gpu_idx = int(parts[0].strip())
                            raw = parts[1].strip()

                            if raw.upper() in ("N/A", "NA", ""):
                                power = None
                            else:
                                power = float(raw)

                            powers.append((gpu_idx, power))
                        except (ValueError, IndexError):
                            continue

                return powers
        except subprocess.TimeoutExpired:
            print("Warning: nvidia-smi timeout", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error reading GPU power: {e}", file=sys.stderr)

        return []


class CPUPowerReader:
    """Read CPU power via Intel RAPL (Running Average Power Limit)"""

    def __init__(self):
        """Initialize RAPL power reader"""
        self.rapl_domains = []
        self._discover_rapl_domains()
        self.last_energy = {}
        self.last_timestamp = {}

    def _discover_rapl_domains(self):
        """Discover available RAPL power domains"""
        import glob
        import os

        # Find all RAPL domains
        for domain_path in glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*/"):
            try:
                name_file = os.path.join(domain_path, "name")
                energy_file = os.path.join(domain_path, "energy_uj")

                if os.path.exists(name_file) and os.path.exists(energy_file):
                    with open(name_file, "r") as f:
                        name = f.read().strip()

                    self.rapl_domains.append(
                        {"name": name, "energy_file": energy_file, "path": domain_path}
                    )
            except Exception:
                continue

    def read_cpu_power(self) -> Optional[float]:
        """
        Read CPU package power via RAPL

        Returns:
            CPU power in watts (average since last read), or None if unavailable
        """
        if not self.rapl_domains:
            return None

        import time

        current_time = time.time()
        total_power = 0.0
        valid_readings = 0

        for domain in self.rapl_domains:
            try:
                energy_file = domain["energy_file"]
                domain_name = domain["name"]

                # Read current energy in microjoules
                with open(energy_file, "r") as f:
                    current_energy_uj = int(f.read().strip())

                # Calculate power if we have a previous reading
                if domain_name in self.last_energy:
                    energy_diff_uj = current_energy_uj - self.last_energy[domain_name]
                    time_diff_s = current_time - self.last_timestamp[domain_name]

                    # Handle counter wraparound (RAPL counters are typically 32-bit)
                    if energy_diff_uj < 0:
                        # Assume 32-bit counter max (~4.29 TJ)
                        energy_diff_uj += 2**32

                    if time_diff_s > 0:
                        # Power = Energy / Time (convert microjoules to watts)
                        power_watts = (energy_diff_uj / 1e6) / time_diff_s
                        total_power += power_watts
                        valid_readings += 1

                # Store current reading for next iteration
                self.last_energy[domain_name] = current_energy_uj
                self.last_timestamp[domain_name] = current_time

            except Exception:
                continue

        return total_power if valid_readings > 0 else None


def main():
    global running

    parser = argparse.ArgumentParser(
        description="Unified Power Logger (WattsUp + GPU + CPU)"
    )
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--wattsup-ports",
        nargs="+",
        default=["/dev/ttyUSB0", "/dev/ttyUSB1"],
        help="WattsUp USB serial ports (default: /dev/ttyUSB0 /dev/ttyUSB1)",
    )
    parser.add_argument(
        "--gpu-only", action="store_true", help="Log GPU power only (skip WattsUp)"
    )
    parser.add_argument(
        "--no-cpu", action="store_true", help="Skip CPU power monitoring (RAPL)"
    )
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print("Unified Power Logger")
    print("=" * 70)
    print(f"Output file: {args.output}")
    print(f"Sampling interval: {args.interval}s")

    # Initialize WattsUp readers
    wattsup_reader = None
    if not args.gpu_only:
        print(f"\nInitializing WattsUp monitors:")
        wattsup_reader = WattsUpReader(args.wattsup_ports, args.interval)
        wattsup_reader.configure()
        time.sleep(0.5)  # Let WattsUp stabilize

    # Initialize CPU power reader
    cpu_reader = None
    if not args.no_cpu:
        print(f"\nInitializing CPU power monitor:")
        cpu_reader = CPUPowerReader()
        if cpu_reader.rapl_domains:
            for domain in cpu_reader.rapl_domains:
                print(f"  RAPL domain: {domain['name']}")
            # Prime the CPU reader with initial reading
            cpu_reader.read_cpu_power()
            time.sleep(0.1)
        else:
            print("  Warning: No RAPL domains found, CPU power unavailable")
            cpu_reader = None

    # Check GPU availability
    print(f"\nDetecting GPUs:")
    gpu_powers = GPUPowerReader.read_gpu_power()
    if gpu_powers:
        for gpu_idx, power in gpu_powers:
            print(f"  GPU {gpu_idx}: {power if power else 'N/A'} W")
    else:
        print("  No GPUs detected")

    num_gpus = len(gpu_powers)

    print("\n" + "=" * 70)
    print("Starting power logging... (Press Ctrl+C to stop)")
    print("=" * 70 + "\n")

    # Open output file
    with open(args.output, "w") as f:
        # Write CSV header
        header_parts = ["timestamp"]

        if not args.gpu_only:
            for i in range(len(args.wattsup_ports)):
                header_parts.append(f"pm{i + 1}_watts")
            header_parts.append("system_total_watts")

        if cpu_reader:
            header_parts.append("cpu_watts")

        for i in range(num_gpus):
            header_parts.append(f"gpu{i}_watts")

        if num_gpus > 0:
            header_parts.append("gpu_total_watts")

        f.write(",".join(header_parts) + "\n")
        f.flush()

        sample_count = 0
        start_time = time.time()
        last_print = start_time

        while running:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            row = [timestamp]

            # Read WattsUp power
            if not args.gpu_only and wattsup_reader:
                pm_powers = wattsup_reader.read_power()

                # Add individual PM powers
                for pm_power in pm_powers:
                    row.append(f"{pm_power:.2f}" if pm_power is not None else "N/A")

                # System total = sum of ALL power supplies (pm1 + pm2). Only set when we have
                # readings from every port; otherwise we would record a single supply as "total".
                num_ports = len(args.wattsup_ports)
                valid_powers = [p for p in pm_powers if p is not None]
                system_total = sum(valid_powers) if len(valid_powers) == num_ports else None
                row.append(f"{system_total:.2f}" if system_total is not None else "N/A")

            # Read CPU power
            cpu_power = None
            if cpu_reader:
                cpu_power = cpu_reader.read_cpu_power()
                row.append(f"{cpu_power:.2f}" if cpu_power is not None else "N/A")

            # Read GPU power
            gpu_powers = GPUPowerReader.read_gpu_power()
            gpu_power_dict = {idx: power for idx, power in gpu_powers}

            # Add individual GPU powers
            for i in range(num_gpus):
                power = gpu_power_dict.get(i)
                row.append(f"{power:.2f}" if power is not None else "N/A")

            # Add GPU total power
            if num_gpus > 0:
                valid_gpu_powers = [p for _, p in gpu_powers if p is not None]
                gpu_total = sum(valid_gpu_powers) if valid_gpu_powers else None
                row.append(f"{gpu_total:.2f}" if gpu_total is not None else "N/A")

            # Write row to file
            f.write(",".join(row) + "\n")
            f.flush()
            sample_count += 1

            # Print status every 5 seconds
            current_time = time.time()
            if current_time - last_print >= 5.0:
                elapsed = current_time - start_time
                rate = sample_count / elapsed if elapsed > 0 else 0

                status_parts = [f"Samples: {sample_count}", f"Rate: {rate:.1f}/s"]

                if not args.gpu_only and wattsup_reader and pm_powers:
                    valid_powers = [p for p in pm_powers if p is not None]
                    if valid_powers:
                        status_parts.append(f"System: {sum(valid_powers):.1f}W")

                if cpu_power is not None:
                    status_parts.append(f"CPU: {cpu_power:.1f}W")

                if gpu_powers:
                    valid_gpu = [p for _, p in gpu_powers if p is not None]
                    if valid_gpu:
                        gpu_strs = [
                            f"GPU{i}: {gpu_power_dict[i]:.1f}W" if gpu_power_dict.get(i) is not None else f"GPU{i}: N/A"
                            for i in range(num_gpus)
                        ]
                        status_parts.append(" | ".join(gpu_strs))
                        status_parts.append(f"GPU total: {sum(valid_gpu):.1f}W")

                print(f"  {' | '.join(status_parts)}")
                last_print = current_time

            # Sleep until next sample
            time.sleep(args.interval)

    # Cleanup
    if wattsup_reader:
        wattsup_reader.close()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Logging complete!")
    print(f"  Total samples: {sample_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Average rate: {sample_count / elapsed:.1f} samples/s")
    print(f"  Output: {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
