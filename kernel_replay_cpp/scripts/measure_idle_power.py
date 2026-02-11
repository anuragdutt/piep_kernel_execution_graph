#!/home/pace/piep_kernel_execution_graph/.venv/bin/python3
"""
Measure Idle/Baseline System Power

Uses the remote WattsUp probe on etracker1 to measure etracker2's idle power.
Run this when the system is idle (no workloads running) to establish a baseline.

Usage:
    python measure_idle_power.py --duration 60
    python measure_idle_power.py --duration 120 --output idle_power.csv
"""

import argparse
import subprocess
import sys
import time
import signal
import os
import datetime

SSH_KEY = "/home/pace/piep_kernel_execution_graph/etracker1_key"
REMOTE_HOST = "130.245.127.111"
REMOTE_USER = "pace"

running = True


def signal_handler(sig, frame):
    global running
    running = False


def measure_idle_power(duration_seconds, output_file=None, interval=1.0):
    """Measure idle power by reading from etracker1's WattsUp probe"""
    
    # Ensure SSH key has correct permissions
    if not os.path.exists(SSH_KEY):
        print(f"Error: SSH key not found: {SSH_KEY}")
        sys.exit(1)
    
    os.chmod(SSH_KEY, 0o600)
    
    print("="*70)
    print("Idle System Power Measurement")
    print("="*70)
    print(f"Remote probe: etracker1:/dev/ttyUSB0 → measures etracker2's power")
    print(f"Duration: {duration_seconds}s")
    print(f"Sampling interval: {interval}s")
    print("="*70)
    print("\nMake sure the system is IDLE (no workloads running)!\n")
    
    # Create a temporary script file on the remote host
    remote_script_path = f"/tmp/measure_idle_{os.getpid()}.py"
    
    # Python script to run on etracker1 to read power samples
    remote_script = f"""import serial, datetime, time, signal, sys

running = True

def handler(sig, frame):
    global running
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

s = serial.Serial('/dev/ttyUSB0', 115200, timeout=3)
s.write(b'#L,W,3,E,,{int(interval)};')
s.write(b'#O,W,1,2')
time.sleep(0.5)

samples = []
start_time = time.time()
timeout = {duration_seconds}

while running and (time.time() - start_time) < timeout:
    line = s.readline()
    if line and line.startswith(b'#d'):
        fields = line.split(b',')
        if len(fields) > 5:
            w = float(fields[3]) / 10
            timestamp = datetime.datetime.now()
            samples.append(w)
            print(f'{{timestamp}},{{len(samples)}},{{w:.1f}}', flush=True)

s.close()

if samples:
    avg = sum(samples) / len(samples)
    print(f'# SUMMARY: {{len(samples)}} samples, avg={{avg:.2f}}W, min={{min(samples):.2f}}W, max={{max(samples):.2f}}W', file=sys.stderr)
"""
    
    # Write script to remote host
    with subprocess.Popen([
        "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"cat > {remote_script_path}"
    ], stdin=subprocess.PIPE) as proc:
        proc.communicate(remote_script.encode())
    
    # Run remote script via SSH
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"python3 {remote_script_path}"
    ]
    
    print("Starting measurement...")
    print(f"Collecting samples for {duration_seconds} seconds...\n")
    
    samples = []
    
    try:
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read output line by line
        log_file = None
        if output_file:
            log_file = open(output_file, 'w')
            log_file.write("timestamp,id,power\n")
        
        for line in proc.stdout:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse: timestamp,id,power
                parts = line.split(',')
                if len(parts) == 3:
                    try:
                        power = float(parts[2])
                        samples.append(power)
                        
                        # Print every 10th sample
                        if len(samples) % 10 == 0:
                            print(f"  Sample {len(samples)}: {power:.1f}W")
                        
                        # Write to log file
                        if log_file:
                            log_file.write(f"{line}\n")
                            log_file.flush()
                    except ValueError:
                        pass
        
        if log_file:
            log_file.close()
        
        # Get stderr (summary)
        stderr = proc.stderr.read()
        
        proc.wait()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        proc.terminate()
        proc.wait()
    
    # Clean up remote script
    subprocess.run([
        "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"rm -f {remote_script_path}"
    ], capture_output=True)
    
    # Calculate statistics
    if samples:
        avg_power = sum(samples) / len(samples)
        min_power = min(samples)
        max_power = max(samples)
        std_dev = (sum((x - avg_power)**2 for x in samples) / len(samples)) ** 0.5
        
        print("\n" + "="*70)
        print("IDLE POWER MEASUREMENT RESULTS")
        print("="*70)
        print(f"Samples collected: {len(samples)}")
        print(f"Duration: {duration_seconds}s")
        print(f"Average power: {avg_power:.2f}W")
        print(f"Min power: {min_power:.2f}W")
        print(f"Max power: {max_power:.2f}W")
        print(f"Std deviation: {std_dev:.2f}W")
        
        if output_file:
            print(f"\nLog saved to: {output_file}")
        
        print("\n" + "="*70)
        print("To use this baseline in your benchmark, run:")
        print(f"  ./run_with_system_power.sh --idle-power {avg_power:.2f} --model ... --runs ...")
        print("="*70)
        
        return avg_power
    else:
        print("\nError: No samples collected!")
        return None


def main():
    global running
    
    parser = argparse.ArgumentParser(description='Measure idle/baseline system power')
    parser.add_argument('-d', '--duration', type=float, default=60.0, 
                       help='Measurement duration in seconds (default: 60)')
    parser.add_argument('-o', '--output', default=None,
                       help='Optional: save raw samples to CSV file')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                       help='Sampling interval in seconds (default: 1.0)')
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Measure idle power
    avg_power = measure_idle_power(args.duration, args.output, args.interval)
    
    if avg_power is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
