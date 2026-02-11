#!/home/pace/piep_kernel_execution_graph/.venv/bin/python3
"""
Remote Power Logger for etracker2 (reads from etracker1's WattsUp probe)

The WattsUp probes are swapped:
  - etracker1's /dev/ttyUSB0 → measures etracker2's system power
  - etracker2's /dev/ttyUSB0 → measures etracker1's system power

This script SSHes into etracker1, starts the WattsUp power logger (ppm.py),
and periodically fetches the log file back to etracker2.

Usage:
    python remote_power_logger.py -o system_power.csv
    
The script runs until interrupted (SIGINT/SIGTERM), then copies the final log.
"""

import argparse
import os
import subprocess
import sys
import signal
import time
import datetime

# Remote host configuration (etracker1 has our power probe)
REMOTE_HOST = "130.245.127.111"
REMOTE_USER = "pace"
SSH_KEY = "/home/pace/piep_kernel_execution_graph/etracker1_key"
REMOTE_PPM_SCRIPT = "/home/pace/piep_optim/power_polling/ppm.py"
REMOTE_LOG_DIR = "/tmp/etracker2_power_logs"

running = True
ssh_proc = None
remote_log_path = None


def signal_handler(sig, frame):
    """Handle termination signals"""
    global running
    running = False
    print("\nStopping remote power logger...")


def ensure_remote_directory():
    """Ensure the remote log directory exists"""
    cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"mkdir -p {REMOTE_LOG_DIR}"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Error creating remote directory: {result.stderr.decode()}")
        sys.exit(1)


def start_remote_ppm_logger(remote_log_file, interval=1.0):
    """Start ppm.py on etracker1 in the background via SSH"""
    remote_path = os.path.join(REMOTE_LOG_DIR, remote_log_file)
    
    # Command to run on remote host: run ppm.py in logging mode
    remote_cmd = f"cd /home/pace/piep_optim/power_polling && python3 -u ppm.py -l -o '{remote_path}' -s {interval}"
    
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        remote_cmd
    ]
    
    print(f"Starting remote power logger on {REMOTE_HOST}:{remote_path}")
    print(f"Command: {' '.join(ssh_cmd)}")
    
    # Start SSH process in background
    proc = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE
    )
    
    return proc, remote_path


def stop_remote_ppm_logger():
    """Kill all ppm.py processes on etracker1"""
    kill_cmd = "pkill -f 'python3.*ppm.py'"
    
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        kill_cmd
    ]
    
    subprocess.run(ssh_cmd, capture_output=True)
    print("Stopped remote power logger")


def fetch_remote_log(remote_path, local_path):
    """Copy the power log from etracker1 to local filesystem"""
    scp_cmd = [
        "scp",
        "-i", SSH_KEY,
        "-q",
        "-o", "StrictHostKeyChecking=no",
        f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}",
        local_path
    ]
    
    result = subprocess.run(scp_cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Warning: Failed to fetch remote log: {result.stderr.decode()}")
        return False
    return True


def main():
    global running, ssh_proc, remote_log_path
    
    parser = argparse.ArgumentParser(description='Remote System Power Logger (via etracker1 WattsUp probe)')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file (local path)')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--fetch-interval', type=float, default=5.0, help='How often to fetch log from remote (default: 5s)')
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Ensure SSH key exists and has correct permissions
    if not os.path.exists(SSH_KEY):
        print(f"Error: SSH key not found: {SSH_KEY}")
        sys.exit(1)
    
    os.chmod(SSH_KEY, 0o600)
    
    print("="*70)
    print("Remote System Power Logger")
    print("="*70)
    print(f"Remote host: {REMOTE_HOST} (etracker1)")
    print(f"Remote probe: /dev/ttyUSB0 → measures etracker2's system power")
    print(f"Local output: {args.output}")
    print(f"Sampling interval: {args.interval}s")
    print(f"Fetch interval: {args.fetch_interval}s")
    print("="*70)
    
    # Ensure remote directory exists
    ensure_remote_directory()
    
    # Generate unique remote log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_log_file = f"power_etracker2_{timestamp}.csv"
    
    # Start remote logger
    ssh_proc, remote_log_path = start_remote_ppm_logger(remote_log_file, args.interval)
    
    # Wait for logger to start
    time.sleep(3)
    
    # Check if process started successfully
    if ssh_proc.poll() is not None:
        print("Error: Remote power logger failed to start!")
        stdout, stderr = ssh_proc.communicate()
        print(f"stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
        sys.exit(1)
    
    print(f"Remote logger started (PID: {ssh_proc.pid})")
    print(f"Fetching log every {args.fetch_interval}s to {args.output}")
    print("Press Ctrl+C to stop\n")
    
    # Periodically fetch the log file
    sample_count = 0
    fetch_count = 0
    last_fetch = time.time()
    
    try:
        while running:
            time.sleep(1)
            
            # Fetch log periodically
            if time.time() - last_fetch >= args.fetch_interval:
                if fetch_remote_log(remote_log_path, args.output):
                    fetch_count += 1
                    # Count samples in local file
                    if os.path.exists(args.output):
                        with open(args.output, 'r') as f:
                            sample_count = sum(1 for line in f) - 1  # -1 for header
                    
                    print(f"  Fetched log (fetch #{fetch_count}, {sample_count} samples)")
                
                last_fetch = time.time()
            
            # Check if remote process died
            if ssh_proc.poll() is not None:
                print("\nWarning: Remote logger process died!")
                break
    
    except KeyboardInterrupt:
        pass
    
    # Clean up
    print("\nCleaning up...")
    stop_remote_ppm_logger()
    
    if ssh_proc and ssh_proc.poll() is None:
        ssh_proc.terminate()
        try:
            ssh_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ssh_proc.kill()
            ssh_proc.wait()
    
    # Final fetch
    print("Performing final log fetch...")
    time.sleep(2)  # Give remote logger time to flush
    if fetch_remote_log(remote_log_path, args.output):
        print(f"Final log saved to: {args.output}")
        
        # Print summary
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                lines = f.readlines()
                sample_count = len(lines) - 1  # -1 for header
            print(f"Total samples: {sample_count}")
    else:
        print("Warning: Final fetch failed!")
        sys.exit(1)
    
    print("="*70)


if __name__ == '__main__':
    main()
