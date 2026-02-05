#!/usr/bin/env python3
"""
WattsUp power meter logger (single or multiple ports).
Same protocol as pm1.py: configures meter with #L,W,3,E,,interval; then logs.
Power = fields[3]/10 (watts). CSV: time,id,pm1[,pm2,...],sum

Usage:
  python wattsup_logger.py --ports /dev/ttyUSB0 --interval 1 -o power.csv
  python wattsup_logger.py --ports /dev/ttyUSB0,/dev/ttyUSB1 --interval 1 -o power.csv
"""

import argparse
import datetime
import serial
import signal
import sys
import time

EXTERNAL_MODE = 'E'
FULLHANDLING = 2
running = True


def signal_handler(sig, frame):
    global running
    running = False
    print("\nStopping...")


class WattsUp:
    def __init__(self, ports, interval):
        self.sockets = [serial.Serial(port, 115200, timeout=2) for port in ports]
        self.interval = max(1.0, float(interval))

    def set_mode(self):
        """Configure each meter: external mode, report every interval seconds (same as pm1.py)."""
        for s in self.sockets:
            try:
                s.write(str.encode('#L,W,3,%s,,%d;' % (EXTERNAL_MODE, int(self.interval))))
                s.write(str.encode('#O,W,1,%d' % FULLHANDLING))
            except Exception as e:
                print("Warning: set_mode failed:", e)

    def fetch_power(self):
        powers = []
        for s in self.sockets:
            line = s.readline()
            if line and line.startswith(b'#d'):
                fields = line.split(b',')
                if len(fields) > 5:
                    powers.append(float(fields[3]) / 10)
        return powers

    def log(self, logfile):
        self.set_mode()
        n = 0
        cols = ','.join(f'pm{i + 1}' for i in range(len(self.sockets)))
        with open(logfile, 'w') as f:
            f.write('time,id,' + cols + ',sum\n')
            f.flush()
            print(f"Logging to {logfile} ({len(self.sockets)} meter(s), interval={self.interval}s). Ctrl+C to stop.")
            while running:
                powers = self.fetch_power()
                if powers:
                    total_power = sum(powers)
                    timestamp = datetime.datetime.now()
                    f.write(f"{timestamp},{n}," + ','.join(f"{p:.1f}" for p in powers) + f",{total_power:.1f}\n")
                    f.flush()
                    n += int(self.interval)
                time.sleep(self.interval)


def main():
    parser = argparse.ArgumentParser(description="WattsUp power logger (pm1-style protocol)")
    parser.add_argument("--ports", required=True, help="Comma-separated serial ports (e.g. /dev/ttyUSB0 or /dev/ttyUSB0,/dev/ttyUSB1)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default 1)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    ports = [p.strip() for p in args.ports.split(",")]
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        meter = WattsUp(ports, args.interval)
        meter.log(args.output)
    except serial.SerialException as e:
        print("Serial error:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
