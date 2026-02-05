#!/usr/bin/env python3
"""Headless power meter logger for scripted use.

Simplified version of pm1.py that runs without curses.
Reads from a single WattsUp power meter on /dev/ttyUSB0.

Usage:
    python power_logger_headless.py -o power_log.csv
"""

import argparse
import datetime
import os
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
    print("\nStopping logger...")

class WattsUpHeadless:
    def __init__(self, port, interval):
        self.port = port
        self.interval = interval
        self.s = None
        
    def connect(self):
        try:
            self.s = serial.Serial(self.port, 115200, timeout=2)
            print(f"Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    def mode(self, runmode):
        # Same as pm1.py: #L,W,3,<mode>,,<interval>; then #O,W,1,FULLHANDLING for external
        if self.s:
            try:
                self.s.write(str.encode('#L,W,3,%s,,%d;' % (runmode, max(1, int(self.interval)))))
                if runmode == EXTERNAL_MODE:
                    self.s.write(str.encode('#O,W,1,%d' % FULLHANDLING))
            except Exception:
                pass
    
    def log(self, logfile):
        global running
        
        if not self.s:
            print("Not connected!")
            return
        
        print(f"Logging to {logfile}...")
        self.mode(EXTERNAL_MODE)
        
        with open(logfile, 'w') as o:
            o.write('time,id,power\n')
            o.flush()
            
            n = 0
            samples = 0
            
            while running:
                try:
                    line = self.s.readline()
                    
                    if line and line.startswith(b'#d'):
                        fields = line.split(b',')
                        if len(fields) > 5:
                            w = float(fields[3]) / 10
                            timestamp = datetime.datetime.now()
                            o.write(f'{timestamp},{n},{w}\n')
                            o.flush()
                            samples += 1
                            n += self.interval
                            
                            # Print occasional status
                            if samples % 10 == 0:
                                print(f"  Samples: {samples}, Last power: {w:.1f}W")
                                
                except serial.SerialException:
                    print("Serial connection lost!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        print(f"Logged {samples} samples to {logfile}")
    
    def close(self):
        if self.s:
            self.s.close()

def main():
    parser = argparse.ArgumentParser(description='Headless power meter logger')
    parser.add_argument('-o', '--outfile', default='power_log.csv', help='Output CSV file')
    parser.add_argument('-p', '--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('-s', '--interval', type=float, default=1.0, help='Sample interval')
    args = parser.parse_args()
    
    # Set up signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if not os.path.exists(args.port):
        print(f"Error: Serial port {args.port} not found!")
        print("Make sure the WattsUp meter is connected.")
        sys.exit(1)
    
    meter = WattsUpHeadless(args.port, args.interval)
    
    if meter.connect():
        meter.log(args.outfile)
        meter.close()
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
