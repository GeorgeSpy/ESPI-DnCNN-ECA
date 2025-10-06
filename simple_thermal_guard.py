#!/usr/bin/env python3
"""
Simple Thermal Guard - CPU Usage Based Temperature Estimation
Monitors CPU usage and estimates temperature for thermal protection
"""

import psutil
import time
import subprocess
import argparse
from datetime import datetime

class SimpleThermalGuard:
    def __init__(self, temp_limit=77, check_interval=5, base_temp=20, temp_multiplier=0.4):
        self.temp_limit = temp_limit
        self.check_interval = check_interval
        self.base_temp = base_temp  # Base temperature at 0% CPU (lowered to 20°C)
        self.temp_multiplier = temp_multiplier  # Temperature increase per % CPU usage (lowered to 0.4)
        self.running = True

    def estimate_cpu_temperature(self, cpu_usage):
        """Estimate CPU temperature based on usage"""
        # Formula: Base temp + (CPU usage * multiplier)
        estimated_temp = self.base_temp + (cpu_usage * self.temp_multiplier)
        return estimated_temp

    def get_python_processes(self):
        """Get all running Python processes"""
        python_procs = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        python_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error getting Python processes: {e}")
        return python_procs

    def kill_python_processes(self):
        """Kill all Python processes"""
        python_procs = self.get_python_processes()
        killed_count = 0

        print(f"\n🚨 EMERGENCY SHUTDOWN - Estimated CPU temp exceeded {self.temp_limit}°C!")
        print(f"Found {len(python_procs)} Python processes to terminate...")

        for proc in python_procs:
            try:
                print(f"Killing Python process PID={proc.pid}: {proc.name()}")
                proc.terminate()
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Could not kill process {proc.pid}: {e}")

        # Wait and force kill if needed
        time.sleep(2)
        for proc in python_procs:
            try:
                if proc.is_running():
                    print(f"Force killing stubborn process PID={proc.pid}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        print(f"✅ Emergency shutdown complete. Terminated {killed_count} Python processes.")
        return killed_count

    def log_status(self, estimated_temp, cpu_usage):
        """Log current status with estimation details"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        python_count = len(self.get_python_processes())

        # Color coding based on temperature
        temp_status = ""
        if estimated_temp >= self.temp_limit:
            temp_status = " 🚨 CRITICAL!"
        elif estimated_temp >= self.temp_limit - 5:
            temp_status = " ⚠️  HIGH"
        elif estimated_temp >= self.temp_limit - 10:
            temp_status = " 🟡 WARM"

        status = f"[{timestamp}] Estimated: {estimated_temp:.1f}°C | CPU: {cpu_usage:.1f}% | Python procs: {python_count}{temp_status}"
        print(status)

    def calibrate(self, actual_temp):
        """Adjust estimation parameters based on actual temperature reading"""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        if cpu_usage > 5:  # Only calibrate when there's some CPU activity
            # Calculate what the multiplier should be
            temp_diff = actual_temp - self.base_temp
            new_multiplier = temp_diff / cpu_usage

            # Gradually adjust (moving average)
            self.temp_multiplier = (self.temp_multiplier * 0.8) + (new_multiplier * 0.2)
            print(f"📊 Calibration: Actual={actual_temp:.1f}°C, CPU={cpu_usage:.1f}%, New multiplier={self.temp_multiplier:.3f}")

    def monitor(self):
        """Main monitoring loop"""
        print(f"🔥 Simple Thermal Guard started")
        print(f"Temperature limit: {self.temp_limit}°C")
        print(f"Check interval: {self.check_interval}s")
        print(f"Estimation formula: {self.base_temp}°C + (CPU% × {self.temp_multiplier:.2f})")
        print(f"Monitoring... (Ctrl+C to stop)")
        print(f"💡 Compare with Core Temp and provide actual readings for calibration!\n")

        try:
            while self.running:
                # Get CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)

                # Estimate temperature
                estimated_temp = self.estimate_cpu_temperature(cpu_usage)

                # Log status
                self.log_status(estimated_temp, cpu_usage)

                # Check if estimated temperature exceeded limit
                if estimated_temp >= self.temp_limit:
                    killed = self.kill_python_processes()
                    if killed > 0:
                        print(f"\n🛡️  System protected! Monitor will continue running...")
                        print(f"You can restart Python processes when CPU usage drops\n")

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n⏹️  Thermal guard stopped by user")
            self.running = False
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")

def main():
    parser = argparse.ArgumentParser(description='Simple CPU Thermal Guard - Usage-based estimation')
    parser.add_argument('--temp-limit', type=float, default=77.0,
                       help='Temperature limit in Celsius (default: 77)')
    parser.add_argument('--check-interval', type=float, default=5.0,
                       help='Check interval in seconds (default: 5)')
    parser.add_argument('--base-temp', type=float, default=30.0,
                       help='Base temperature at 0%% CPU (default: 30)')
    parser.add_argument('--multiplier', type=float, default=0.6,
                       help='Temperature increase per %% CPU usage (default: 0.6)')
    parser.add_argument('--calibrate', type=float,
                       help='Provide actual temperature reading for calibration')

    args = parser.parse_args()

    guard = SimpleThermalGuard(
        temp_limit=args.temp_limit,
        check_interval=args.check_interval,
        base_temp=args.base_temp,
        temp_multiplier=args.multiplier
    )

    if args.calibrate:
        print(f"🔧 Calibration mode: Actual temperature = {args.calibrate}°C")
        guard.calibrate(args.calibrate)
        return

    guard.monitor()

if __name__ == "__main__":
    main()