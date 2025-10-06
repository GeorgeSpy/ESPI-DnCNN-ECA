#!/usr/bin/env python3
"""
CPU Thermal Guard - Monitors CPU temperature and kills Python processes if overheating
Author: Claude Code Assistant
Usage: python cpu_thermal_guard.py [--temp-limit 77] [--check-interval 5]
"""

import psutil
import time
import subprocess
import argparse
import sys
from datetime import datetime

class CPUThermalGuard:
    def __init__(self, temp_limit=77, check_interval=5):
        self.temp_limit = temp_limit
        self.check_interval = check_interval
        self.running = True

    def get_cpu_temperature(self):
        """Get CPU temperature using Core Temp and other methods for Windows"""
        try:
            # Method 1: Core Temp shared memory (most accurate for Ryzen)
            temp = self._read_coretemp_shared_memory()
            if temp is not None:
                return temp

            # Method 2: Core Temp log file
            temp = self._read_coretemp_log()
            if temp is not None:
                return temp

            # Method 3: Try psutil sensors (works on some systems)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            for entry in entries:
                                if entry.current:
                                    return entry.current

            # Method 4: Try WMI MSAcpi_ThermalZoneTemperature
            try:
                import wmi
                w = wmi.WMI(namespace="root\\wmi")
                temperature_info = w.MSAcpi_ThermalZoneTemperature()[0]
                # Convert from tenth of Kelvin to Celsius
                temp_celsius = (temperature_info.CurrentTemperature / 10.0) - 273.15
                return temp_celsius
            except Exception as e:
                pass

            # Method 5: Try OpenHardwareMonitor (if available)
            try:
                import wmi
                w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                sensors = w.Sensor()
                for sensor in sensors:
                    if sensor.SensorType == 'Temperature' and 'cpu' in sensor.Name.lower():
                        return float(sensor.Value)
            except:
                pass

            print("Warning: Could not read CPU temperature from any source")
            return None

        except Exception as e:
            print(f"Error getting temperature: {e}")
            return None

    def _read_coretemp_shared_memory(self):
        """Read Core Temp data from shared memory"""
        try:
            import ctypes
            from ctypes import wintypes

            # Try to access Core Temp shared memory directly
            try:
                # Open Core Temp shared memory mapping
                hMapFile = ctypes.windll.kernel32.OpenFileMappingW(
                    0x0004,  # FILE_MAP_READ
                    False,
                    "CoreTempMappingObject"
                )

                if hMapFile:
                    # Map the shared memory
                    pBuf = ctypes.windll.kernel32.MapViewOfFile(
                        hMapFile,
                        0x0004,  # FILE_MAP_READ
                        0, 0, 0
                    )

                    if pBuf:
                        # Read Core Temp data structure
                        # First 4 bytes: uiLoad (should be non-zero if data is valid)
                        uiLoad = ctypes.c_uint.from_address(pBuf).value

                        if uiLoad > 0:
                            # Temperature data usually starts around offset 160-256
                            for offset in [160, 180, 256, 300]:
                                try:
                                    # Read temperature as float (4 bytes each)
                                    temp_addr = pBuf + offset
                                    temps = []

                                    # Read up to 16 temperature values
                                    for i in range(16):
                                        temp_val = ctypes.c_float.from_address(temp_addr + i*4).value
                                        if 20 < temp_val < 150:  # Valid temperature range
                                            temps.append(temp_val)

                                    if temps:
                                        return max(temps)  # Return hottest core
                                except:
                                    continue

                        ctypes.windll.kernel32.UnmapViewOfFile(pBuf)
                    ctypes.windll.kernel32.CloseHandle(hMapFile)

            except Exception:
                pass

            # Fallback: Try to read from Core Temp window title
            return self._read_coretemp_window_title()

        except Exception:
            pass

        return None

    def _read_coretemp_window_title(self):
        """Read temperature from Core Temp window title"""
        try:
            import ctypes
            from ctypes import wintypes

            # Find Core Temp window
            def enum_windows_proc(hwnd, lParam):
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
                    window_title = buffer.value

                    # Look for Core Temp in title
                    if "Core Temp" in window_title and "°C" in window_title:
                        # Extract temperature from title
                        # Typical format: "Core Temp 1.18.1 - CPU: 65°C"
                        import re
                        temps = re.findall(r'(\d+(?:\.\d+)?)°C', window_title)
                        if temps:
                            temp_values = [float(t) for t in temps if 20 < float(t) < 150]
                            if temp_values:
                                lParam.append(max(temp_values))
                return True

            temperatures = []
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, ctypes.POINTER(ctypes.c_int))
            enum_proc = EnumWindowsProc(enum_windows_proc)

            ctypes.windll.user32.EnumWindows(enum_proc, ctypes.cast(ctypes.pointer(ctypes.py_object(temperatures)), ctypes.POINTER(ctypes.c_int)))

            if temperatures:
                return temperatures[0]

        except Exception:
            pass

        return None

    def _read_coretemp_log(self):
        """Read Core Temp data from log file"""
        try:
            import os
            import glob

            # Common Core Temp log locations
            possible_paths = [
                os.path.expanduser(r"~\Documents\CoreTemp.csv"),
                r"C:\Program Files\Core Temp\CoreTemp.csv",
                r"C:\Program Files (x86)\Core Temp\CoreTemp.csv",
                r"C:\CoreTemp\CoreTemp.csv"
            ]

            # Also try to find Core Temp installation via registry or search
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                  r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall") as key:
                    for i in range(1024):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    if "Core Temp" in display_name:
                                        install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                        possible_paths.append(os.path.join(install_location, "CoreTemp.csv"))
                                except:
                                    pass
                        except:
                            break
            except:
                pass

            # Try to find any CoreTemp.csv files
            for drive in ['C:', 'D:', 'E:']:
                try:
                    files = glob.glob(f"{drive}\\**\\CoreTemp.csv", recursive=True)
                    possible_paths.extend(files)
                except:
                    pass

            for log_path in possible_paths:
                try:
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:  # Skip header
                                last_line = lines[-1].strip()
                                parts = last_line.split(',')
                                if len(parts) >= 3:
                                    # Usually format: Time,Core0,Core1,Core2,...
                                    temps = []
                                    for i in range(1, len(parts)):
                                        try:
                                            temp = float(parts[i])
                                            if 20 < temp < 150:  # Reasonable range
                                                temps.append(temp)
                                        except:
                                            continue
                                    if temps:
                                        return max(temps)  # Return hottest core
                except:
                    continue

        except Exception:
            pass

        return None

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

        print(f"\n🚨 EMERGENCY SHUTDOWN - CPU temperature exceeded {self.temp_limit}°C!")
        print(f"Found {len(python_procs)} Python processes to terminate...")

        for proc in python_procs:
            try:
                print(f"Killing Python process PID={proc.pid}: {proc.name()}")
                proc.terminate()
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Could not kill process {proc.pid}: {e}")

        # Wait a bit and force kill if needed
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

    def log_status(self, temp, cpu_usage):
        """Log current status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        python_count = len(self.get_python_processes())

        status = f"[{timestamp}] CPU: {temp:.1f}°C | Usage: {cpu_usage:.1f}% | Python procs: {python_count}"

        if temp >= self.temp_limit - 5:  # Warning at 72°C if limit is 77°C
            status += " ⚠️  APPROACHING LIMIT"

        print(status)

    def monitor(self):
        """Main monitoring loop"""
        print(f"🔥 CPU Thermal Guard started")
        print(f"Temperature limit: {self.temp_limit}°C")
        print(f"Check interval: {self.check_interval}s")
        print(f"Monitoring... (Ctrl+C to stop)\n")

        try:
            while self.running:
                # Get CPU temperature
                temp = self.get_cpu_temperature()

                if temp is None:
                    print("❌ Could not read CPU temperature - installing required packages...")
                    print("Try: pip install wmi pywin32")
                    time.sleep(self.check_interval)
                    continue

                # Get CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)

                # Log status
                self.log_status(temp, cpu_usage)

                # Check if temperature exceeded limit
                if temp >= self.temp_limit:
                    killed = self.kill_python_processes()
                    if killed > 0:
                        print(f"\n🛡️  System protected! Monitor will continue running...")
                        print(f"You can restart your Python processes when temperature drops below {self.temp_limit-5}°C\n")

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n⏹️  Thermal guard stopped by user")
            self.running = False
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")

def main():
    parser = argparse.ArgumentParser(description='CPU Thermal Guard - Protect system from overheating')
    parser.add_argument('--temp-limit', type=float, default=77.0,
                       help='Temperature limit in Celsius (default: 77)')
    parser.add_argument('--check-interval', type=float, default=5.0,
                       help='Check interval in seconds (default: 5)')
    parser.add_argument('--test', action='store_true',
                       help='Test temperature reading without monitoring')

    args = parser.parse_args()

    guard = CPUThermalGuard(temp_limit=args.temp_limit, check_interval=args.check_interval)

    if args.test:
        print("🧪 Testing temperature reading...")
        temp = guard.get_cpu_temperature()
        if temp:
            print(f"Current CPU temperature: {temp:.1f}°C")
        else:
            print("❌ Could not read CPU temperature")
            print("Try installing: pip install wmi pywin32")
        return

    guard.monitor()

if __name__ == "__main__":
    main()