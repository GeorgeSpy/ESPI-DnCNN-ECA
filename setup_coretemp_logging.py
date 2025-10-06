#!/usr/bin/env python3
"""
Core Temp Setup Helper - Enables logging for thermal guard integration
"""

import os
import sys
import subprocess
import winreg

def find_coretemp_path():
    """Find Core Temp installation path"""
    possible_paths = [
        r"C:\Program Files\Core Temp\Core Temp.exe",
        r"C:\Program Files (x86)\Core Temp\Core Temp.exe",
        r"C:\CoreTemp\Core Temp.exe"
    ]

    # Check registry
    try:
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
                                exe_path = os.path.join(install_location, "Core Temp.exe")
                                if os.path.exists(exe_path):
                                    return exe_path
                        except:
                            pass
                except:
                    break
    except:
        pass

    # Check common paths
    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def main():
    print("🔥 Core Temp Setup Helper")
    print("=" * 40)

    # Find Core Temp
    coretemp_path = find_coretemp_path()

    if not coretemp_path:
        print("❌ Core Temp not found!")
        print("\n📥 Please download and install Core Temp:")
        print("   https://www.alcpu.com/CoreTemp/")
        print("\n📋 Installation steps:")
        print("   1. Download Core Temp")
        print("   2. Install to default location")
        print("   3. Run this script again")
        return

    print(f"✅ Found Core Temp: {coretemp_path}")

    # Check if Core Temp is running
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Core Temp.exe'],
                              capture_output=True, text=True)
        is_running = "Core Temp.exe" in result.stdout
    except:
        is_running = False

    if is_running:
        print("✅ Core Temp is running")
    else:
        print("⚠️  Core Temp is not running")
        print("\n🚀 Starting Core Temp...")
        try:
            subprocess.Popen([coretemp_path])
            print("✅ Core Temp started")
        except Exception as e:
            print(f"❌ Failed to start Core Temp: {e}")
            return

    print("\n📊 Core Temp Configuration:")
    print("   1. Open Core Temp settings (Options -> Settings)")
    print("   2. Go to 'Logging' tab")
    print("   3. Enable 'Log to file'")
    print("   4. Set log interval to 1-5 seconds")
    print("   5. Note the log file location")
    print("\n🛡️ The thermal guard will automatically find and use the log file")

    # Check for existing log
    log_locations = [
        os.path.expanduser(r"~\Documents\CoreTemp.csv"),
        os.path.join(os.path.dirname(coretemp_path), "CoreTemp.csv")
    ]

    for log_path in log_locations:
        if os.path.exists(log_path):
            print(f"\n✅ Found existing log: {log_path}")
            # Read last line to verify it's working
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        print(f"   Last entry: {lines[-1].strip()}")
            except:
                pass
            break
    else:
        print("\n⚠️  No log file found yet - enable logging in Core Temp settings")

if __name__ == "__main__":
    main()