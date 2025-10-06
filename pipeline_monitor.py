#!/usr/bin/env python3
"""
Monitor the comprehensive pipeline execution progress.
"""
import os
import time
from pathlib import Path

def count_files_in_directory(directory):
    """Count files in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def monitor_pipeline():
    """Monitor pipeline progress."""
    print("🚀 ESPI Pipeline Monitor")
    print("=" * 50)
    
    # Expected file counts (approximate)
    expected_counts = {
        "W01": 3923,  # From previous analysis
        "W02": 4029,
        "W03": 2989
    }
    
    directories = {
        "W01": "C:/ESPI_TEMP/GPU_FULL2/W01_CLEAN_u16",
        "W02": "C:/ESPI_TEMP/GPU_FULL2/W02_CLEAN_u16", 
        "W03": "C:/ESPI_TEMP/GPU_FULL2/W03_CLEAN_u16"
    }
    
    while True:
        print(f"\n📊 Progress Update - {time.strftime('%H:%M:%S')}")
        
        all_complete = True
        for dataset, directory in directories.items():
            current_count = count_files_in_directory(directory)
            expected = expected_counts[dataset]
            progress = (current_count / expected * 100) if expected > 0 else 0
            
            status = "✅ COMPLETE" if current_count >= expected else f"🔄 {progress:.1f}%"
            print(f"  {dataset}: {current_count}/{expected} files {status}")
            
            if current_count < expected:
                all_complete = False
        
        if all_complete:
            print("\n🎉 All re-inference processes completed!")
            break
            
        print("\n⏳ Waiting 30 seconds...")
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor_pipeline()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")
