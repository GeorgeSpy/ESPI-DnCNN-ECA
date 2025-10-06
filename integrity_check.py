"""Quick integrity check for critical files."""
import pandas as pd
import os
from pathlib import Path

def main():
    print("=== INTEGRITY CHECK (Post-Cleanup) ===")
    
    # Check critical files
    files = [
        "C:/ESPI_TEMP/denoise_finetune_GPU_FULLSET/checkpoints/best.pth",
        "C:/ESPI_TEMP/features/all_features_merged_complete.csv",
        "C:/ESPI_TEMP/features/labels_fixed_bins_complete.csv",
        "C:/ESPI_TEMP/label_map.json"
    ]
    
    print("\nChecking critical files...")
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)
            print(f"OK {os.path.basename(file)}: {size:.1f} MB")
        else:
            print(f"MISSING: {os.path.basename(file)}")
    
    # Check sample count and class distribution
    try:
        df = pd.read_csv("C:/ESPI_TEMP/features/labels_fixed_bins_complete.csv")
        print(f"\nTotal samples: {len(df)}")
        print("\nClass distribution:")
        class_dist = df["class_name"].value_counts()
        for class_name, count in class_dist.items():
            pct = 100.0 * count / len(df)
            print(f"  {class_name}: {count} ({pct:.1f}%)")
    except Exception as e:
        print(f"Error reading labels CSV: {e}")
    
    # Check for empty NPY files in key ranges
    print("\nChecking NPY integrity...")
    root = Path("C:/ESPI_TEMP/GPU_FULL2")
    key_ranges = [(320, 345), (500, 525), (540, 570), (715, 1190)]
    
    for min_freq, max_freq in key_ranges:
        corrupted = 0
        total = 0
        
        for freq_dir in root.rglob("*Hz_*.0db"):
            if "_viz" in str(freq_dir):
                continue
                
            freq_str = freq_dir.name[:4]
            try:
                freq = int(freq_str)
                if min_freq <= freq <= max_freq:
                    npy_dir = freq_dir / "phase_unwrapped_npy"
                    if npy_dir.exists():
                        for npy_file in npy_dir.glob("*.npy"):
                            total += 1
                            if npy_file.stat().st_size < 200:
                                corrupted += 1
                                print(f"  WARNING Corrupted: {freq_dir.name}/{npy_file.name} ({npy_file.stat().st_size} bytes)")
            except ValueError:
                continue
                
        if total > 0:
            pct = 100.0 * corrupted / total
            print(f"  Range {min_freq}-{max_freq} Hz: {corrupted}/{total} corrupted ({pct:.1f}%)")

if __name__ == "__main__":
    main()
