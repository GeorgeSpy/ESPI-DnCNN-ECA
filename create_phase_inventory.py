"""Create comprehensive phase inventory for audit and recovery."""
import os
import re
import pandas as pd
from pathlib import Path

def main():
    root = Path("C:/ESPI_TEMP/GPU_FULL2")
    rows = []
    
    # Find all frequency directories
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        dirname = dirpath.name
        
        # Match frequency pattern: 0155Hz_90.0db
        if re.match(r'^\d{4}Hz_\d{2,3}\.0db$', dirname):
            # Skip _viz directories
            if '_viz' in str(dirpath):
                continue
            
            # Determine set (W01, W02, W03)
            parts = dirpath.parts
            set_name = None
            for part in parts:
                if re.match(r'W0\d_PhaseOut', part):
                    set_name = part
                    break
            
            if not set_name:
                continue
            
            # Check for various artifacts
            unwrapped_dir = dirpath / "phase_unwrapped_npy"
            has_unwrapped = unwrapped_dir.exists()
            
            empty_npy = False
            if has_unwrapped:
                npy_files = list(unwrapped_dir.glob("*.npy"))
                if npy_files:
                    # Check if any file is suspiciously small (<200 bytes)
                    empty_npy = any(f.stat().st_size < 200 for f in npy_files)
            
            has_features = (dirpath / "nodal_features.csv").exists()
            has_qc = (dirpath / "qc_align_B_IRLS" / "summary.json").exists()
            has_qc_done = (dirpath / ".qc.done").exists()
            
            rows.append({
                'set': set_name.split('_')[0],  # W01, W02, W03
                'freq_dir': dirname,
                'has_unwrapped': has_unwrapped,
                'has_features': has_features,
                'has_qc': has_qc,
                'has_qc_done': has_qc_done,
                'empty_npy': empty_npy,
                'full_path': str(dirpath)
            })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = "C:/ESPI_TEMP/phase_inventory.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Created inventory: {output_path}")
    print(f"\n=== INVENTORY SUMMARY ===")
    print(f"Total frequency directories: {len(df)}")
    print(f"With unwrapped phase: {df['has_unwrapped'].sum()}")
    print(f"With features: {df['has_features'].sum()}")
    print(f"With QC: {df['has_qc'].sum()}")
    print(f"With QC done marker: {df['has_qc_done'].sum()}")
    print(f"Empty/corrupted NPY: {df['empty_npy'].sum()}")
    print()
    print(f"Missing unwrapped: {len(df) - df['has_unwrapped'].sum()}")
    print(f"Missing features: {df['has_unwrapped'].sum() - df['has_features'].sum() - df['empty_npy'].sum()}")
    print(f"Missing QC: {df['has_unwrapped'].sum() - df['has_qc'].sum() - df['empty_npy'].sum()}")
    
    print(f"\n=== BY DATASET ===")
    for set_name in sorted(df['set'].unique()):
        subset = df[df['set'] == set_name]
        print(f"\n{set_name}:")
        print(f"  Total: {len(subset)}")
        print(f"  With unwrapped: {subset['has_unwrapped'].sum()}")
        print(f"  With features: {subset['has_features'].sum()}")
        print(f"  Empty NPY: {subset['empty_npy'].sum()}")

if __name__ == "__main__":
    main()

