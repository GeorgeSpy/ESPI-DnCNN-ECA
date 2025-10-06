#!/usr/bin/env python3
"""
Build comprehensive training pairs from all available wood data.
Finds real noisy files in subdirectories and matches with averaged clean files.
"""
import csv
import random
from pathlib import Path
from typing import List, Tuple

def find_all_real_files(real_base_dir: Path) -> List[Path]:
    """Find all real noisy files recursively."""
    real_files = []
    for real_file in real_base_dir.rglob("*.png"):
        real_files.append(real_file)
    return real_files

def find_matching_avg_file(real_file: Path, avg_dir: Path) -> Path:
    """Find matching averaged file for a real noisy file."""
    # Extract frequency and db from real file path
    # e.g., "0040Hz_90.0db_00.png" -> "0040Hz_90.0db.png"
    real_name = real_file.name
    parts = real_name.split('_')
    if len(parts) >= 2:
        freq_db = '_'.join(parts[:2])  # e.g., "0040Hz_90.0db"
        avg_file = avg_dir / f"{freq_db}.png"
        return avg_file
    return None

def main():
    # Base directories
    base_dir = Path("C:/ESPI/data")
    output_file = Path("C:/ESPI_TEMP/comprehensive_training_pairs.csv")
    
    all_pairs = []
    
    # Process all wood datasets
    wood_sets = ["W01", "W02", "W03"]
    
    for wood_set in wood_sets:
        print(f"\nProcessing {wood_set}...")
        
        # Real noisy directories
        real_dirs = [
            base_dir / f"wood_real_A/{wood_set}_ESPI_90db",
            base_dir / f"wood_real_B/{wood_set}_ESPI_90db", 
            base_dir / f"wood_real_C/{wood_set}_ESPI_90db"
        ]
        
        # Averaged clean directory
        avg_dir = base_dir / f"wood_Averaged/{wood_set}_ESPI_90db-Averaged"
        
        if not avg_dir.exists():
            print(f"Warning: {avg_dir} not found, skipping {wood_set}")
            continue
            
        # Find all real files and match with averaged files
        for real_dir in real_dirs:
            if real_dir.exists():
                real_files = find_all_real_files(real_dir)
                print(f"  {real_dir.name}: {len(real_files)} real files")
                
                matched_pairs = 0
                for real_file in real_files:
                    avg_file = find_matching_avg_file(real_file, avg_dir)
                    if avg_file and avg_file.exists():
                        all_pairs.append((real_file, avg_file))
                        matched_pairs += 1
                
                print(f"    Matched: {matched_pairs} pairs")
    
    # Add some pseudo pairs for data augmentation
    print(f"\nAdding pseudo pairs for data augmentation...")
    pseudo_pairs = 0
    for wood_set in wood_sets:
        avg_dir = base_dir / f"wood_Averaged/{wood_set}_ESPI_90db-Averaged"
        if avg_dir.exists():
            avg_files = list(avg_dir.glob("*.png"))
            # Add 50% of averaged files as pseudo pairs
            for avg_file in avg_files[:len(avg_files)//2]:
                all_pairs.append((avg_file, avg_file))  # Same file for pseudo-noisy
                pseudo_pairs += 1
    
    print(f"Added {pseudo_pairs} pseudo pairs")
    
    # Shuffle all pairs
    random.shuffle(all_pairs)
    
    # Write to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['noisy', 'clean'])
        for noisy, clean in all_pairs:
            writer.writerow([str(noisy), str(clean)])
    
    print(f"\n[SUCCESS] Created {len(all_pairs)} training pairs")
    print(f"Saved to: {output_file}")
    
    # Show distribution by dataset
    print("\nDistribution by dataset:")
    for wood_set in wood_sets:
        count = sum(1 for noisy, _ in all_pairs if wood_set in str(noisy))
        print(f"  {wood_set}: {count} pairs")
    
    # Show distribution by type
    real_count = sum(1 for noisy, clean in all_pairs if noisy != clean)
    pseudo_count = sum(1 for noisy, clean in all_pairs if noisy == clean)
    print(f"\nDistribution by type:")
    print(f"  Real pairs: {real_count}")
    print(f"  Pseudo pairs: {pseudo_count}")

if __name__ == "__main__":
    random.seed(42)
    main()

