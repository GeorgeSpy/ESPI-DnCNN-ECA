#!/usr/bin/env python3
"""
Build comprehensive training pairs from all available wood data.
Creates real noisy -> averaged clean pairs and pseudo-noisy -> averaged clean pairs.
"""
import csv
import random
from pathlib import Path
from typing import List, Tuple

def find_matching_files(real_dir: Path, avg_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching files between real noisy and averaged clean directories."""
    pairs = []
    
    # Get all real files
    real_files = list(real_dir.glob("*.png"))
    print(f"Found {len(real_files)} real files in {real_dir}")
    
    for real_file in real_files:
        # Try to find matching averaged file
        # Remove any suffixes like _00, _01, etc.
        base_name = real_file.stem
        # Remove frequency and db info to get base name
        parts = base_name.split('_')
        if len(parts) >= 2:
            # Keep frequency and db parts
            freq_db = '_'.join(parts[:2])  # e.g., "0040Hz_90.0db"
            avg_file = avg_dir / f"{freq_db}.png"
            
            if avg_file.exists():
                pairs.append((real_file, avg_file))
    
    return pairs

def create_pseudo_pairs(avg_dir: Path, num_pairs: int = 1000) -> List[Tuple[Path, Path]]:
    """Create pseudo-noisy pairs by using averaged files as both noisy and clean."""
    avg_files = list(avg_dir.glob("*.png"))
    pairs = []
    
    # Create pairs by randomly selecting files
    for _ in range(min(num_pairs, len(avg_files))):
        noisy_file = random.choice(avg_files)
        clean_file = noisy_file  # Same file for pseudo-noisy
        pairs.append((noisy_file, clean_file))
    
    return pairs

def main():
    # Base directories
    base_dir = Path("C:/ESPI/data")
    output_file = Path("C:/ESPI_TEMP/full_training_pairs.csv")
    
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
            
        # Find real pairs
        for real_dir in real_dirs:
            if real_dir.exists():
                pairs = find_matching_files(real_dir, avg_dir)
                all_pairs.extend(pairs)
                print(f"  {real_dir.name}: {len(pairs)} pairs")
        
        # Create pseudo pairs for this set
        pseudo_pairs = create_pseudo_pairs(avg_dir, num_pairs=500)
        all_pairs.extend(pseudo_pairs)
        print(f"  Pseudo pairs: {len(pseudo_pairs)}")
    
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

if __name__ == "__main__":
    random.seed(42)
    main()

