#!/usr/bin/env python3
"""
Create proper 80/20 real:pseudo training pairs for better generalization.
"""
import csv
import random
from pathlib import Path
from typing import List, Tuple

def create_real_pairs() -> List[Tuple[Path, Path]]:
    """Create real noisy -> averaged clean pairs."""
    base_dir = Path("C:/ESPI/data")
    pairs = []
    
    wood_sets = ["W01", "W02", "W03"]
    
    for wood_set in wood_sets:
        print(f"Processing {wood_set} real pairs...")
        
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
                real_files = list(real_dir.rglob("*.png"))
                print(f"  {real_dir.name}: {len(real_files)} real files")
                
                for real_file in real_files:
                    # Extract frequency and db from real file path
                    real_name = real_file.name
                    parts = real_name.split('_')
                    if len(parts) >= 2:
                        freq_db = '_'.join(parts[:2])  # e.g., "0040Hz_90.0db"
                        avg_file = avg_dir / f"{freq_db}.png"
                        if avg_file.exists():
                            pairs.append((real_file, avg_file))
    
    return pairs

def create_pseudo_pairs() -> List[Tuple[Path, Path]]:
    """Create pseudo-noisy pairs using averaged files."""
    base_dir = Path("C:/ESPI/data")
    pairs = []
    
    wood_sets = ["W01", "W02", "W03"]
    
    for wood_set in wood_sets:
        print(f"Processing {wood_set} pseudo pairs...")
        
        # Averaged clean directory
        avg_dir = base_dir / f"wood_Averaged/{wood_set}_ESPI_90db-Averaged"
        
        if avg_dir.exists():
            avg_files = list(avg_dir.glob("*.png"))
            print(f"  {avg_dir.name}: {len(avg_files)} averaged files")
            
            # Use each averaged file as both noisy and clean (pseudo-noisy)
            for avg_file in avg_files:
                pairs.append((avg_file, avg_file))
    
    return pairs

def main():
    # Create separate datasets
    print("Creating real pairs...")
    real_pairs = create_real_pairs()
    
    print("\nCreating pseudo pairs...")
    pseudo_pairs = create_pseudo_pairs()
    
    print(f"\nReal pairs: {len(real_pairs)}")
    print(f"Pseudo pairs: {len(pseudo_pairs)}")
    
    # Save separate datasets
    base_dir = Path("C:/ESPI_TEMP/pairs")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Save real pairs
    real_file = base_dir / "pairs_real_avg_all.csv"
    with open(real_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['noisy', 'clean'])
        for noisy, clean in real_pairs:
            writer.writerow([str(noisy), str(clean)])
    
    # Save pseudo pairs
    pseudo_file = base_dir / "pairs_pseudonoisy_all.csv"
    with open(pseudo_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['noisy', 'clean'])
        for noisy, clean in pseudo_pairs:
            writer.writerow([str(noisy), str(clean)])
    
    print(f"\nSaved real pairs to: {real_file}")
    print(f"Saved pseudo pairs to: {pseudo_file}")
    
    # Calculate target for 80/20 mix
    target_pseudo = int(len(real_pairs) * 0.25)  # 20% of total = 25% of real
    print(f"\nFor 80/20 mix:")
    print(f"  Real pairs: {len(real_pairs)}")
    print(f"  Target pseudo: {target_pseudo}")
    print(f"  Available pseudo: {len(pseudo_pairs)}")
    print(f"  Enough pseudo: {len(pseudo_pairs) >= target_pseudo}")

if __name__ == "__main__":
    random.seed(42)
    main()

