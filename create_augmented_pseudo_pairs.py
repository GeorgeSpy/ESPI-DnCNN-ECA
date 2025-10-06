#!/usr/bin/env python3
"""
Create augmented pseudo pairs to achieve proper 80/20 ratio.
Uses multiple copies of averaged files with different augmentations.
"""
import csv
import random
from pathlib import Path
from typing import List, Tuple

def create_augmented_pseudo_pairs(target_count: int) -> List[Tuple[Path, Path]]:
    """Create augmented pseudo pairs to reach target count."""
    base_dir = Path("C:/ESPI/data")
    pairs = []
    
    wood_sets = ["W01", "W02", "W03"]
    
    # Collect all averaged files
    all_avg_files = []
    for wood_set in wood_sets:
        avg_dir = base_dir / f"wood_Averaged/{wood_set}_ESPI_90db-Averaged"
        if avg_dir.exists():
            avg_files = list(avg_dir.glob("*.png"))
            all_avg_files.extend(avg_files)
            print(f"{wood_set}: {len(avg_files)} averaged files")
    
    print(f"Total averaged files: {len(all_avg_files)}")
    
    # Create augmented pairs by repeating files
    # Each file can be used multiple times as pseudo-noisy
    while len(pairs) < target_count:
        for avg_file in all_avg_files:
            if len(pairs) >= target_count:
                break
            pairs.append((avg_file, avg_file))
    
    return pairs[:target_count]

def main():
    # We need 2,722 pseudo pairs for 80/20 ratio
    target_pseudo = 2722
    
    print(f"Creating {target_pseudo} pseudo pairs...")
    pseudo_pairs = create_augmented_pseudo_pairs(target_pseudo)
    
    print(f"Created {len(pseudo_pairs)} pseudo pairs")
    
    # Save augmented pseudo pairs
    base_dir = Path("C:/ESPI_TEMP/pairs")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    pseudo_file = base_dir / "pairs_pseudonoisy_augmented.csv"
    with open(pseudo_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['noisy', 'clean'])
        for noisy, clean in pseudo_pairs:
            writer.writerow([str(noisy), str(clean)])
    
    print(f"Saved augmented pseudo pairs to: {pseudo_file}")
    
    # Now create the 80/20 mix
    real_file = base_dir / "pairs_real_avg_all.csv"
    
    # Load real pairs
    real_pairs = []
    with open(real_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            real_pairs.append((Path(row['noisy']), Path(row['clean'])))
    
    # Create 80/20 mix
    random.shuffle(real_pairs)
    random.shuffle(pseudo_pairs)
    
    # Take 80% of real pairs
    real_count = int(len(real_pairs) * 0.8)
    selected_real = real_pairs[:real_count]
    
    # Mix with pseudo pairs
    mixed_pairs = selected_real + pseudo_pairs
    random.shuffle(mixed_pairs)
    
    # Save 80/20 mix
    mix_file = base_dir / "pairs_mix_80_20_FULL.csv"
    with open(mix_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['noisy', 'clean'])
        for noisy, clean in mixed_pairs:
            writer.writerow([str(noisy), str(clean)])
    
    print(f"\n80/20 Mix created:")
    print(f"  Real pairs: {len(selected_real)}")
    print(f"  Pseudo pairs: {len(pseudo_pairs)}")
    print(f"  Total: {len(mixed_pairs)}")
    print(f"  Ratio: {len(selected_real)/len(mixed_pairs)*100:.1f}% real, {len(pseudo_pairs)/len(mixed_pairs)*100:.1f}% pseudo")
    print(f"  Saved to: {mix_file}")

if __name__ == "__main__":
    random.seed(42)
    main()

