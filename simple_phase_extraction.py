#!/usr/bin/env python3
"""
Simple Phase Extraction - No Unicode Characters
Processes W02 and W03 phase extraction efficiently
"""

import os
import subprocess
import sys
import time

def process_dataset(dataset_name, batch_size=10):
    """Process phase extraction for a dataset in batches"""
    
    input_dir = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset_name.upper()}_CLEAN_u16"
    output_dir = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset_name.upper()}_PhaseOut_b18_cs16_ff100"
    roi_mask = "C:\\ESPI_TEMP\\roi_mask.png"
    
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories (exclude _viz directories)
    subdirs = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d)) and not d.endswith("_viz")]
    
    # Get already processed directories (exclude _viz and other non-phase dirs)
    if os.path.exists(output_dir):
        processed = [d for d in os.listdir(output_dir) 
                    if os.path.isdir(os.path.join(output_dir, d)) and 
                    not d.endswith("_viz") and 
                    not d.startswith("qc_") and
                    not d.startswith("debug")]
    else:
        processed = []
    
    # Get remaining directories
    remaining = [d for d in subdirs if d not in processed]
    
    print(f"{dataset_name.upper()}: {len(processed)}/{len(subdirs)} completed, {len(remaining)} remaining")
    
    if not remaining:
        print(f"{dataset_name.upper()} phase extraction complete!")
        return True
    
    # Process batch
    batch = remaining[:batch_size]
    print(f"Processing {dataset_name.upper()} batch: {len(batch)} frequencies")
    
    for i, subdir in enumerate(batch):
        input_path = os.path.join(input_dir, subdir)
        output_path = os.path.join(output_dir, subdir)
        
        print(f"  {i+1}/{len(batch)}: {subdir}")
        
        cmd = [
            sys.executable,
            "C:\\ESPI_DnCNN\\phase_extract_fft_STRICT_FIXED.py",
            "--input-dir", input_path,
            "--output-dir", output_path,
            "--band", "18", "--center-suppress", "16", "--flatfield", "100",
            "--annulus", "8", "300", "--roi-mask", roi_mask, "--unwrap", "auto"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                print(f"    SUCCESS: {subdir}")
            else:
                print(f"    ERROR: {subdir}")
                
        except Exception as e:
            print(f"    EXCEPTION: {subdir} - {str(e)}")
    
    return len(remaining) <= batch_size

def main():
    """Main execution function"""
    
    print("SIMPLE PHASE EXTRACTION STARTING")
    print("=" * 50)
    
    start_time = time.time()
    
    # Process W02 and W03
    w02_complete = False
    w03_complete = False
    
    while not (w02_complete and w03_complete):
        if not w02_complete:
            w02_complete = process_dataset("W02", batch_size=5)
        
        if not w03_complete:
            w03_complete = process_dataset("W03", batch_size=5)
        
        if not (w02_complete and w03_complete):
            print("Waiting 10 seconds before next batch...")
            time.sleep(10)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 50)
    print(f"PHASE EXTRACTION COMPLETE")
    print(f"Total time: {duration/60:.1f} minutes")
    print("All datasets processed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
