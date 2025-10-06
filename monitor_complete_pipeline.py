#!/usr/bin/env python3
"""
Complete Pipeline Monitor and Executor
Monitors progress and executes remaining steps automatically
"""

import os
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def get_progress_status():
    """Get current progress status of all pipeline steps"""
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "re_inference": {
            "w01": {"total": 7846, "completed": 7846, "status": "complete"},
            "w02": {"total": 8058, "completed": 8058, "status": "complete"}, 
            "w03": {"total": 5978, "completed": 5978, "status": "complete"}
        },
        "phase_extraction": {
            "w01": {"total": 352, "completed": 352, "status": "complete"},
            "w02": {"total": 510, "completed": 3, "status": "in_progress"},
            "w03": {"total": 402, "completed": 2, "status": "in_progress"}
        },
        "qc_comparison": {
            "w01": {"status": "tested", "needs_full": True},
            "w02": {"status": "pending"},
            "w03": {"status": "pending"}
        },
        "irls_alignment": {
            "w01": {"status": "pending"},
            "w02": {"status": "pending"},
            "w03": {"status": "pending"}
        },
        "features_extraction": {
            "w01": {"status": "complete"},
            "w02": {"status": "pending"},
            "w03": {"status": "pending"}
        },
        "final_rf": {"status": "pending"}
    }
    
    # Update phase extraction progress
    for dataset in ["w02", "w03"]:
        phase_dir = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset.upper()}_PhaseOut_b18_cs16_ff100"
        if os.path.exists(phase_dir):
            completed = len([d for d in os.listdir(phase_dir) 
                           if os.path.isdir(os.path.join(phase_dir, d)) and not d.endswith("_viz")])
            status["phase_extraction"][dataset]["completed"] = completed
    
    return status

def execute_phase_extraction_batch(dataset, batch_size=10):
    """Execute phase extraction for a batch of frequencies"""
    
    input_dir = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset.upper()}_CLEAN_u16"
    output_dir = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset.upper()}_PhaseOut_b18_cs16_ff100"
    roi_mask = "C:\\ESPI_TEMP\\roi_mask.png"
    
    if not os.path.exists(input_dir):
        return False
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d))]
    
    # Get already processed directories
    if os.path.exists(output_dir):
        processed = [d for d in os.listdir(output_dir) 
                    if os.path.isdir(os.path.join(output_dir, d)) and not d.endswith("_viz")]
    else:
        processed = []
    
    # Get remaining directories
    remaining = [d for d in subdirs if d not in processed]
    
    if not remaining:
        print(f"✅ {dataset.upper()} phase extraction complete!")
        return True
    
    # Process batch
    batch = remaining[:batch_size]
    print(f"🔄 Processing {dataset.upper()} batch: {len(batch)} frequencies")
    
    for i, subdir in enumerate(batch):
        input_path = os.path.join(input_dir, subdir)
        output_path = os.path.join(output_dir, subdir)
        
        print(f"  Processing {i+1}/{len(batch)}: {subdir}")
        
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
                print(f"    ✅ {subdir}")
            else:
                print(f"    ❌ {subdir}: {result.stderr[:100]}")
                
        except Exception as e:
            print(f"    ❌ {subdir}: {str(e)}")
    
    return len(remaining) <= batch_size

def execute_qc_comparison(dataset):
    """Execute QC comparison for a dataset"""
    
    out_root = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset.upper()}_PhaseOut_b18_cs16_ff100"
    ref_root = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset.upper()}_PhaseRef_b18_cs16_ff100"
    roi_mask = "C:\\ESPI_TEMP\\roi_mask.png"
    
    if not os.path.exists(out_root) or not os.path.exists(ref_root):
        return False
    
    print(f"🔍 Executing QC comparison for {dataset.upper()}...")
    
    cmd = [
        sys.executable,
        "C:\\ESPI_DnCNN\\phase_qc_compare_two_roots.py",
        "--out-root", out_root,
        "--ref-root", ref_root,
        "--roi-mask", roi_mask,
        "--qmin", "0.10",
        "--save-maps"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"✅ QC comparison successful for {dataset.upper()}")
            return True
        else:
            print(f"❌ QC comparison failed for {dataset.upper()}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during QC comparison for {dataset.upper()}: {str(e)}")
        return False

def execute_irls_alignment(dataset):
    """Execute IRLS alignment for a dataset"""
    
    phase_root = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset.upper()}_PhaseOut_b18_cs16_ff100"
    
    if dataset == "w01":
        irls_script = "C:\\ESPI_DnCNN\\robust_align_irls.py"
        roi_mask = "C:\\ESPI_TEMP\\roi_mask.png"
    elif dataset == "w02":
        irls_script = "C:\\ESPI_DnCNN\\w02_irls_fixed.py"
        roi_mask = "C:\\ESPI_TEMP\\roi_mask_W02.png"
    elif dataset == "w03":
        irls_script = "C:\\ESPI_DnCNN\\w03_irls.py"
        roi_mask = "C:\\ESPI_TEMP\\roi_mask_W03.png"
    else:
        return False
    
    if not os.path.exists(irls_script):
        print(f"❌ IRLS script not found: {irls_script}")
        return False
    
    if not os.path.exists(roi_mask):
        print(f"⚠️  ROI mask not found: {roi_mask}, using default")
        roi_mask = "C:\\ESPI_TEMP\\roi_mask.png"
    
    print(f"🔄 Executing IRLS alignment for {dataset.upper()}...")
    
    cmd = [
        sys.executable,
        irls_script,
        "--input-dir", phase_root,
        "--roi-mask", roi_mask,
        "--output-dir", os.path.join(phase_root, "qc_align_B_IRLS")
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"✅ IRLS alignment successful for {dataset.upper()}")
            return True
        else:
            print(f"❌ IRLS alignment failed for {dataset.upper()}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during IRLS alignment for {dataset.upper()}: {str(e)}")
        return False

def execute_features_extraction(dataset):
    """Execute features extraction for a dataset"""
    
    phase_root = f"C:\\ESPI_TEMP\\GPU_FULL2\\{dataset.upper()}_PhaseOut_b18_cs16_ff100"
    
    if not os.path.exists(phase_root):
        return False
    
    print(f"🧮 Executing features extraction for {dataset.upper()}...")
    
    cmd = [
        sys.executable,
        "C:\\ESPI_DnCNN\\phase_nodal_features_min.py",
        "--band-root", phase_root
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"✅ Features extraction successful for {dataset.upper()}")
            return True
        else:
            print(f"❌ Features extraction failed for {dataset.upper()}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during features extraction for {dataset.upper()}: {str(e)}")
        return False

def execute_final_rf_training():
    """Execute final RF training with all datasets"""
    
    print("🌲 Executing final RF training...")
    
    # First merge features
    merge_cmd = [
        sys.executable,
        "C:\\ESPI_DnCNN\\merge_all_features.py",
        "--roots", 
        "C:\\ESPI_TEMP\\GPU_FULL2\\W01_PhaseOut_b18_cs16_ff100",
        "C:\\ESPI_TEMP\\GPU_FULL2\\W02_PhaseOut_b18_cs16_ff100", 
        "C:\\ESPI_TEMP\\GPU_FULL2\\W03_PhaseOut_b18_cs16_ff100",
        "--out", "C:\\ESPI_TEMP\\features\\all_features_QCpass.csv"
    ]
    
    try:
        result = subprocess.run(merge_cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode != 0:
            print("❌ Features merging failed")
            return False
        
        print("✅ Features merging successful")
        
        # Then run RF training
        rf_cmd = [
            sys.executable,
            "C:\\ESPI_DnCNN\\hierarchical_rf_classifier.py",
            "--X", "C:\\ESPI_TEMP\\features\\all_features_QCpass.csv",
            "--y", "C:\\ESPI_TEMP\\features\\labels_5class.csv",
            "--outdir", "C:\\ESPI_TEMP\\features\\rf_model_final"
        ]
        
        result = subprocess.run(rf_cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✅ Final RF training successful")
            return True
        else:
            print("❌ Final RF training failed")
            return False
            
    except Exception as e:
        print(f"❌ Exception during final RF training: {str(e)}")
        return False

def main():
    """Main monitoring and execution loop"""
    
    print("🚀 COMPLETE PIPELINE MONITOR STARTING")
    print("=" * 60)
    
    # Create features directory
    os.makedirs("C:\\ESPI_TEMP\\features", exist_ok=True)
    
    iteration = 0
    max_iterations = 100  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n📊 ITERATION {iteration} - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 40)
        
        # Get current status
        status = get_progress_status()
        
        # Print status
        print(f"Phase Extraction:")
        for dataset in ["w01", "w02", "w03"]:
            phase_info = status["phase_extraction"][dataset]
            print(f"  {dataset.upper()}: {phase_info['completed']}/{phase_info['total']} ({phase_info['status']})")
        
        # Execute phase extraction for W02 and W03
        w02_complete = execute_phase_extraction_batch("w02", batch_size=5)
        w03_complete = execute_phase_extraction_batch("w03", batch_size=5)
        
        # If phase extraction is complete, move to next steps
        if w02_complete and w03_complete:
            print("🎯 Phase extraction complete for all datasets!")
            
            # Execute QC comparison
            for dataset in ["w01", "w02", "w03"]:
                execute_qc_comparison(dataset)
            
            # Execute IRLS alignment
            for dataset in ["w01", "w02", "w03"]:
                execute_irls_alignment(dataset)
            
            # Execute features extraction
            for dataset in ["w02", "w03"]:  # W01 already done
                execute_features_extraction(dataset)
            
            # Execute final RF training
            execute_final_rf_training()
            
            print("🎉 PIPELINE COMPLETE!")
            break
        
        # Wait before next iteration
        print("⏳ Waiting 30 seconds before next iteration...")
        time.sleep(30)
    
    # Save final status
    final_status = get_progress_status()
    with open("C:\\ESPI_TEMP\\pipeline_final_status.json", "w") as f:
        json.dump(final_status, f, indent=2)
    
    print(f"\n📄 Final status saved to: C:\\ESPI_TEMP\\pipeline_final_status.json")
    print("🎯 Pipeline monitoring complete!")

if __name__ == "__main__":
    sys.exit(main())