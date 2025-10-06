#!/usr/bin/env python3
"""
Execute Features Extraction and Final RF Training
Complete pipeline from features to classification results
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_features_extraction(phase_root, dataset_name):
    """Run features extraction for a dataset"""
    
    if not os.path.exists(phase_root):
        print(f"❌ Phase root not found: {phase_root}")
        return False
    
    print(f"🧮 Starting features extraction for {dataset_name}...")
    
    cmd = [
        sys.executable,
        r"C:\ESPI_DnCNN\phase_nodal_features_min.py",
        "--band-root", phase_root
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"✅ Features extraction successful for {dataset_name}")
            return True
        else:
            print(f"❌ Features extraction failed for {dataset_name}")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Exception during features extraction for {dataset_name}: {str(e)}")
        return False

def run_merge_features():
    """Run features merging for all datasets"""
    
    print("🔗 Starting features merging...")
    
    cmd = [
        sys.executable,
        r"C:\ESPI_DnCNN\merge_all_features.py",
        "--roots", 
        r"C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100",
        r"C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100", 
        r"C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100",
        "--out", r"C:\ESPI_TEMP\features\all_features_QCpass.csv"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✅ Features merging successful")
            return True
        else:
            print("❌ Features merging failed")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Exception during features merging: {str(e)}")
        return False

def run_dedup_and_numeric():
    """Run deduplication and numeric features creation"""
    
    print("🔧 Starting deduplication and numeric features...")
    
    # Create features directory
    os.makedirs(r"C:\ESPI_TEMP\features", exist_ok=True)
    
    # Deduplication
    dedup_cmd = [
        sys.executable,
        r"C:\ESPI_DnCNN\simple_dedup.py",
        "--inp", r"C:\ESPI_TEMP\features\all_features_QCpass.csv",
        "--out", r"C:\ESPI_TEMP\features\all_features_QCpass_dedup.csv"
    ]
    
    # Numeric features
    numeric_cmd = [
        sys.executable,
        r"C:\ESPI_DnCNN\create_numeric_features.py",
        "--inp", r"C:\ESPI_TEMP\features\all_features_QCpass_dedup.csv",
        "--out", r"C:\ESPI_TEMP\features\features_numeric_only.csv"
    ]
    
    try:
        # Run deduplication
        result1 = subprocess.run(dedup_cmd, capture_output=True, text=True, 
                               encoding='utf-8', errors='ignore')
        
        if result1.returncode != 0:
            print("❌ Deduplication failed")
            print(f"   Error: {result1.stderr[:200]}...")
            return False
        
        # Run numeric features
        result2 = subprocess.run(numeric_cmd, capture_output=True, text=True, 
                               encoding='utf-8', errors='ignore')
        
        if result2.returncode != 0:
            print("❌ Numeric features creation failed")
            print(f"   Error: {result2.stderr[:200]}...")
            return False
        
        print("✅ Deduplication and numeric features successful")
        return True
        
    except Exception as e:
        print(f"❌ Exception during dedup/numeric: {str(e)}")
        return False

def run_labels_creation():
    """Run labels creation"""
    
    print("🏷️  Starting labels creation...")
    
    cmd = [
        sys.executable,
        r"C:\ESPI_DnCNN\create_labels_corrected.py",
        "--inp", r"C:\ESPI_TEMP\features\features_numeric_only.csv",
        "--out", r"C:\ESPI_TEMP\features\labels_5class.csv"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✅ Labels creation successful")
            return True
        else:
            print("❌ Labels creation failed")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Exception during labels creation: {str(e)}")
        return False

def run_final_rf_training():
    """Run final RF training with hierarchical approach"""
    
    print("🌲 Starting final RF training...")
    
    cmd = [
        sys.executable,
        r"C:\ESPI_DnCNN\hierarchical_rf_classifier.py",
        "--X", r"C:\ESPI_TEMP\features\features_numeric_only.csv",
        "--y", r"C:\ESPI_TEMP\features\labels_5class.csv",
        "--outdir", r"C:\ESPI_TEMP\features\rf_model_final"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✅ Final RF training successful")
            print(f"   Output: {result.stdout[-300:]}...")
            return True
        else:
            print("❌ Final RF training failed")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Exception during RF training: {str(e)}")
        return False

def main():
    """Main execution function"""
    
    # Dataset configurations
    datasets = [
        {
            "name": "W01",
            "phase_root": r"C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100"
        },
        {
            "name": "W02",
            "phase_root": r"C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100"
        },
        {
            "name": "W03", 
            "phase_root": r"C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"
        }
    ]
    
    print("🧮 FEATURES & RF TRAINING STARTING")
    print("=" * 50)
    
    start_time = time.time()
    
    # Step 1: Features extraction for each dataset
    print("📊 Step 1: Features Extraction")
    features_success = 0
    for dataset in datasets:
        if run_features_extraction(dataset["phase_root"], dataset["name"]):
            features_success += 1
    
    if features_success != len(datasets):
        print(f"⚠️  Only {features_success}/{len(datasets)} datasets processed")
        return 1
    
    # Step 2: Merge features
    print("\n🔗 Step 2: Features Merging")
    if not run_merge_features():
        return 1
    
    # Step 3: Dedup and numeric
    print("\n🔧 Step 3: Deduplication & Numeric Features")
    if not run_dedup_and_numeric():
        return 1
    
    # Step 4: Labels creation
    print("\n🏷️  Step 4: Labels Creation")
    if not run_labels_creation():
        return 1
    
    # Step 5: Final RF training
    print("\n🌲 Step 5: Final RF Training")
    if not run_final_rf_training():
        return 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 50)
    print(f"🎉 FEATURES & RF TRAINING COMPLETE")
    print(f"⏱️  Total time: {duration/60:.1f} minutes")
    print("🎯 All steps completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())