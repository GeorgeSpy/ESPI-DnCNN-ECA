#!/usr/bin/env python3
import pandas as pd
import pathlib
import argparse
import os

def find_features_files(root_dir):
    """Find all nodal_features.csv files in the directory tree"""
    features_files = []
    root_path = pathlib.Path(root_dir)
    
    for csv_file in root_path.rglob("nodal_features.csv"):
        # Extract dataset name from path (e.g., W01_PhaseOut_b18_cs16_ff100)
        path_parts = csv_file.parts
        dataset_name = None
        for part in path_parts:
            if part.startswith("W") and "_PhaseOut_" in part:
                dataset_name = part.split("_")[0]  # Extract W01, W02, W03
                break
        
        if dataset_name:
            features_files.append((csv_file, dataset_name))
    
    return features_files

def load_features_file(file_path, dataset_name):
    """Load a single features CSV file and add dataset info"""
    try:
        df = pd.read_csv(file_path)
        df["dataset"] = dataset_name
        df["source_file"] = str(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Merge all nodal features from datasets")
    parser.add_argument("--root", default="C:\\ESPI_TEMP\\GPU_FULL2", 
                       help="Root directory containing W01/W02/W03 PhaseOut directories")
    parser.add_argument("--out", default="C:\\ESPI_TEMP\\features\\all_features_merged.csv",
                       help="Output file for merged features")
    parser.add_argument("--qc-out", default="C:\\ESPI_TEMP\\features\\all_features_QCpass.csv",
                       help="Output file for QC-passed features")
    args = parser.parse_args()
    
    print(f"Searching for features files in: {args.root}")
    
    # Find all features files
    features_files = find_features_files(args.root)
    print(f"Found {len(features_files)} features files")
    
    if not features_files:
        print("No features files found!")
        return
    
    # Load all features
    all_features = []
    for file_path, dataset_name in features_files:
        print(f"Loading {dataset_name}: {file_path.name}")
        df = load_features_file(file_path, dataset_name)
        if df is not None:
            all_features.append(df)
    
    if not all_features:
        print("No features could be loaded!")
        return
    
    # Merge all features
    merged_df = pd.concat(all_features, ignore_index=True)
    print(f"Total features loaded: {len(merged_df)}")
    
    # Show summary by dataset
    print("\nFeatures by dataset:")
    print(merged_df["dataset"].value_counts())
    
    # Save merged features
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged_df.to_csv(args.out, index=False)
    print(f"\nSaved merged features to: {args.out}")
    
    # Apply basic QC if QC columns exist
    qc_columns = ["rmse", "pct_pi2", "rmse_unwrapped"]
    available_qc = [col for col in qc_columns if col in merged_df.columns]
    
    if available_qc:
        print(f"\nApplying QC filters using columns: {available_qc}")
        
        # Basic QC thresholds
        qc_mask = pd.Series([True] * len(merged_df))
        
        if "rmse" in merged_df.columns:
            qc_mask &= (merged_df["rmse"] <= 4.0)
        if "rmse_unwrapped" in merged_df.columns:
            qc_mask &= (merged_df["rmse_unwrapped"] <= 4.0)
        if "pct_pi2" in merged_df.columns:
            qc_mask &= (merged_df["pct_pi2"] <= 50.0)
        
        qc_passed = merged_df[qc_mask].copy()
        print(f"QC passed: {len(qc_passed)} / {len(merged_df)} ({len(qc_passed)/len(merged_df)*100:.1f}%)")
        
        # Save QC-passed features
        qc_passed.to_csv(args.qc_out, index=False)
        print(f"Saved QC-passed features to: {args.qc_out}")
        
        # Show QC retention by dataset
        print("\nQC retention by dataset:")
        print(qc_passed["dataset"].value_counts())
    else:
        print("\nNo QC columns found - saving all features as QC-passed")
        merged_df.to_csv(args.qc_out, index=False)

if __name__ == "__main__":
    main()
