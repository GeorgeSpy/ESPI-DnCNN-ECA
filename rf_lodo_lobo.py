# C:\ESPI_DnCNN\rf_lodo_lobo.py
# Leave-One-Dataset-Out and Leave-One-Bin-Out analysis
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

IN_CSV = r"C:\ESPI_TEMP\features\labels_fixed_bins_enhanced.csv"
OUT_DIR = r"C:\ESPI_TEMP\RF_EVAL"
os.makedirs(OUT_DIR, exist_ok=True)

# Load and prepare data
df = pd.read_csv(IN_CSV)
df = df.replace([np.inf,-np.inf], np.nan).fillna(df.median(numeric_only=True))

# Feature selection
ALWAYS_EXCLUDE = {"class_name","class_id","label","target","dataset","set","split","group_id",
                  "file","filepath","path","relpath","name","id"}
LEAKAGE = [r"^freq$", r"^freq_hz$", r"^level_db$", r"^dist_"]
def is_leak(c): 
    import re
    return any(re.match(p,c) for p in LEAKAGE)

num_cols = [c for c in df.columns if c not in ALWAYS_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
pattern_cols = [c for c in num_cols if not is_leak(c)]

y = df["class_name"].astype(str).values
classes = sorted(np.unique(y))

# Get dataset and frequency info
if "freq_hz" in df: f = df["freq_hz"].astype(float)
elif "freq" in df: f = df["freq"].astype(float)
else: raise SystemExit("need freq_hz or freq")

datasets = df["dataset"].astype(str) if "dataset" in df.columns else ["UNK"] * len(df)
unique_datasets = sorted(datasets.unique())

print(f"Found datasets: {unique_datasets}")
print(f"Dataset sizes: {datasets.value_counts().to_dict()}")

def lodo_analysis():
    """Leave-One-Dataset-Out cross-validation"""
    print("\n=== Leave-One-Dataset-Out Analysis ===")
    
    X = df[pattern_cols].values
    results = []
    
    for test_dataset in unique_datasets:
        print(f"Testing on dataset: {test_dataset}")
        
        # Split: train on all others, test on this one
        train_mask = datasets != test_dataset
        test_mask = datasets == test_dataset
        
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"  Skipping {test_dataset} - no samples in train or test")
            continue
            
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train model
        rf = RandomForestClassifier(n_estimators=600, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            'test_dataset': test_dataset,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'accuracy': float(acc),
            'macro_f1': float(macro_f1)
        })
        
        print(f"  Train: {train_mask.sum()}, Test: {test_mask.sum()}, Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}")
    
    # Summary
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['macro_f1'] for r in results])
        print(f"\nLODO Average: Accuracy={avg_acc:.4f}, Macro-F1={avg_f1:.4f}")
        
        with open(os.path.join(OUT_DIR, "lodo_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def lobo_analysis():
    """Leave-One-Bin-Out cross-validation (5Hz bins)"""
    print("\n=== Leave-One-Bin-Out Analysis ===")
    
    X = df[pattern_cols].values
    
    # Create frequency bins (5Hz width)
    global f
    freq_bins = (f / 5.0).apply(np.floor).astype(int)
    unique_bins = sorted(freq_bins.unique())
    
    print(f"Found frequency bins: {unique_bins}")
    print(f"Bin sizes: {freq_bins.value_counts().to_dict()}")
    
    results = []
    
    for test_bin in unique_bins:
        print(f"Testing on bin: {test_bin*5}-{(test_bin+1)*5} Hz")
        
        # Split: train on all other bins, test on this one
        train_mask = freq_bins != test_bin
        test_mask = freq_bins == test_bin
        
        if train_mask.sum() < 10 or test_mask.sum() < 5:  # Need minimum samples
            print(f"  Skipping bin {test_bin} - insufficient samples")
            continue
            
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train model
        rf = RandomForestClassifier(n_estimators=600, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            'test_bin': int(test_bin),
            'freq_range': f"{test_bin*5}-{(test_bin+1)*5}",
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'accuracy': float(acc),
            'macro_f1': float(macro_f1)
        })
        
        print(f"  Train: {train_mask.sum()}, Test: {test_mask.sum()}, Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}")
    
    # Summary
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['macro_f1'] for r in results])
        print(f"\nLOBO Average: Accuracy={avg_acc:.4f}, Macro-F1={avg_f1:.4f}")
        
        with open(os.path.join(OUT_DIR, "lobo_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    print("=== LODO & LOBO Analysis ===")
    print(f"Dataset: {len(df)} samples, {len(classes)} classes")
    print(f"Pattern features: {len(pattern_cols)}")
    
    # Run analyses
    lodo_results = lodo_analysis()
    lobo_results = lobo_analysis()
    
    # Compare with standard CV
    print(f"\n=== Comparison with Standard CV ===")
    print(f"Standard CV (from previous): Pattern-Only ~90.15% accuracy")
    if lodo_results:
        lodo_acc = np.mean([r['accuracy'] for r in lodo_results])
        print(f"LODO: {lodo_acc:.4f} accuracy (expected drop due to dataset shift)")
    if lobo_results:
        lobo_acc = np.mean([r['accuracy'] for r in lobo_results])
        print(f"LOBO: {lobo_acc:.4f} accuracy (tests frequency generalization)")

if __name__ == "__main__":
    main()
