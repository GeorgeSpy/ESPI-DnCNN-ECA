# C:\ESPI_DnCNN\rf_negative_control.py
# Negative control (label shuffle) and feature stability analysis
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance

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

# Groups for CV
if "freq_hz" in df: f = df["freq_hz"].astype(float)
elif "freq" in df: f = df["freq"].astype(float)
else: raise SystemExit("need freq_hz or freq")
dset = df["dataset"].astype(str) if "dataset" in df.columns else ["UNK"] * len(df)
groups = dset.astype(str)+"_"+(np.floor(f/5).astype(int)).astype(str)

def negative_control_test():
    """Test with shuffled labels - should give ~1/6 accuracy"""
    print("\n=== Negative Control: Label Shuffle Test ===")
    
    X = df[pattern_cols].values
    
    # Shuffle labels
    y_shuffled = np.random.RandomState(42).permutation(y)
    
    # Cross-validation with shuffled labels
    try:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y_shuffled, groups=groups)
    except Exception:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y_shuffled)
    
    accuracies = []
    macro_f1s = []
    
    for train_idx, test_idx in splits:
        rf = RandomForestClassifier(n_estimators=600, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=42)
        rf.fit(X[train_idx], y_shuffled[train_idx])
        y_pred = rf.predict(X[test_idx])
        
        acc = accuracy_score(y_shuffled[test_idx], y_pred)
        f1 = f1_score(y_shuffled[test_idx], y_pred, average='macro', zero_division=0)
        
        accuracies.append(acc)
        macro_f1s.append(f1)
    
    avg_acc = np.mean(accuracies)
    avg_f1 = np.mean(macro_f1s)
    expected_acc = 1.0 / len(classes)  # Random chance
    
    print(f"Shuffled labels accuracy: {avg_acc:.4f} (expected ~{expected_acc:.4f})")
    print(f"Shuffled labels macro-F1: {avg_f1:.4f}")
    print(f"Classes: {len(classes)} (random chance = {expected_acc:.4f})")
    
    result = {
        'test_type': 'label_shuffle',
        'shuffled_accuracy': float(avg_acc),
        'shuffled_macro_f1': float(avg_f1),
        'expected_random': float(expected_acc),
        'n_classes': len(classes),
        'interpretation': 'Low accuracy confirms no hidden leakage'
    }
    
    with open(os.path.join(OUT_DIR, "negative_control.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def feature_stability_analysis():
    """Test feature importance stability across different random seeds"""
    print("\n=== Feature Importance Stability ===")
    
    X = df[pattern_cols].values
    
    # Test with multiple seeds
    seeds = [42, 123, 456, 789, 999]
    all_importances = []
    
    for seed in seeds:
        print(f"Testing seed {seed}...")
        
        # Train model
        rf = RandomForestClassifier(n_estimators=600, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=seed)
        rf.fit(X, y)
        
        # Get feature importances
        importances = dict(zip(pattern_cols, rf.feature_importances_))
        all_importances.append(importances)
    
    # Calculate stability (correlation between importance rankings)
    importances_df = pd.DataFrame(all_importances)
    corr_matrix = importances_df.corr()
    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    # Get top features across all seeds
    avg_importances = importances_df.mean()
    top_features = avg_importances.sort_values(ascending=False).head(10)
    
    print(f"Average correlation between importance rankings: {avg_correlation:.4f}")
    print(f"Top 10 most stable features:")
    for i, (feat, imp) in enumerate(top_features.items(), 1):
        print(f"  {i:2d}. {feat}: {imp:.4f}")
    
    # Check which features are consistently in top-10
    top_10_consistency = {}
    for feat in pattern_cols:
        count = sum(1 for imp_dict in all_importances 
                   if feat in sorted(imp_dict.keys(), key=lambda x: imp_dict[x], reverse=True)[:10])
        top_10_consistency[feat] = count
    
    stable_features = [feat for feat, count in top_10_consistency.items() if count >= 4]  # 4/5 seeds
    
    print(f"\nFeatures consistently in top-10 (4+ seeds): {len(stable_features)}")
    for feat in stable_features:
        print(f"  - {feat}")
    
    result = {
        'stability_analysis': {
            'avg_correlation': float(avg_correlation),
            'top_10_features': {k: float(v) for k, v in top_features.items()},
            'consistently_stable': stable_features,
            'stability_count': {k: int(v) for k, v in top_10_consistency.items()}
        }
    }
    
    with open(os.path.join(OUT_DIR, "feature_stability.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def main():
    print("=== Negative Control & Feature Stability Analysis ===")
    print(f"Dataset: {len(df)} samples, {len(classes)} classes")
    print(f"Pattern features: {len(pattern_cols)}")
    
    # Run analyses
    negative_result = negative_control_test()
    stability_result = feature_stability_analysis()
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Negative control (shuffled): {negative_result['shuffled_accuracy']:.4f} accuracy")
    print(f"Expected random: {negative_result['expected_random']:.4f}")
    print(f"Feature stability correlation: {stability_result['stability_analysis']['avg_correlation']:.4f}")
    print(f"Consistently stable features: {len(stability_result['stability_analysis']['consistently_stable'])}")

if __name__ == "__main__":
    main()
