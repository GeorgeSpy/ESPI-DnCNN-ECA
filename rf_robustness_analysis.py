# C:\ESPI_DnCNN\rf_robustness_analysis.py
# Bootstrap CI, Confusion Matrices, Per-class Metrics
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

IN_CSV = r"C:\ESPI_TEMP\features\labels_fixed_bins_enhanced.csv"
OUT_DIR = r"C:\ESPI_TEMP\RF_EVAL"
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(IN_CSV)
df = df.replace([np.inf,-np.inf], np.nan).fillna(df.median(numeric_only=True))

# Feature selection (same as before)
ALWAYS_EXCLUDE = {"class_name","class_id","label","target","dataset","set","split","group_id",
                  "file","filepath","path","relpath","name","id"}
LEAKAGE = [r"^freq$", r"^freq_hz$", r"^level_db$", r"^dist_"]
def is_leak(c): 
    import re
    return any(re.match(p,c) for p in LEAKAGE)

num_cols = [c for c in df.columns if c not in ALWAYS_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
pattern_cols = [c for c in num_cols if not is_leak(c)]
hybrid_cols = list(num_cols)

y = df["class_name"].astype(str).values
classes = sorted(np.unique(y))

# Groups for CV
if "freq_hz" in df: f = df["freq_hz"].astype(float)
elif "freq" in df: f = df["freq"].astype(float)
else: raise SystemExit("need freq_hz or freq")
dset = df["dataset"] if "dataset" in df else "UNK"
groups = dset.astype(str)+"_"+(np.floor(f/5).astype(int)).astype(str)

def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence intervals for accuracy and macro-F1"""
    n_samples = len(y_true)
    accuracies = []
    macro_f1s = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        
        acc = accuracy_score(y_true_boot, y_pred_boot)
        f1 = f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        
        accuracies.append(acc)
        macro_f1s.append(f1)
    
    # Calculate percentiles
    alpha = 1 - confidence
    acc_ci = np.percentile(accuracies, [100*alpha/2, 100*(1-alpha/2)])
    f1_ci = np.percentile(macro_f1s, [100*alpha/2, 100*(1-alpha/2)])
    
    return {
        'accuracy': {'mean': np.mean(accuracies), 'ci': acc_ci.tolist()},
        'macro_f1': {'mean': np.mean(macro_f1s), 'ci': f1_ci.tolist()}
    }

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{title} - Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_setup(setup_name, feature_cols):
    """Complete analysis for one setup"""
    print(f"\n=== Analyzing {setup_name} ===")
    
    X = df[feature_cols].values
    
    # Cross-validation predictions
    try:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups=groups)
    except Exception:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y)
    
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in splits:
        rf = RandomForestClassifier(n_estimators=600, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=42)
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(y_pred)
    
    # Metrics
    acc = accuracy_score(all_y_true, all_y_pred)
    macro_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    
    # Bootstrap CI
    print("Computing bootstrap confidence intervals...")
    bootstrap_results = bootstrap_ci(all_y_true, all_y_pred)
    
    # Per-class metrics
    cls_report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix plot
    cm_path = os.path.join(OUT_DIR, f"confusion_matrix_{setup_name.lower()}.png")
    plot_confusion_matrix(all_y_true, all_y_pred, classes, setup_name, cm_path)
    
    # Save results
    results = {
        'setup': setup_name,
        'n_features': len(feature_cols),
        'features_used': feature_cols,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'bootstrap_ci': bootstrap_results,
        'per_class_metrics': cls_report,
        'classes': classes
    }
    
    with open(os.path.join(OUT_DIR, f"robustness_{setup_name.lower()}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"Accuracy: {acc:.4f} (95% CI: {bootstrap_results['accuracy']['ci'][0]:.4f}-{bootstrap_results['accuracy']['ci'][1]:.4f})")
    print(f"Macro-F1: {macro_f1:.4f} (95% CI: {bootstrap_results['macro_f1']['ci'][0]:.4f}-{bootstrap_results['macro_f1']['ci'][1]:.4f})")
    print(f"Confusion matrix saved to: {cm_path}")
    
    return results

def main():
    print("=== Robustness Analysis ===")
    print(f"Dataset: {len(df)} samples, {len(classes)} classes")
    print(f"Classes: {classes}")
    
    # Analyze both setups
    pattern_results = analyze_setup("Pattern_Only", pattern_cols)
    hybrid_results = analyze_setup("Hybrid", hybrid_cols)
    
    # Comparison summary
    comparison = {
        'pattern_only': {
            'accuracy': pattern_results['accuracy'],
            'macro_f1': pattern_results['macro_f1'],
            'accuracy_ci': pattern_results['bootstrap_ci']['accuracy']['ci'],
            'macro_f1_ci': pattern_results['bootstrap_ci']['macro_f1']['ci']
        },
        'hybrid': {
            'accuracy': hybrid_results['accuracy'],
            'macro_f1': hybrid_results['macro_f1'],
            'accuracy_ci': hybrid_results['bootstrap_ci']['accuracy']['ci'],
            'macro_f1_ci': hybrid_results['bootstrap_ci']['macro_f1']['ci']
        }
    }
    
    with open(os.path.join(OUT_DIR, "robustness_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Pattern-Only: {pattern_results['accuracy']:.4f} ± {np.diff(pattern_results['bootstrap_ci']['accuracy']['ci'])[0]/2:.4f}")
    print(f"Hybrid:       {hybrid_results['accuracy']:.4f} ± {np.diff(hybrid_results['bootstrap_ci']['accuracy']['ci'])[0]/2:.4f}")
    print(f"All results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
