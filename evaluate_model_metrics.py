import argparse, json, pandas as pd, numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def read_auto(path):
    return pd.read_csv(path, sep=None, engine="python")

def pick_name_column(df):
    for cand in ["name","filename","file","id"]:
        if cand in df.columns: return cand
    return df.columns[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained model .joblib file")
    ap.add_argument("--meta", required=True, help="model metadata .json file")
    ap.add_argument("--feats", required=True, help="features CSV file")
    ap.add_argument("--labels", required=True, help="labels CSV file")
    ap.add_argument("--outdir", required=True, help="output directory for metrics")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model and metadata
    model = joblib.load(args.model)
    with open(args.meta, 'r') as f:
        meta = json.load(f)

    # Load features and labels
    F = read_auto(args.feats)
    L = read_auto(args.labels)

    name_f = pick_name_column(F)
    name_l = pick_name_column(L)

    # Prepare data (same logic as training)
    labels = dict(zip(L[name_l].astype(str), L.iloc[:, -1].astype(int)))
    Xnum = F.drop(columns=[name_f], errors="ignore").apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.dropna(axis=1, how="all")

    names = F[name_f].astype(str).tolist()
    y_true = []
    keep_idx = []
    for i, n in enumerate(names):
        if n in labels:
            y_true.append(int(labels[n]))
            keep_idx.append(i)

    X = Xnum.iloc[keep_idx].to_numpy(dtype=np.float32)
    y_true = np.array(y_true, dtype=int)

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(list(set(y_true)))

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(outdir / "confusion_matrix.csv")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=[f"Class_{c}" for c in classes])
    
    # Save per-class metrics
    with open(outdir / "per_class_metrics.txt", 'w') as f:
        f.write("=== PER-CLASS METRICS ===\n\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n\n")
        f.write("=== DETAILED REPORT ===\n\n")
        f.write(report)

    # Save summary metrics
    summary = {
        "model_path": args.model,
        "features_path": args.feats,
        "labels_path": args.labels,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": [int(c) for c in classes],
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "confusion_matrix": cm.astype(int).tolist()
    }

    with open(outdir / "metrics_summary.json", 'w') as f:
        json.dump(summary, indent=2, fp=f)

    print(f"[OK] Metrics saved to {outdir}")
    print(f"[INFO] Macro F1: {macro_f1:.4f}")
    print(f"[INFO] Weighted F1: {weighted_f1:.4f}")
    print(f"[INFO] Classes: {classes}")

if __name__ == "__main__":
    main()
