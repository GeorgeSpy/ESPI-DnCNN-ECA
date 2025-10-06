#!/usr/bin/env python3
# modes_label_and_train_min.py
# Map clusters -> human labels and train a simple classifier (RandomForest).
# Saves model + scaler + a small report.

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def parse_mapping(s):
    # Example: "0=(0,1);1=(1,0);2=(1,1);3=other"
    out = {}
    for part in re.split(r"[;,\s]+", s.strip()):
        if not part: continue
        if "=" not in part: continue
        k, v = part.split("=", 1)
        k = k.strip(); v = v.strip()
        if k.isdigit():
            out[int(k)] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged_with_clusters.csv from report step")
    ap.add_argument("--labels", required=True, help='Mapping like: 0=(0,1);1=(1,0);2=(1,1);3=other')
    ap.add_argument("--min-valid", type=int, default=200, help="Min valid_px to keep a sample")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged)
    if "cluster" not in df.columns:
        raise SystemExit("merged CSV must include 'cluster' column")
    if "valid_px" in df.columns:
        df = df[df["valid_px"] >= int(args.min_valid)].copy()

    mapping = parse_mapping(args.labels)
    if not mapping:
        raise SystemExit("No valid mapping parsed from --labels")

    # Assign human labels
    df["label"] = df["cluster"].map(mapping).fillna("unknown")

    # Features: keep numeric, drop id-like
    Xdf = df.select_dtypes(include=[np.number]).copy()
    for col in ("freq_hz","valid_px","cluster"):
        if col in Xdf.columns:
            Xdf.drop(columns=[col], inplace=True)
    y = df["label"].values
    ids = df["id"].values if "id" in df.columns else np.arange(len(df))

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        Xdf.values, y, ids, test_size=0.25, random_state=42, stratify=y
    )

    # Scale + RF
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rep = classification_report(y_test, y_pred, digits=3)
    cm  = confusion_matrix(y_test, y_pred)

    # Save artifacts
    model_path = outdir / "modes_rf_pipeline.joblib"
    joblib.dump(pipe, model_path)

    # Save mapping + used feature names
    meta = {
        "cluster_to_label": mapping,
        "feature_names": list(Xdf.columns)
    }
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save report
    with open(outdir / "report.txt", "w", encoding="utf-8") as f:
        f.write(rep + "\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))

    # Save predictions file
    pred_df = pd.DataFrame({
        "id": ids_test,
        "true_label": y_test,
        "pred_label": y_pred
    })
    pred_df.to_csv(outdir / "predictions_test.csv", index=False, encoding="utf-8")

    print("Saved:")
    print(" -", model_path)
    print(" -", outdir / "meta.json")
    print(" -", outdir / "report.txt")
    print(" -", outdir / "predictions_test.csv")

if __name__ == "__main__":
    main()
