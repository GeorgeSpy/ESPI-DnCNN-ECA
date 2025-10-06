# ASCII-only, Windows-friendly
# Runs two setups:
#   1) PATTERN-ONLY  (no freq / no dist / no level_db)
#   2) HYBRID        (pattern + prior: freq_hz, dist_*, level_db)
#
# Outputs JSON reports and feature lists under C:\ESPI_TEMP\RF_EVAL

import os
import re
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

IN_CSV  = r"C:\ESPI_TEMP\features\labels_fixed_bins_enhanced.csv"
OUT_DIR = r"C:\ESPI_TEMP\RF_EVAL"

RANDOM_STATE = 42
N_SPLITS = 3
N_ESTIMATORS = 600
N_JOBS = -1

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def load_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    if "class_name" not in df.columns:
        raise ValueError("CSV must contain 'class_name' column")
    return df

def infer_freq_hz(df):
    # if freq_hz exists and numeric, use it
    if "freq_hz" in df.columns and pd.api.types.is_numeric_dtype(df["freq_hz"]):
        return df["freq_hz"].astype(float)
    # fallback to 'freq' if numeric
    if "freq" in df.columns and pd.api.types.is_numeric_dtype(df["freq"]):
        return df["freq"].astype(float)
    # try parse from name/path-like columns
    candidates = [c for c in df.columns if "path" in c.lower() or "file" in c.lower() or "name" in c.lower()]
    for col in candidates:
        vals = df[col].astype(str).fillna("")
        m = vals.str.extract(r"(\d{3,4})Hz", expand=False)
        if m.notna().any():
            parsed = pd.to_numeric(m, errors="coerce")
            if parsed.notna().any():
                return parsed.fillna(parsed.median()).astype(float)
    # last resort: all NaN (will break groups; better raise)
    raise ValueError("Could not infer freq_hz. Please ensure 'freq_hz' numeric column exists.")

def get_dataset_col(df):
    # prefer 'dataset' if present; else try to infer from path
    if "dataset" in df.columns:
        return df["dataset"].astype(str)
    candidates = [c for c in df.columns if "path" in c.lower() or "file" in c.lower() or "name" in c.lower()]
    for col in candidates:
        s = df[col].astype(str).fillna("")
        # heuristic: find W01/W02/W03 token
        token = s.str.extract(r"(W0[1-9])", expand=False)
        if token.notna().any():
            return token.fillna("UNK").astype(str)
    return pd.Series(["UNK"]*len(df), index=df.index, dtype="string")

def build_groups(dataset, freq_hz):
    # group per (dataset, freq_bin) where freq_bin = floor(freq/5)
    freq_bin = (freq_hz.fillna(freq_hz.median())/5.0).apply(np.floor).astype(int)
    return dataset.astype(str) + "_" + freq_bin.astype(str)

# Columns to exclude from features in all cases
ALWAYS_EXCLUDE = {
    "class_name", "class_id", "label", "target",
    "dataset", "set", "split", "group_id",
    "file", "filepath", "path", "relpath", "name", "id",
}

# Regex to detect leakage / prior columns (excluded in PATTERN-ONLY; allowed in HYBRID)
LEAKAGE_PATTERNS = [
    r"^freq$", r"^freq_hz$", r"^level_db$",
    r"^dist_",          # any engineered distance-to-bin center
]

def is_leak_col(col):
    for pat in LEAKAGE_PATTERNS:
        if re.match(pat, col):
            return True
    return False

def numeric_feature_candidates(df):
    # numeric columns only
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # drop always-excluded meta
    num_cols = [c for c in num_cols if c not in ALWAYS_EXCLUDE]
    return num_cols

def split_feature_sets(df):
    num_cols = numeric_feature_candidates(df)
    # PATTERN-ONLY: remove leakage/prior cols
    pattern_cols = [c for c in num_cols if not is_leak_col(c)]
    # HYBRID: keep everything (pattern + prior cols)
    hybrid_cols  = list(num_cols)
    return pattern_cols, hybrid_cols

def rf_eval(df, feature_cols, setup_name, out_json):
    # y
    y = df["class_name"].astype(str).values
    # X
    X = df[feature_cols].values

    # groups
    try:
        freq_hz = infer_freq_hz(df)
    except Exception:
        # If cannot infer, fall back to no-groups stratified CV (not ideal)
        freq_hz = pd.Series([np.nan]*len(df))
    dataset = get_dataset_col(df)
    groups = build_groups(dataset, freq_hz)

    # CV
    cv = None
    try:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        y_pred = cross_val_predict(
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=None,
                n_jobs=N_JOBS,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
            ),
            X, y, cv=cv.split(X, y, groups=groups), n_jobs=N_JOBS, method="predict"
        )
    except ValueError:
        # fallback if grouping fails due to tiny classes
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        y_pred = cross_val_predict(
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=None,
                n_jobs=N_JOBS,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
            ),
            X, y, cv=cv, n_jobs=N_JOBS, method="predict"
        )

    # Metrics
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    cls_report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_norm = confusion_matrix(y, y_pred, labels=labels, normalize="true")

    # Fit one model on all data for feature importances (interpretation only)
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        n_jobs=N_JOBS,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)
    importances = dict(sorted(zip(feature_cols, rf.feature_importances_), key=lambda kv: kv[1], reverse=True))

    report = {
        "setup": setup_name,
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "features_used": feature_cols,
        "classes": list(map(str, labels)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": cls_report,
        "confusion_matrix": {
            "labels": list(map(str, labels)),
            "raw": cm.tolist(),
            "normalized_true": cm_norm.tolist(),
        },
        "feature_importances": importances,
        "cv": {
            "n_splits": N_SPLITS,
            "grouped": isinstance(cv, StratifiedGroupKFold)
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report

def main():
    ensure_outdir(OUT_DIR)
    df = load_table(IN_CSV).copy()

    # Keep only rows with non-na numeric in at least some features
    # (avoid completely empty rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    # basic impute for numeric cols to keep CV stable
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())

    # Build the two setups
    pattern_cols, hybrid_cols = split_feature_sets(df)

    # Save feature lists (reproducibility)
    with open(os.path.join(OUT_DIR, "feature_list_pattern.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pattern_cols))
    with open(os.path.join(OUT_DIR, "feature_list_hybrid.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hybrid_cols))

    # Evaluate
    rep_pattern = rf_eval(df, pattern_cols, "PATTERN_ONLY", os.path.join(OUT_DIR, "report_pattern.json"))
    rep_hybrid  = rf_eval(df, hybrid_cols,  "HYBRID",       os.path.join(OUT_DIR, "report_hybrid.json"))

    # Comparison
    comp = {
        "baseline_note": "Use this to compare with your frozen baseline (e.g. 0.553 accuracy).",
        "pattern_only": {
            "accuracy": rep_pattern["accuracy"],
            "macro_f1": rep_pattern["macro_f1"],
            "weighted_f1": rep_pattern["weighted_f1"],
            "n_features": rep_pattern["n_features"]
        },
        "hybrid": {
            "accuracy": rep_hybrid["accuracy"],
            "macro_f1": rep_hybrid["macro_f1"],
            "weighted_f1": rep_hybrid["weighted_f1"],
            "n_features": rep_hybrid["n_features"]
        },
        "delta_hybrid_minus_pattern": {
            "accuracy": rep_hybrid["accuracy"] - rep_pattern["accuracy"],
            "macro_f1": rep_hybrid["macro_f1"] - rep_pattern["macro_f1"],
            "weighted_f1": rep_hybrid["weighted_f1"] - rep_pattern["weighted_f1"],
        }
    }
    with open(os.path.join(OUT_DIR, "compare_baseline_pattern_hybrid.json"), "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)

    # Console summary
    def pct(x): return f"{100.0*x:.2f}%"
    print("=== SUMMARY ===")
    print(f"N = {len(df)}")
    print(f"PATTERN-ONLY: acc={pct(rep_pattern['accuracy'])}, macroF1={pct(rep_pattern['macro_f1'])}, weightedF1={pct(rep_pattern['weighted_f1'])}, feats={rep_pattern['n_features']}")
    print(f"HYBRID:       acc={pct(rep_hybrid['accuracy'])}, macroF1={pct(rep_hybrid['macro_f1'])}, weightedF1={pct(rep_hybrid['weighted_f1'])}, feats={rep_hybrid['n_features']}")
    print("Reports written to:", OUT_DIR)

if __name__ == "__main__":
    main()
