# -*- coding: utf-8 -*-
r"""
ESPI pipeline runner (features → dedup → train → predict)

Usage:
  python run_espi_pipeline.py --stages features,dedup,train,predict      # dry-run
  python run_espi_pipeline.py --stages features,train --run               # execute

Notes:
- Interpreter: uses sys.executable by default (or %ESPI_PY% if set)
- Paths: adjust CONFIG below once. Keep ASCII paths or raw strings.
"""

import os, sys, shlex, subprocess
from pathlib import Path
import argparse

# --- Python interpreter selection (future-proof) ---
VENV_PY = Path(os.environ.get("ESPI_PY", sys.executable))
if not VENV_PY.exists():
    raise RuntimeError(f"Python interpreter not found: {VENV_PY}")
print(f"[INFO] Using python: {VENV_PY}")

# --- CONFIG: φτιάξ' τα μια φορά εδώ ---
CONFIG = {
    # ROI
    "roi_mask": r"C:\ESPI_TEMP\roi_mask.png",

    # PHASE root (τελικό preset που έχουμε κλειδώσει)
    "phase_root": r"C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_PhaseOut_DEN_b18_cs16_ff100",

    # FEATURES out
    "features_csv":       r"C:\ESPI_TEMP\features\smoke_features.csv",
    "features_dedup_csv": r"C:\ESPI_TEMP\features\smoke_features_dedup.csv",

    # LABELS (merged) & MODEL out
    "labels_csv": r"C:\ESPI_TEMP\features\cluster_assignments_merged.csv",
    "model_out":  r"C:\ESPI_TEMP\features\rf_model\modes_rf_pipeline.joblib",
    "meta_out":   r"C:\ESPI_TEMP\features\rf_model\meta.json",  # θα το γράψει το train

    # DEBUG & NUMERIC FEATURES
    "debug_dir":  r"C:\ESPI_TEMP\features\debug_report",
    "feats_numeric": r"C:\ESPI_TEMP\features\debug_report\features_numeric_only.csv",

    # PRED out
    "pred_csv":   r"C:\ESPI_TEMP\features\smoke_predictions.csv",

    # Feature extraction tune
    "edge_quantile": "0.85",
}

# --- helper: build command as list (χωρίς θέματα quoting) ---
def cmd(*parts: str) -> list[str]:
    return [str(p) for p in parts if p is not None and str(p) != ""]

def run_cmd(args: list[str], do_run: bool):
    printable = " ".join(shlex.quote(a) for a in args)
    print("> ", printable)
    if do_run:
        subprocess.run(args, check=True)

def stage_features(do_run: bool):
    Path(CONFIG["features_csv"]).parent.mkdir(parents=True, exist_ok=True)
    run_cmd(cmd(
        VENV_PY, r"C:\ESPI_DnCNN\espi_features_nodal.py",
        "--phase-root", CONFIG["phase_root"],
        "--roi-mask",   CONFIG["roi_mask"],
        "--out-csv",    CONFIG["features_csv"],
        "--edge-quantile", CONFIG["edge_quantile"]
    ), do_run)

def stage_dedup(do_run: bool):
    run_cmd(cmd(
        VENV_PY, r"C:\ESPI_DnCNN\dedup_features_by_name.py",
        "--in",  CONFIG["features_csv"],
        "--out", CONFIG["features_dedup_csv"]
    ), do_run)

def stage_debug(do_run: bool):
    Path(CONFIG["debug_dir"]).mkdir(parents=True, exist_ok=True)
    run_cmd(cmd(
        VENV_PY, r"C:\ESPI_DnCNN\debug_features_labels.py",
        "--feats",  CONFIG["features_dedup_csv"],
        "--labels", CONFIG["labels_csv"],
        "--out",    CONFIG["debug_dir"]
    ), do_run)

def stage_train(do_run: bool):
    Path(CONFIG["model_out"]).parent.mkdir(parents=True, exist_ok=True)
    run_cmd(cmd(
        VENV_PY, r"C:\ESPI_DnCNN\train_from_features_robust.py",
        "--feats",     CONFIG["feats_numeric"],
        "--labels",    CONFIG["labels_csv"],
        "--out-model", CONFIG["model_out"],
        "--out-meta",  CONFIG["meta_out"]
    ), do_run)

def stage_predict(do_run: bool):
    args = cmd(
        VENV_PY, r"C:\ESPI_DnCNN\modes_predict_min.py",
        "--features", CONFIG["feats_numeric"],
        "--model", CONFIG["model_out"],
        "--meta",  CONFIG["meta_out"],
        "--outcsv", CONFIG["pred_csv"]
    )
    run_cmd(args, do_run)

STAGE_FUN = {
    "features": stage_features,
    "dedup":    stage_dedup,
    "debug":    stage_debug,
    "train":    stage_train,
    "predict":  stage_predict,
    # Προαιρετικά μπορείς να προσθέσεις "phase": stage_phase αν το χρειαστείς
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="features,dedup,debug,train,predict",
                    help="comma separated: features,dedup,debug,train,predict or 'all'")
    ap.add_argument("--run", action="store_true", help="execute commands (otherwise dry-run)")
    args = ap.parse_args()

    stages = [s.strip().lower() for s in args.stages.split(",")]
    if stages == ["all"]:
        stages = list(STAGE_FUN.keys())

    print("[PLAN]", " → ".join(stages), "|", "RUN" if args.run else "DRY-RUN")

    for s in stages:
        fn = STAGE_FUN.get(s)
        if not fn:
            print(f"[WARN] unknown stage: {s} (skip)")
            continue
        print(f"[STAGE] {s}")
        fn(args.run)

if __name__ == "__main__":
    main()