# Usage:
#   C:\ESPI_VENV2\Scripts\python.exe compare_runs.py ^
#     --baseline C:\ESPI_TEMP\FREEZE_YYYYMMDD-HHMM\RF_baseline.json ^
#     --improved C:\ESPI_TEMP\RF_FINAL_IMPROVED\rf_results_complete.json
import argparse, json, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--improved", required=True)
args = ap.parse_args()

with open(args.baseline,"r") as f: base = json.load(f)
with open(args.improved,"r") as f: imp = json.load(f)

def mm_baseline(rep): 
    return {
        "acc": rep.get("accuracy", np.nan),
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"]
    }

def mm_improved(rep):
    return {
        "acc": rep.get("accuracy", np.nan),
        "macro_f1": rep.get("macro_f1", np.nan),
        "weighted_f1": rep.get("weighted_f1", np.nan)
    }

b = mm_baseline(base["report"])
i = mm_improved(imp)
delta = {k: (i[k]-b[k]) for k in b.keys()}
print("# COMPARISON\n")
print("Baseline :", b)
print("Improved :", i)
print("Delta    :", delta)
