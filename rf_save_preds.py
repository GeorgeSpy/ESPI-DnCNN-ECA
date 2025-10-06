# C:\ESPI_DnCNN\rf_save_preds.py  (ASCII-only)
import os, re, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
IN_CSV  = r"C:\ESPI_TEMP\features\labels_fixed_bins_enhanced.csv"
OUT_DIR = r"C:\ESPI_TEMP\RF_EVAL"
os.makedirs(OUT_DIR, exist_ok=True)

ALWAYS_EXCLUDE = {"class_name","class_id","label","target","dataset","set","split","group_id",
                  "file","filepath","path","relpath","name","id"}
LEAKAGE = [r"^freq$", r"^freq_hz$", r"^level_db$", r"^dist_"]
def is_leak(c): return any(re.match(p,c) for p in LEAKAGE)

df = pd.read_csv(IN_CSV)
df = df.replace([np.inf,-np.inf], np.nan).fillna(df.median(numeric_only=True))
y = df["class_name"].astype(str).values

# groups: dataset + 5Hz bin
if "freq_hz" in df: f = df["freq_hz"].astype(float)
elif "freq" in df:  f = df["freq"].astype(float)
else: raise SystemExit("need freq_hz or freq")
dset = df["dataset"] if "dataset" in df else "UNK"
groups = dset.astype(str)+"_"+(np.floor(f/5).astype(int)).astype(str)

num_cols = [c for c in df.columns if c not in ALWAYS_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

pattern_cols = [c for c in num_cols if not is_leak(c)]
hybrid_cols  = list(num_cols)

def run_setup(cols, tag):
    X = df[cols].values
    try:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups=groups)
    except Exception:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y)
    rows = []
    for fi,(tr,te) in enumerate(splits):
        rf = RandomForestClassifier(n_estimators=600, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=42)
        rf.fit(X[tr], y[tr])
        yp = rf.predict(X[te])
        for i,j in enumerate(te):
            rows.append((int(j), str(y[j]), str(yp[i]), int(fi)))
    out = pd.DataFrame(rows, columns=["idx","y_true","y_pred","fold"])
    out["setup"]=tag
    out.to_csv(os.path.join(OUT_DIR, f"preds_{tag}.csv"), index=False)
    acc = accuracy_score(out["y_true"], out["y_pred"])
    with open(os.path.join(OUT_DIR, f"summary_{tag}.json"),"w",encoding="utf-8") as f:
        json.dump({"setup":tag,"n":len(df),"features":cols,"accuracy":acc}, f, indent=2)

run_setup(pattern_cols, "pattern")
run_setup(hybrid_cols,  "hybrid")
print("Wrote preds_pattern.csv / preds_hybrid.csv in", OUT_DIR)
