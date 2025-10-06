# Usage:
#   C:\ESPI_VENV2\Scripts\python.exe train_hier_rf.py ^
#     --features C:\ESPI_TEMP\features\all_features_aug.csv ^
#     --labels   C:\ESPI_TEMP\features\labels_fixed_bins.csv ^
#     --outdir   C:\ESPI_TEMP\RF_FINAL ^
#     --groups dataset --alpha 0.7
import argparse, os, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

INBINS = ["mode_(1,1)H","mode_(1,1)T","mode_(1,2)","mode_(2,1)","mode_higher"]

def make_prior(df_labels, classes):
    # Gaussian prior around bin centers; fallback flat for other_unknown
    centers = {
        "mode_(1,1)H": 180.0,
        "mode_(1,1)T": 335.0,
        "mode_(1,2)":  512.0,
        "mode_(2,1)":  555.0,
        "mode_higher": 860.0,
    }
    sigmas = {k: (10.0 if v<400 else 25.0) for k,v in centers.items()}
    Xfreq = df_labels["freq_hz"].values.astype(float)
    prior = np.zeros((len(Xfreq), len(classes)), dtype=float)
    for j, cname in enumerate(classes):
        if cname in centers:
            mu = centers[cname]; s = sigmas[cname]
            prior[:, j] = np.exp(-0.5*((Xfreq-mu)/s)**2)
        else:
            prior[:, j] = 0.05
    # normalize
    prior = prior / np.clip(prior.sum(1, keepdims=True), 1e-9, None)
    return prior

def stage_train_predict(X, y, groups, class_names, alpha, outdir, stage_tag):
    rf = RandomForestClassifier(
        n_estimators=1200, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=42, oob_score=True
    )
    pipe = Pipeline([("scale", StandardScaler(with_mean=False)), ("rf", rf)])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1)
    y_pred = probs.argmax(1)
    rep = classification_report(y, y_pred, target_names=class_names, digits=3, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"report_{stage_tag}.json"), "w") as f:
        json.dump({"report":rep, "confusion":cm.tolist()}, f, indent=2)
    return probs, y_pred, rep, cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--groups", default="dataset")
    ap.add_argument("--alpha", type=float, default=0.7, help="blend weight for model vs prior")
    args = ap.parse_args()

    X = pd.read_csv(args.features)
    L = pd.read_csv(args.labels)
    # merge on keys; common columns: dataset,freq_hz,level_db or filename-based key
    keys = [k for k in ["dataset","freq_hz","level_db","dir","freq_dir"] if k in X.columns and k in L.columns]
    if not keys: 
        # Try alternative merge keys
        keys = [k for k in ["dataset","freq_hz"] if k in X.columns and k in L.columns]
        if not keys: raise SystemExit("No common merge keys between features and labels")
    df = pd.merge(X, L, on=keys, how="inner")
    
    # Remove duplicates based on merge keys
    df = df.drop_duplicates(subset=keys, keep='first')
    
    # Only drop rows with NaN in essential columns, not optional ones like GLCM
    essential_cols = ["class_name", "freq_hz"] + [c for c in ["valid_px","zero_frac","chg_h","chg_v","chg_d1","chg_d2","grad_mean","grad_std","lap_mad","phase_std"] if c in df.columns]
    df = df.dropna(subset=essential_cols)
    
    # Handle duplicate columns from merge (keep _x versions from features)
    for col in ["valid_px","zero_frac","chg_h","chg_v","chg_d1","chg_d2","grad_mean","grad_std","lap_mad","phase_std"]:
        if f"{col}_x" in df.columns:
            df[col] = df[f"{col}_x"]
        elif f"{col}_y" in df.columns:
            df[col] = df[f"{col}_y"]
    
    # FEATURES (keep original + pack)
    base_cols = ["valid_px","zero_frac","chg_h","chg_v","chg_d1","chg_d2","grad_mean","grad_std","lap_mad","phase_std"]
    extra_cols = [c for c in ["hv_ratio","diag_ratio","nodal_complexity","grad_cv","lapz","glcm_contrast","glcm_hom","glcm_entropy",
                              "dist_mode_(1,1)H","dist_mode_(1,1)T","dist_mode_(1,2)","dist_mode_(2,1)","dist_mode_higher"] if c in df.columns]
    freq_cols = [c for c in ["freq_hz","level_db"] if c in df.columns]
    feat_cols = base_cols + extra_cols + freq_cols
    Xmat = df[feat_cols].values
    y_names = df["class_name"].values
    groups = df[args.groups].values if args.groups in df.columns else None

    # === Stage A: other_unknown vs in_bins
    yA_names = np.where(y_names=="other_unknown","other_unknown","in_bins")
    classesA = ["in_bins","other_unknown"]
    yA = np.array([classesA.index(nm) for nm in yA_names])
    probsA, yA_pred, repA, cmA = stage_train_predict(Xmat, yA, groups, classesA, args.alpha, args.outdir, "A")

    # === Stage B: only in_bins
    maskB = (yA_names=="in_bins")
    Xb = Xmat[maskB]; yb_names = y_names[maskB]
    classesB = INBINS
    yb = np.array([classesB.index(nm) for nm in yb_names])
    groupsB = groups[maskB] if groups is not None else None
    probsB, yB_pred, repB, cmB = stage_train_predict(Xb, yb, groupsB, classesB, args.alpha, args.outdir, "B")

    # === Prior blending over full set (only for in_bins)
    prior_full = make_prior(df, INBINS)
    # build full probs over 6 classes
    P_full = np.zeros((len(df), 6), dtype=float)
    # map order: [ (1,1)H,(1,1)T,(1,2),(2,1),higher, other_unknown ]
    # indices in P_full
    idx_map = {nm:i for i,nm in enumerate(INBINS + ["other_unknown"])}
    # fill other_unknown from Stage A
    P_full[:, idx_map["other_unknown"]] = probsA[:, classesA.index("other_unknown")]
    # fill in_bins probs as product blend
    Pin = np.clip(probsB, 1e-9, 1.0)
    prior = np.clip(prior_full[maskB], 1e-9, 1.0)
    blend = np.zeros_like(Pin)
    # geometric blend in log space
    logb = args.alpha*np.log(Pin) + (1.0-args.alpha)*np.log(prior)
    blend = np.exp(logb)
    blend = blend / blend.sum(1, keepdims=True)
    # place back
    P_full[maskB, :len(INBINS)] = blend

    y_pred_names = np.array(INBINS + ["other_unknown"])[P_full.argmax(1)]
    rep_all = classification_report(y_names, y_pred_names, labels=INBINS+["other_unknown"], digits=3, output_dict=True)
    cm_all = confusion_matrix(y_names, y_pred_names, labels=INBINS+["other_unknown"])
    with open(os.path.join(args.outdir, "report_FINAL.json"), "w") as f:
        json.dump({
            "features_used": feat_cols,
            "classes": INBINS+["other_unknown"],
            "report": rep_all,
            "confusion": cm_all.tolist()
        }, f, indent=2)
    print(json.dumps({"macro_f1":rep_all["macro avg"]["f1-score"], "acc":rep_all["accuracy"]}, indent=2))

if __name__ == "__main__":
    main()
