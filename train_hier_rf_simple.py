# Simplified hierarchical RF training
import argparse, os, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

INBINS = ["mode_(1,1)H","mode_(1,1)T","mode_(1,2)","mode_(2,1)","mode_higher"]

def stage_train_predict(X, y, groups, class_names, outdir, stage_tag):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=42, oob_score=True
    )
    pipe = Pipeline([("scale", StandardScaler()), ("rf", rf)])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
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
    args = ap.parse_args()

    # Load features (already has labels merged)
    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} samples")
    
    # Check if we have class_name column
    if "class_name" not in df.columns:
        # Load labels separately and merge
        labels = pd.read_csv(args.labels)
        df = pd.merge(df, labels[["dataset","freq_hz","class_name"]], on=["dataset","freq_hz"], how="inner")
        df = df.drop_duplicates(subset=["dataset","freq_hz"], keep='first')
        print(f"After merge: {len(df)} samples")
    
    # Feature columns (exclude metadata)
    exclude_cols = ["id", "dataset", "source_file", "class_id", "class_name", "material", "freq_hz.1"]
    feat_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_x") and not c.endswith("_y")]
    
    Xmat = df[feat_cols].values
    y_names = df["class_name"].values
    groups = df[args.groups].values if args.groups in df.columns else None
    
    print(f"Features: {len(feat_cols)} columns")
    print(f"Classes: {np.unique(y_names)}")
    print(f"Class distribution: {pd.Series(y_names).value_counts().to_dict()}")

    # === Stage A: other_unknown vs in_bins
    yA_names = np.where(y_names=="other_unknown","other_unknown","in_bins")
    classesA = ["in_bins","other_unknown"]
    yA = np.array([classesA.index(nm) for nm in yA_names])
    probsA, yA_pred, repA, cmA = stage_train_predict(Xmat, yA, groups, classesA, args.outdir, "A")

    # === Stage B: only in_bins
    maskB = (yA_names=="in_bins")
    Xb = Xmat[maskB]; yb_names = y_names[maskB]
    classesB = INBINS
    yb = np.array([classesB.index(nm) for nm in yb_names])
    groupsB = groups[maskB] if groups is not None else None
    probsB, yB_pred, repB, cmB = stage_train_predict(Xb, yb, groupsB, classesB, args.outdir, "B")

    # === Final predictions
    y_pred_names = np.full(len(df), "other_unknown")
    y_pred_names[maskB] = np.array(INBINS)[probsB.argmax(1)]
    
    rep_all = classification_report(y_names, y_pred_names, labels=INBINS+["other_unknown"], digits=3, output_dict=True)
    cm_all = confusion_matrix(y_names, y_pred_names, labels=INBINS+["other_unknown"])
    
    with open(os.path.join(args.outdir, "report_FINAL.json"), "w") as f:
        json.dump({
            "features_used": feat_cols,
            "classes": INBINS+["other_unknown"],
            "report": rep_all,
            "confusion": cm_all.tolist()
        }, f, indent=2)
    
    print("\n=== FINAL RESULTS ===")
    print(json.dumps({"macro_f1":rep_all["macro avg"]["f1-score"], "acc":rep_all["accuracy"]}, indent=2))

if __name__ == "__main__":
    main()
