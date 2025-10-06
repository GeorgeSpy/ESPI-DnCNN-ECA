import argparse, os, json, pandas as pd
from pathlib import Path

def read_auto(path):
    # auto-detect delimiter (comma/semicolon/tab) safely
    return pd.read_csv(path, sep=None, engine="python")

def pick_name_column(df):
    for cand in ["name","filename","file","id"]:
        if cand in df.columns: return cand
    # αλλιώς, υποθέτουμε η 1η είναι το όνομα
    return df.columns[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="output report dir")
    args=ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # --- load
    F = read_auto(args.feats)
    L = read_auto(args.labels)

    name_col_f = pick_name_column(F)
    name_col_l = pick_name_column(L)

    # --- coerce numerics from features (χωρίς την name/filename)
    non_feature_cols = {name_col_f}
    num = F.drop(columns=list(non_feature_cols), errors="ignore").apply(pd.to_numeric, errors="coerce")
    # πέτα εντελώς κενές/NaN στήλες
    num = num.dropna(axis=1, how="all")

    # --- sets & intersection
    feats_names = set(F[name_col_f].astype(str))
    label_map   = dict(zip(L[name_col_l].astype(str), L[L.columns[-1]].astype(int)))  # τελευταία στήλη = label
    labels_names= set(label_map.keys())
    inter = sorted(feats_names & labels_names)

    # --- reports
    info = {
        "features_path": args.feats,
        "labels_path": args.labels,
        "feats_shape_raw": [int(F.shape[0]), int(F.shape[1])],
        "feats_name_col": name_col_f,
        "feats_numeric_cols": [c for c in num.columns],
        "feats_numeric_count": int(num.shape[1]),
        "labels_shape_raw": [int(L.shape[0]), int(L.shape[1])],
        "labels_name_col": name_col_l,
        "names_in_feats": len(feats_names),
        "names_in_labels": len(labels_names),
        "names_intersection": len(inter),
    }
    (outdir / "debug_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # γράψε unmatched λίστες
    (outdir / "unmatched_in_feats.txt").write_text(
        "\n".join(sorted(feats_names - labels_names)), encoding="utf-8")
    (outdir / "unmatched_in_labels.txt").write_text(
        "\n".join(sorted(labels_names - feats_names)), encoding="utf-8")

    # γράψε ένα "καθαρό" features CSV μόνο με name + numeric cols
    clean = pd.concat([F[[name_col_f]], num], axis=1)
    clean_out = outdir / "features_numeric_only.csv"
    clean.to_csv(clean_out, index=False)
    print("[OK] Wrote report to", outdir)
    print("[OK] Numeric-only features ->", clean_out)
    print("[INFO]", info)

if __name__ == "__main__":
    main()
