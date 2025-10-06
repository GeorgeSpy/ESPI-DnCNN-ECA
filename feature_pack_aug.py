# Usage:
#   C:\ESPI_VENV2\Scripts\python.exe feature_pack_aug.py ^
#     --features C:\ESPI_TEMP\features\all_features_merged.csv ^
#     --out C:\ESPI_TEMP\features\all_features_aug.csv ^
#     --root C:\ESPI_TEMP\GPU_FULL2 --try-glcm
import argparse, os, json, glob
import numpy as np
import pandas as pd

def first_npy(dirpath):
    cand = glob.glob(os.path.join(dirpath, "phase_unwrapped_npy", "*.npy"))
    return cand[0] if cand else None

def load_phase_for_row(row, root):
    # Try to infer freq directory
    # Prefer explicit columns if exist, else reconstruct from dataset/freq/db
    for col in ["freq_dir","dir","freq_path"]:
        if col in row and isinstance(row[col], str) and os.path.isdir(row[col]):
            npy = first_npy(row[col]); 
            if npy: return np.load(npy)
    if all(c in row for c in ["dataset","freq_hz","level_db"]):
        ds = str(row["dataset"])
        hz = int(round(float(row["freq_hz"])))
        db = float(row["level_db"])
        # pattern: 0155Hz_90.0db
        freq_dir = f"{hz:04d}Hz_{db:.1f}db"
        cand_dir = os.path.join(root, f"{ds}_PhaseOut_b18_cs16_ff100", freq_dir)
        if os.path.isdir(cand_dir):
            npy = first_npy(cand_dir)
            if npy: return np.load(npy)
    return None

def glcm_feats(img_u16, levels=32, max_samples=200000):
    # Very light GLCM-like: quantize, compute normalized co-occurrence on (right,down) shifts
    # Works without skimage
    h, w = img_u16.shape
    if h*w > max_samples:
        # center crop
        s = int(np.sqrt(max_samples))
        y0 = (h - s)//2; x0 = (w - s)//2
        img_u16 = img_u16[y0:y0+s, x0:x0+s]
    x = img_u16.astype(np.float32)
    x = x - np.nanmin(x); denom = np.nanmax(x) - np.nanmin(x)
    if denom <= 0: return {"glcm_contrast":0.0,"glcm_hom":1.0,"glcm_entropy":0.0}
    x = (x/denom)
    q = np.clip((x*(levels-1)).astype(np.int32), 0, levels-1)
    def cooc(dx, dy):
        A = q[max(0,dy):q.shape[0]+min(0,dy), max(0,dx):q.shape[1]+min(0,dx)]
        B = q[max(0,-dy):q.shape[0]-max(0,dy), max(0,-dx):q.shape[1]-max(0,dx)]
        M = np.zeros((levels, levels), dtype=np.float64)
        for i in range(levels):
            mask = (A==i)
            if mask.any():
                vals, cnts = np.unique(B[mask], return_counts=True)
                M[i, vals] += cnts
        s = M.sum()
        return (M/s) if s>0 else M
    P = (cooc(1,0) + cooc(0,1)) / 2.0
    ii, jj = np.indices(P.shape)
    contrast = ((ii-jj)**2 * P).sum()
    hom = (P / (1.0 + (ii-jj)**2)).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        plogp = P * np.log2(np.where(P>0, P, 1))
    entropy = -plogp.sum()
    return {"glcm_contrast":float(contrast), "glcm_hom":float(hom), "glcm_entropy":float(entropy)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="Root with *_PhaseOut_b18_cs16_ff100")
    ap.add_argument("--try-glcm", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    # ensure columns exist
    for c in ["valid_px","zero_frac","chg_h","chg_v","chg_d1","chg_d2","grad_mean","grad_std","lap_mad","phase_std"]:
        if c not in df.columns: raise SystemExit(f"missing column: {c}")
    # freq_hz / level_db if missing try to parse from name or dir
    if "freq_hz" not in df.columns and "freq" in df.columns: df["freq_hz"] = df["freq"]
    if "level_db" not in df.columns: 
        if "db" in df.columns: df["level_db"] = df["db"]
        else: df["level_db"] = np.nan

    eps = 1e-6
    df["hv_ratio"] = (df["chg_h"]+eps)/(df["chg_v"]+eps)
    df["diag_ratio"] = (df["chg_d1"]+eps)/(df["chg_d2"]+eps)
    df["nodal_complexity"] = df["chg_h"]+df["chg_v"]+df["chg_d1"]+df["chg_d2"]
    df["grad_cv"] = (df["grad_std"]+eps)/(np.abs(df["grad_mean"])+eps)
    df["lapz"] = (df["lap_mad"] - df["lap_mad"].mean())/(df["lap_mad"].std()+eps)
    # simple bin-distance priors (centers matching your bins)
    centers = {
        "mode_(1,1)H": 180.0,
        "mode_(1,1)T": 335.0,
        "mode_(1,2)":  512.0,
        "mode_(2,1)":  555.0,
        "mode_higher": 860.0
    }
    for k,v in centers.items():
        df[f"dist_{k}"] = np.abs(df["freq_hz"] - v)

    if args.try_glcm and args.root is not None:
        gl_list = []
        for i, row in df.iterrows():
            arr = load_phase_for_row(row, args.root)
            if arr is None:
                gl_list.append({"glcm_contrast":np.nan,"glcm_hom":np.nan,"glcm_entropy":np.nan})
            else:
                # scale phase to positive range for GLCM
                a = arr.astype(np.float32)
                a = a - np.nanmin(a)
                gl_list.append(glcm_feats(a))
        gl = pd.DataFrame(gl_list)
        df = pd.concat([df.reset_index(drop=True), gl.reset_index(drop=True)], axis=1)

    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {df.shape[0]} rows and {df.shape[1]} cols")

if __name__ == "__main__":
    main()
