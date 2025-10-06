#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_denoise_batch.py
---------------------
Evaluate denoising results against "clean" (Averaged) references.
Metrics per image (and summary): PSNR, SSIM, EdgeF1 (tolerant, 1px dilation).
Optionally also evaluate "noisy" baseline if provided.

Usage examples:
python eval_denoise_batch.py ^
  --clean-root "C:\...\W01_ESPI_90db-Averaged" ^
  --denoised-root "C:\...\W01_ESPI_90db-Denoised"

python eval_denoise_batch.py ^
  --clean-root "C:\...\W01_ESPI_90db-Averaged" ^
  --noisy-root "C:\...\W01_ESPI_90db-PseudoNoisy" ^
  --denoised-root "C:\...\W01_ESPI_90db-Denoised"
"""

import argparse, math, numpy as np, csv
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

VALID_EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}

def imread_uint01(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ("I;16","I;16B","I","L"):
        im = ImageOps.grayscale(im)
    if im.mode in ("I;16","I;16B","I"):
        arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def gather_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)

def psnr(x, y, eps=1e-12):
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0: return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse + eps))

def ssim_metric(x, y):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(x, y, data_range=1.0))
    except Exception:
        mu_x = x.mean(); mu_y = y.mean()
        sig_x = x.var(); sig_y = y.var()
        sig_xy = ((x - mu_x) * (y - mu_y)).mean()
        C1 = 0.01**2; C2 = 0.03**2
        return float(((2*mu_x*mu_y + C1) * (2*sig_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2) + 1e-12))

def canny_edges(img01: np.ndarray):
    try:
        from skimage.feature import canny
        return canny(img01, sigma=1.0).astype(np.uint8)
    except Exception:
        # Sobel magnitude + percentile threshold fallback
        from scipy.ndimage import sobel as _sobel  # may fail if scipy missing
        try:
            gx = _sobel(img01, axis=1); gy = _sobel(img01, axis=0)
            mag = np.sqrt(gx*gx + gy*gy)
            thr = np.percentile(mag, 85.0)
            return (mag >= thr).astype(np.uint8)
        except Exception:
            # last resort: simple gradient
            gy, gx = np.gradient(img01)
            mag = np.sqrt(gx*gx + gy*gy)
            thr = np.percentile(mag, 85.0)
            return (mag >= thr).astype(np.uint8)

def dilate(b: np.ndarray, rad: int = 1) -> np.ndarray:
    # binary dilation via max of shifted neighbors (no scipy/cv2 needed)
    H, W = b.shape
    out = b.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy == 0 and dx == 0: continue
            shifted = np.zeros_like(b)
            y0 = max(0, dy); y1 = H + min(0, dy)
            x0 = max(0, dx); x1 = W + min(0, dx)
            shifted[y0:y1, x0:x1] = b[y0-dy:y1-dy, x0-dx:x1-dx]
            out = np.maximum(out, shifted)
    if rad > 1:
        for _ in range(rad-1):
            out = dilate(out, 1)
    return out

def edge_f1(ref_img: np.ndarray, test_img: np.ndarray, dilate_rad: int = 1) -> float:
    e_ref = canny_edges(ref_img)
    e_tst = canny_edges(test_img)
    ref_d = dilate(e_ref, dilate_rad)
    tst_d = dilate(e_tst, dilate_rad)
    tp = np.sum(e_tst * ref_d)
    fp = np.sum(e_tst * (1 - ref_d))
    fn = np.sum(e_ref * (1 - tst_d))
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    if prec + rec <= 0: return 0.0
    return float(2*prec*rec / (prec + rec))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", required=True)
    ap.add_argument("--denoised-root", required=True)
    ap.add_argument("--noisy-root", default="")
    ap.add_argument("--csv", default="denoise_eval.csv")
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    den_root   = Path(args.denoised_root)
    noisy_root = Path(args.noisy_root) if args.noisy_root else None

    clean_files = gather_files(clean_root)
    if not clean_files:
        print("No clean images found under", clean_root); return

    rows = [("rel_path",
             "psnr_noisy","ssim_noisy","edgeF1_noisy",
             "psnr_denoised","ssim_denoised","edgeF1_denoised")]
    acc = {"n":0,
           "psnr_n":0.0, "ssim_n":0.0, "edge_n":0.0,
           "psnr_d":0.0, "ssim_d":0.0, "edge_d":0.0}

    for cf in tqdm(clean_files, desc="Evaluating"):
        rel = cf.relative_to(clean_root)
        clean = imread_uint01(cf)

        # find matching in noisy/denoised (same relative path, any ext)
        def find_match(root):
            if root is None: return None
            base = (root / rel).with_suffix("")  # drop ext
            # try any valid ext
            for ext in VALID_EXTS:
                cand = Path(str(base) + ext)
                if cand.exists(): return cand
            # also try .png specifically
            cand = (root / rel).with_suffix(".png")
            return cand if cand.exists() else None

        noisy_f = find_match(noisy_root) if noisy_root else None
        den_f   = find_match(den_root)

        psn, ssn, en = (None, None, None)
        if noisy_f is not None and noisy_f.exists():
            noisy = imread_uint01(noisy_f)
            H = min(clean.shape[0], noisy.shape[0]); W = min(clean.shape[1], noisy.shape[1])
            clean_c = clean[:H,:W]; noisy_c = noisy[:H,:W]
            psn = psnr(noisy_c, clean_c); ssn = ssim_metric(noisy_c, clean_c); en = edge_f1(clean_c, noisy_c)

        if den_f is None or not den_f.exists():
            print("Missing denoised for", rel); continue
        den = imread_uint01(den_f)
        H = min(clean.shape[0], den.shape[0]); W = min(clean.shape[1], den.shape[1])
        clean_c = clean[:H,:W]; den_c = den[:H,:W]

        psd = psnr(den_c, clean_c); ssd = ssim_metric(den_c, clean_c); ed = edge_f1(clean_c, den_c)

        rows.append((str(rel),
                     "" if psn is None else f"{psn:.4f}",
                     "" if ssn is None else f"{ssn:.6f}",
                     "" if en  is None else f"{en:.6f}",
                     f"{psd:.4f}", f"{ssd:.6f}", f"{ed:.6f}"))

        acc["n"] += 1
        if psn is not None:
            acc["psnr_n"] += psn; acc["ssim_n"] += ssn; acc["edge_n"] += en
        acc["psnr_d"] += psd; acc["ssim_d"] += ssd; acc["edge_d"] += ed

    # write CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(rows)

    # print summary
    n = acc["n"]
    if n > 0:
        if noisy_root:
            print(f"Baseline NOISY  : PSNR {acc['psnr_n']/n:.2f} | SSIM {acc['ssim_n']/n:.3f} | EdgeF1 {acc['edge_n']/n:.3f}")
        print(f"DENOISED vs CLEAN: PSNR {acc['psnr_d']/n:.2f} | SSIM {acc['ssim_d']/n:.3f} | EdgeF1 {acc['edge_d']/n:.3f}")
    print("CSV saved to:", csv_path)

if __name__ == "__main__":
    main()
