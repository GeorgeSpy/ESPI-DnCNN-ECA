# coding: utf-8
"""
Batch denoise from a fine-tuned checkpoint that was trained with a different
head (residual vs clean). AUTO mode δοκιμάζει και τα δύο και επιλέγει ό,τι
μοιάζει «φυσιολογικό» (χωρίς κορεσμό) αφού κάνει σωστό de-normalization.

Απαιτεί το 'denoise_debug_dualmode_COMPAT.py' (το ήδη έχεις) για να φορτώσει
σωστά το CompatNet που ταιριάζει στα weights του checkpoint.
"""

import argparse, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from importlib.machinery import SourceFileLoader
from tqdm import tqdm

# ---------- I/O helpers ----------
def imread_gray_float01(p: Path) -> np.ndarray:
    im = Image.open(p).convert("L")
    a = np.array(im, dtype=np.float32)
    return a / 255.0

def imsave_uint01(arr: np.ndarray, path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)

def percentiles(x: np.ndarray, lo: float, hi: float):
    lo = float(np.percentile(x, lo))
    hi = float(np.percentile(x, hi))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return lo, hi

# ---------- tiling (reflect-safe) ----------
def extract_patch(x, top, left, tile):
    _, _, H, W = x.shape
    h = min(tile, H - top)
    w = min(tile, W - left)
    patch = x[:, :, top:top + h, left:left + w]
    ph = tile - h
    pw = tile - w
    if ph > 0 or pw > 0:
        # reflect χρειάζεται dim>1: αν w==1 ή h==1 κάνε replicate
        mode = "reflect"
        if w == 1 or h == 1:
            mode = "replicate"
        pad = (0, pw, 0, ph)
        patch = F.pad(patch, pad, mode=mode)
    return patch, h, w

def stitch(out, top, left, patch, h, w):
    out[:, :, top:top + h, left:left + w] = patch[:, :, :h, :w]

def forward_tiled(net, x, tile, overlap):
    b, c, H, W = x.shape
    assert tile > overlap >= 0
    step = tile - overlap
    out = torch.zeros_like(x)
    for top in range(0, H, step):
        for left in range(0, W, step):
            patch, h, w = extract_patch(x, top, left, tile)
            with torch.no_grad():
                y = net(patch)
            stitch(out, top, left, y, h, w)
    return out

# ---------- AUTO decision ----------
def choose_mode(x0, y_clean, y_resid):
    """
    x0: input (0..1) np, y_clean/y_resid: candidates already de-normalized to 0..1
    Επιλογή βάσει: (α) ποσοστό κορεσμού, (β) std κοντά στο input.
    """
    def score(y):
        sat = float(((y <= 0.0) | (y >= 1.0)).mean())         # όσο μικρότερο τόσο το καλύτερο
        std = float(y.std() + 1e-6)
        std_in = float(x0.std() + 1e-6)
        std_ratio = abs(math.log((std + 1e-6) / (std_in + 1e-6)))  # όσο μικρότερο τόσο καλύτερο
        return sat + 0.5 * std_ratio

    s_clean = score(y_clean)
    s_resid = score(y_resid)
    return ("clean", y_clean) if s_clean <= s_resid else ("residual", y_resid)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compat", default=r"C:\ESPI_DnCNN\denoise_debug_dualmode_COMPAT.py")
    ap.add_argument("--mode", choices=["auto","clean","residual"], default="auto")
    ap.add_argument("--p_lo", type=float, default=0.5)
    ap.add_argument("--p_hi", type=float, default=99.5)
    ap.add_argument("--viz_percentile", type=float, default=99.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # load CompatNet dynamically from the compat file
    compat_mod = SourceFileLoader("compat", args.compat).load_module()
    # φορτώνει state-dict και στήνει το σωστό net για το checkpoint
    sd, net = compat_mod.build_compat_model(args.ckpt, device=args.device)  # διαθέσιμο στο COMPAT script
    net.eval()

    in_dir  = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in in_dir.glob("*.png")])
    if args.limit > 0:
        files = files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={args.device} mode={args.mode}")

    for p in tqdm(files):
        x0 = imread_gray_float01(p)                         # 0..1
        a, b = percentiles(x0, args.p_lo, args.p_hi)        # per-image input scaling
        x = (x0 - a) / (b - a)                              # normalized input to 0..1
        x = np.clip(x, 0.0, 1.0)

        xt = torch.from_numpy(x)[None, None, ...].to(args.device)

        # raw network output in normalized domain
        yt = forward_tiled(net, xt, args.tile, args.overlap)

        y_norm = yt.squeeze().detach().cpu().numpy().astype(np.float32)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # δύο υποψήφια outputs στο ORIGINAL scale
        y_clean  = np.clip(y_norm * (b - a) + a, 0.0, 1.0)
        y_resid  = np.clip(x0 - y_norm * (b - a), 0.0, 1.0)  # noisy - denorm(residual)

        if args.mode == "clean":
            y = y_clean; picked = "clean"
        elif args.mode == "residual":
            y = y_resid; picked = "residual"
        else:
            picked, y = choose_mode(x0, y_clean, y_resid)

        # ελαφρύ viz-stretch μόνο για αποθήκευση PNG (δεν αλλοιώνει τα .npy)
        v_hi = np.percentile(y, args.viz_percentile)
        if v_hi < 1e-6: v_hi = 1.0
        y_viz = np.clip(y / v_hi, 0.0, 1.0)

        out_path = out_dir / p.name
        imsave_uint01(y_viz, out_path)

        # log για έλεγχο
        sys.stdout.write(f"\r[{p.name}] mode={picked} in[min,max]=({x0.min():.3f},{x0.max():.3f}) "
                         f"out[min,max]=({y.min():.3f},{y.max():.3f})     ")
        sys.stdout.flush()

    print(f"\n[DONE] Saved to {str(out_dir)}")

if __name__ == "__main__":
    main()
