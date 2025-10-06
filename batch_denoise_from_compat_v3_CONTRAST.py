# -*- coding: utf-8 -*-
r"""
batch_denoise_from_compat_v3_CONTRAST.py
- Φορτώνει CompatNet από denoise_debug_dualmode_COMPAT.py
- Υποστηρίζει δύο modes:
    residual:  den = input - model(x)         (τυπικό για DnCNN residual)
    clean:     den = model(x)                  (αν το μοντέλο βγάζει κατευθείαν clean)
- Percentile norm πριν/μετά + auto-invert για frames που βγαίνουν "ανάποδα"
"""

import argparse, importlib.util
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFont, ImageDraw
import torch
import torch.nn.functional as F

# ---------- I/O helpers ----------
def load_module_from_path(py_path):
    spec = importlib.util.spec_from_file_location("compat_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def imread_float01(p):
    im = Image.open(p).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0

def imsave_uint01(arr, out_path):
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(out_path)

def to_tensor01(a, device):
    return torch.from_numpy(a[None, None, :, :]).to(device)

# ---------- norm / invert ----------
def apply_percentile_norm(a, p_low=0.0, p_high=99.8):
    if p_high is None:  # skip
        return a
    lo = np.percentile(a, p_low) if p_low and p_low > 0 else a.min()
    hi = np.percentile(a, p_high)
    if hi <= lo:  # guard
        return np.clip(a, 0.0, 1.0)
    b = (a - lo) / (hi - lo)
    return np.clip(b, 0.0, 1.0)

def auto_invert_if_needed(a, thresh=0.65):
    med = float(np.median(a))
    mean = float(np.mean(a))
    # Αν είναι "κατάλευκο" και οι λεπτομέρειες φαίνονται ως μαύρα specs -> γύρνα το.
    if med > thresh and mean > thresh:
        return 1.0 - a, True
    return a, False

# ---------- tiling ----------
def hann2d(h, w):
    wy = np.hanning(max(h, 2))
    wx = np.hanning(max(w, 2))
    w2 = np.outer(wy, wx).astype(np.float32)
    eps = 1e-6
    w2 = (w2 - w2.min()) / (w2.max() - w2.min() + eps)
    w2 = (w2 * (1 - 2*eps) + eps).astype(np.float32)
    return w2

def safe_pad(patch, ph, pw):
    # replicate για να αποφύγουμε crash σε μικρά patches
    return F.pad(patch, (0, pw, 0, ph), mode="replicate")

@torch.no_grad()
def denoise_tiled(model, x, tile=256, overlap=64, mode="residual"):
    """
    x: [1,1,H,W] in [0,1]
    mode: 'residual' -> den = x - model(x)   /  'clean' -> den = model(x)
    """
    _, _, H, W = x.shape
    step = tile - overlap
    if step <= 0:
        raise ValueError("tile must be > overlap")
    out = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    w_np = hann2d(tile, tile)
    w = torch.from_numpy(w_np).to(x.device)[None, None, :, :]

    for top in range(0, H, step):
        for left in range(0, W, step):
            h_sub = min(tile, H - top)
            w_sub = min(tile, W - left)
            patch = x[:, :, top:top+h_sub, left:left+w_sub]
            ph = tile - h_sub
            pw = tile - w_sub
            patch_pad = safe_pad(patch, ph, pw)

            y = model(patch_pad)
            y = y[:, :, :h_sub, :w_sub]
            if mode == "residual":
                den = torch.clamp(patch - y, 0.0, 1.0)
            else:
                den = torch.clamp(y, 0.0, 1.0)

            w_crop = w[:, :, :h_sub, :w_sub]
            out[:, :, top:top+h_sub, left:left+w_sub] += den * w_crop
            wsum[:, :, top:top+h_sub, left:left+w_sub] += w_crop

    out /= torch.clamp(wsum, min=1e-6)
    return out

# ---------- debug triptych ----------
def save_triptych(inp, den, out_path):
    """inp/den: float [0,1], φτιάχνει 3-στηλη εικόνα (input | denoised | hist-stretched)"""
    h, w = inp.shape
    den_st = apply_percentile_norm(den, 0.0, 99.8)
    canvas = Image.new("L", (w*3, h), 0)
    for i, img in enumerate([inp, den, den_st]):
        canvas.paste(Image.fromarray((img*255).astype(np.uint8)), (i*w, 0))
    canvas.save(out_path)

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
    ap.add_argument("--mode", choices=["residual","clean"], default="residual",
                    help="Το μοντέλο σου προβλέπει residual (συνήθως) ή καθαρή εικόνα;")
    ap.add_argument("--percentile-in", type=float, default=99.0,
                    help="percentile για είσοδο (πριν το μοντέλο). 0=off")
    ap.add_argument("--percentile-out", type=float, default=99.8,
                    help="percentile για έξοδο (μετά το blend). 0=off")
    ap.add_argument("--auto-invert", action="store_true",
                    help="Αν το frame βγει 'κατάλευκο', κάνε 1-x στην έξοδο.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-few", type=int, default=0,
                    help="σώσε triptychs για τα πρώτα Ν frames στο _triptych")
    args = ap.parse_args()

    device = torch.device(args.device)
    mod = load_module_from_path(args.compat)
    CompatNet = getattr(mod, "CompatNet")

    # φόρτωσε state-dict (οποιοδήποτε layout)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck
    if isinstance(ck, dict):
        for k in ("model","model_state","state_dict","net","weights"):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]; break

    # constructor με sd (όπως στο COMPAT), αλλιώς fallback
    try:
        model = CompatNet(sd)
        model = model.to(device).eval()
        print("[info] CompatNet(sd) ok.")
    except TypeError:
        model = CompatNet().to(device).eval()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[info] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    in_dir = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    tri_dir = out_dir.parent / (out_dir.name + "_triptych")
    if args.save_few > 0: tri_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={args.device} mode={args.mode}")

    torch.set_grad_enabled(False)
    for i, p in enumerate(files, 1):
        # --- read & (προαιρετικά) norm εισόδου ---
        inp = imread_float01(p)
        if args.percentile_in and args.percentile_in > 0:
            inp_n = apply_percentile_norm(inp, 0.0, args.percentile_in)
        else:
            inp_n = np.clip(inp, 0.0, 1.0)

        x = to_tensor01(inp_n, device)
        den = denoise_tiled(model, x, tile=args.tile, overlap=args.overlap, mode=args.mode)
        den = den.squeeze().cpu().numpy()

        # --- (προαιρετικά) auto-invert + norm εξόδου ---
        if args.auto_invert:
            den, flipped = auto_invert_if_needed(den, thresh=0.65)
        if args.percentile_out and args.percentile_out > 0:
            den = apply_percentile_norm(den, 0.0, args.percentile_out)

        imsave_uint01(den, out_dir / p.name)

        if args.save_few and i <= args.save_few:
            save_triptych(inp_n, den, tri_dir / f"{p.stem}_triptych.png")

        if i % 20 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {p.name}")

    print(f"[DONE] Saved to {out_dir}")

if __name__ == "__main__":
    main()
