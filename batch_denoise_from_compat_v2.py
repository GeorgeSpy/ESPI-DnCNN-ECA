# batch_denoise_from_compat_v2.py
# Χρησιμοποιεί το CompatNet από denoise_debug_dualmode_COMPAT.py,
# το οποίο επιστρέφει RESIDUAL. Κάνει denoised = input - residual.

import argparse, sys
from pathlib import Path
import importlib.util
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

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

def hann2d(h, w):
    wy = np.hanning(max(h, 2))
    wx = np.hanning(max(w, 2))
    w2 = np.outer(wy, wx).astype(np.float32)
    eps = 1e-6
    w2 = (w2 - w2.min()) / (w2.max() - w2.min() + eps)
    w2 = (w2 * (1 - 2*eps) + eps).astype(np.float32)
    return w2

def safe_pad(patch, ph, pw):
    # 'replicate' για να μη σκάει σε μεγάλα pads
    return F.pad(patch, (0, pw, 0, ph), mode="replicate")

@torch.no_grad()
def denoise_tiled_residual(model, x, tile=256, overlap=64):
    # x: [1,1,H,W] in [0,1]
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

            resid = model(patch_pad)
            resid = resid[:, :, :h_sub, :w_sub]
            den   = torch.clamp(patch - resid, 0.0, 1.0)

            w_crop = w[:, :, :h_sub, :w_sub]
            out[:, :, top:top+h_sub, left:left+w_sub] += den * w_crop
            wsum[:, :, top:top+h_sub, left:left+w_sub] += w_crop

    out /= torch.clamp(wsum, min=1e-6)
    return out

def extract_statedict(ck):
    if isinstance(ck, dict):
        for k in ("model", "model_state", "state_dict", "net", "weights"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck  # assume plain state-dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compat", default=r"C:\ESPI_DnCNN\denoise_debug_dualmode_COMPAT.py")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    mod = load_module_from_path(args.compat)
    if not hasattr(mod, "CompatNet"):
        print("[ERR] Compat module does not define CompatNet", file=sys.stderr)
        sys.exit(1)
    CompatNet = getattr(mod, "CompatNet")

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = extract_statedict(ck)

    # 1) Προσπάθησε constructor με sd
    try:
        model = CompatNet(sd)
        model = model.to(device).eval()
        print("[info] Instantiated CompatNet(sd) (constructor took state-dict).")
        loaded_ok = True
    except TypeError:
        # 2) Fallback: constructor χωρίς sd + load_state_dict
        model = CompatNet().to(device).eval()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[info] Fallback load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        loaded_ok = True

    in_dir = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={args.device}")

    torch.set_grad_enabled(False)
    for i, p in enumerate(files, 1):
        arr = imread_float01(p)
        x = torch.from_numpy(arr[None, None, :, :]).to(device)
        den = denoise_tiled_residual(model, x, tile=args.tile, overlap=args.overlap)
        den_np = den.squeeze(0).squeeze(0).cpu().numpy()
        imsave_uint01(den_np, out_dir / p.name)
        if i % 20 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {p.name}")

    print(f"[DONE] Saved to {out_dir}")

if __name__ == "__main__":
    main()
