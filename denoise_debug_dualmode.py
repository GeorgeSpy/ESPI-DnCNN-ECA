# -*- coding: utf-8 -*-
# denoise_debug_dualmode.py
# Runs model and saves: [input | residual-mode | clean-mode] + auto-contrast versions
from __future__ import annotations
import argparse, importlib.util, inspect
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F

def imread01(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), np.float32) / 255.0

def imsave01(x: np.ndarray, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, 0.0, 1.0)
    Image.fromarray((x*255.0+0.5).astype(np.uint8)).save(str(p))

def autocontrast01(x: np.ndarray, lo=1.0, hi=99.0):
    lo_v, hi_v = np.percentile(x, [lo, hi])
    if hi_v <= lo_v + 1e-6: return np.zeros_like(x)
    y = (x - lo_v) / (hi_v - lo_v)
    return np.clip(y, 0.0, 1.0)

def hann2d(h,w):
    return torch.outer(torch.hann_window(h, False), torch.hann_window(w, False))

@torch.no_grad()
def denoise_tiled(model, xt, tile, overlap):
    _,_,H,W = xt.shape
    out = torch.zeros_like(xt)
    acc = torch.zeros_like(xt)
    step = max(1, tile - overlap)
    for top in range(0, H, step):
        for left in range(0, W, step):
            h0 = min(tile, H-top); w0 = min(tile, W-left)
            patch = xt[:,:,top:top+h0, left:left+w0]
            ph = tile-h0; pw = tile-w0
            if ph>0 or pw>0:
                patch = F.pad(patch, (0,pw,0,ph), mode="replicate")
            pred = model(patch)[:,:,:h0,:w0]
            w = hann2d(h0,w0).to(xt.device)[None,None]
            out[:,:,top:top+h0, left:left+w0] += pred*w
            acc[:,:,top:top+h0, left:left+w0] += w
    return out/torch.clamp(acc, min=1e-6)

def try_import_training_class():
    here = Path(__file__).resolve().parent
    for name in [
        "espi_dncnn_lite_eca_full_cpu_safe_FIXED_PATCHED_v2.py",
        "espi_dncnn_lite_eca_full_cpu_safe_FIXED.py",
        "espi_dncnn_lite_eca_full_cpu_safe.py",
    ]:
        f = here/name
        if not f.exists(): continue
        try:
            spec = importlib.util.spec_from_file_location(f.stem, str(f))
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(mod)                 # type: ignore
            for nm, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, nn.Module) and ("dncnn" in nm.lower() or "eca" in nm.lower()):
                    # build kwargs defensively
                    sig = inspect.signature(obj.__init__)
                    kwargs = {}
                    for k in ("in_ch","in_channels","out_ch","out_channels"): 
                        if k in sig.parameters: kwargs[k]=1
                    if "features" in sig.parameters: kwargs["features"]=64
                    if "depth" in sig.parameters: kwargs["depth"]=17
                    if "eca_every" in sig.parameters: kwargs["eca_every"]=4
                    return obj, kwargs
        except Exception:
            pass
    # fallback plain dncnn
    class Fallback(nn.Module):
        def __init__(self, depth=17, ch=64):
            super().__init__()
            layers=[nn.Conv2d(1,ch,3,1,1), nn.ReLU(True)]
            for _ in range(depth-2): layers += [nn.Conv2d(ch,ch,3,1,1), nn.ReLU(True)]
            layers += [nn.Conv2d(ch,1,3,1,1)]
            self.net = nn.Sequential(*layers)
        def forward(self,x): return self.net(x)
    return Fallback, {}

def extract_sd(ck: dict):
    for k in ("model_state","model","state_dict","net","weights"):
        if k in ck and isinstance(ck[k], dict): return ck[k]
    if all(isinstance(v, torch.Tensor) for v in ck.values()): return ck
    raise RuntimeError("No state_dict-like key found")

def make_triptych(inp, den_resid, den_clean, out_path: Path):
    h, w = inp.shape
    canvas = np.zeros((h, w*3), np.float32)
    canvas[:, 0:w]   = inp
    canvas[:, w:2*w] = den_resid
    canvas[:, 2*w:]  = den_clean
    imsave01(canvas, out_path)
    # Also auto-contrast for visibility
    ac = np.zeros_like(canvas)
    ac[:, 0:w]   = autocontrast01(inp)
    ac[:, w:2*w] = autocontrast01(den_resid)
    ac[:, 2*w:]  = autocontrast01(den_clean)
    imsave01(ac, out_path.with_name(out_path.stem + "_AC.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--percentile-norm", type=float, default=0.0) # 0=off, e.g. 99.0
    args = ap.parse_args()

    inp_dir = Path(args.input); out_dir = Path(args.out)
    files = sorted([p for p in inp_dir.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")])
    if args.limit>0: files = files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={args.device}")

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = extract_sd(ck)
    cls, kw = try_import_training_class()
    model = cls(**kw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(args.device).eval()

    for i,f in enumerate(files,1):
        x = imread01(f)
        if args.percentile-norm and args.percentile_norm>0:
            p_lo = 100.0-args.percentile_norm; p_hi = args.percentile_norm
            lo,hi = np.percentile(x,[p_lo,p_hi]).astype(np.float32)
            if hi>lo+1e-6: x = np.clip((x-lo)/(hi-lo),0,1)
        xt = torch.from_numpy(x)[None,None].to(args.device)

        with torch.no_grad():
            pred = denoise_tiled(model, xt, args.tile, args.overlap).detach()
        pred_np = pred.squeeze().cpu().numpy()
        den_resid = np.clip(x - pred_np, 0, 1)    # assume model=RESIDUAL
        den_clean = np.clip(pred_np, 0, 1)        # assume model=CLEAN

        # energy/variance for quick diagnosis
        v_res = float(np.var(den_resid))
        v_cln = float(np.var(den_clean))
        print(f"[{i}/{len(files)}] {f.name}  var(resid)={v_res:.6f}  var(clean)={v_cln:.6f}")

        # save triptychs
        make_triptych(x, den_resid, den_clean, out_dir / (f.stem + "_triptych.png"))

    print(f"[DONE] Triptychs at: {out_dir}")

if __name__ == "__main__":
    main()
