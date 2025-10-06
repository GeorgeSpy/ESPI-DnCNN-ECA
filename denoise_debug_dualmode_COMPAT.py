# -*- coding: utf-8 -*-
# denoise_debug_dualmode_COMPAT.py
# - Φορτώνει checkpoints με κλειδιά τύπου entry./mid./exit. δυναμικά (conv+bn)
# - Βγάζει triptych: [input | (input - pred) ως residual | pred ως clean] + auto-contrast

from __future__ import annotations
import argparse
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

class CompatNet(nn.Module):
    """
    Δυναμικό μοντέλο που χτίζεται από το state_dict:
    - entry.weight/bias : Conv2d(1,64,3,1,1)
    - mid.* : εναλλάξ Conv2d(64,64,3,1,1) και BatchNorm2d(64)
    - exit.weight/bias : Conv2d(64,1,3,1,1)
    Δεν κάνουμε ECA εδώ — σκοπός είναι να φορτώσουμε σωστά τα βάρη για debug.
    """
    def __init__(self, sd: dict):
        super().__init__()
        self.entry = nn.Conv2d(1, 64, 3, 1, 1)
        # Βρες indices mid.* και ταξινόμησέ τους
        mids = sorted({k.split('.')[1] for k in sd.keys() if k.startswith('mid.') and k.endswith('.weight')})
        blocks = []
        used = set()
        i = 0
        while i < len(mids):
            k = int(mids[i])
            wk = f"mid.{k}.weight"
            if wk in sd and sd[wk].ndim == 4:  # Conv
                conv = nn.Conv2d(64, 64, 3, 1, 1)
                blocks.append(conv)
                # Αν ακολουθεί BN (1D weight)
                if i+1 < len(mids):
                    k2 = int(mids[i+1])
                    wk2 = f"mid.{k2}.weight"
                    if wk2 in sd and sd[wk2].ndim == 1:
                        bn = nn.BatchNorm2d(64)
                        blocks.append(bn)
                        i += 1
                blocks.append(nn.ReLU(inplace=True))
            else:
                # Αν είναι BN χωρίς conv, φτιάξ' το και προχώρα
                bn = nn.BatchNorm2d(64)
                blocks.append(bn); blocks.append(nn.ReLU(inplace=True))
            i += 1
        self.mid = nn.Sequential(*blocks)
        self.exit = nn.Conv2d(64, 1, 3, 1, 1)
        # Φόρτωσε βάρη αν ταιριάζουν σε σχήμα
        self._load_from_sd(sd)

    def _assign(self, mod, prefix, sd):
        for name, param in mod.named_parameters():
            k = f"{prefix}.{name}"
            if k in sd and sd[k].shape == param.data.shape:
                param.data.copy_(sd[k])

        # BatchNorm buffers
        if isinstance(mod, nn.BatchNorm2d):
            for bname in ["running_mean", "running_var", "num_batches_tracked"]:
                k = f"{prefix}.{bname}"
                if k in sd and getattr(mod, bname).shape == sd[k].shape:
                    getattr(mod, bname).data.copy_(sd[k])

    def _load_from_sd(self, sd):
        self._assign(self.entry, "entry", sd)
        # map blocks με βάση εμφάνιση στο sd
        # Ξανα-περνάμε τα mid.* κατά σειρά και αντιστοιχίζουμε
        mid_keys = sorted([k for k in sd.keys() if k.startswith("mid.")])
        bidx = 0
        i = 0
        while i < len(mid_keys) and bidx < len(self.mid):
            k = mid_keys[i]
            if isinstance(self.mid[bidx], nn.Conv2d) and k.endswith(".weight") and sd[k].ndim == 4:
                self._assign(self.mid[bidx], k.rsplit('.',1)[0], sd)
                bidx += 1
            elif isinstance(self.mid[bidx], nn.BatchNorm2d) and k.endswith(".weight") and sd[k].ndim == 1:
                base = k.rsplit('.',1)[0]
                self._assign(self.mid[bidx], base, sd)
                bidx += 1
            i += 1
        self._assign(self.exit, "exit", sd)

    def forward(self, x):  # type: ignore
        x = self.entry(x)
        x = self.mid(x)
        x = self.exit(x)
        return x

def extract_sd(ck: dict):
    for k in ("model_state","model","state_dict","net","weights"):
        if k in ck and isinstance(ck[k], dict): return ck[k]
    if all(isinstance(v, torch.Tensor) for v in ck.values()): return ck
    raise RuntimeError("No state_dict-like key found in checkpoint")

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

def make_triptych(inp, den_resid, den_clean, out_path: Path):
    h, w = inp.shape
    canvas = np.zeros((h, w*3), np.float32)
    canvas[:, 0:w]   = inp
    canvas[:, w:2*w] = den_resid
    canvas[:, 2*w:]  = den_clean
    imsave01(canvas, out_path)
    # Auto-contrast
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
    ap.add_argument("--percentile_norm", type=float, default=0.0)  # 0=off
    args = ap.parse_args()

    inp_dir = Path(args.input); out_dir = Path(args.out)
    files = sorted([p for p in inp_dir.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")])
    if args.limit>0: files = files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={args.device}")

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = extract_sd(ck)
    # Αν βρούμε entry/mid/exit → CompatNet, αλλιώς basic
    if any(k.startswith("entry.") for k in sd.keys()):
        model = CompatNet(sd)
        missing, unexpected = [], []
    else:
        # basic DnCNN fallback
        class Fallback(nn.Module):
            def __init__(self, depth=17, ch=64):
                super().__init__()
                layers=[nn.Conv2d(1,ch,3,1,1), nn.ReLU(True)]
                for _ in range(depth-2): layers += [nn.Conv2d(ch,ch,3,1,1), nn.ReLU(True)]
                layers += [nn.Conv2d(ch,1,3,1,1)]
                self.net = nn.Sequential(*layers)
            def forward(self,x): return self.net(x)
        model = Fallback()
        missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

    model.to(args.device).eval()

    for i,f in enumerate(files,1):
        x = imread01(f)
        if args.percentile_norm and args.percentile_norm>0:
            p_lo=100.0-args.percentile_norm; p_hi=args.percentile_norm
            lo,hi = np.percentile(x,[p_lo,p_hi]).astype(np.float32)
            if hi>lo+1e-6: x = np.clip((x-lo)/(hi-lo),0,1)
        xt = torch.from_numpy(x)[None,None].to(args.device)

        with torch.no_grad():
            pred = denoise_tiled(model, xt, args.tile, args.overlap).detach().clamp(0,1)
        pred_np = pred.squeeze().cpu().numpy()
        den_resid = np.clip(x - pred_np, 0, 1)  # assume residual
        den_clean = np.clip(pred_np, 0, 1)      # assume clean
        print(f"[{i}/{len(files)}] {f.name}  var(resid)={float(np.var(den_resid)):.6f}  var(clean)={float(np.var(den_clean)):.6f}")
        make_triptych(x, den_resid, den_clean, out_dir / (f.stem + "_triptych.png"))

    print(f"[DONE] Triptychs at: {out_dir}")

if __name__ == "__main__":
    main()
