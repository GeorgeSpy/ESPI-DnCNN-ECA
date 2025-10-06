# -*- coding: utf-8 -*-
import argparse, os, glob, numpy as np
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class CompatNet(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.entry = nn.Conv2d(1,64,3,1,1)
        self.mid   = nn.ModuleList([])
        # χτίζουμε 15 blocks Conv+BN+ReLU ώστε να «χωρέσει» τα mid.* keys
        for _ in range(15):
            self.mid.append(nn.Conv2d(64,64,3,1,1))
            self.mid.append(nn.BatchNorm2d(64))
            self.mid.append(nn.ReLU(inplace=True))
        self.exit  = nn.Conv2d(64,1,3,1,1)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x):
        y = self.entry(x)
        for i in range(0, len(self.mid), 3):
            y = self.mid[i](y)
            y = self.mid[i+1](y)
            y = self.mid[i+2](y)
        y = self.exit(y)
        return y  # προβλέπει RESIDUAL (χωρίς normalization)

def load_sd(path):
    ck = torch.load(path, map_location="cpu")
    for k in ("model","model_state","state_dict"):
        if k in ck and isinstance(ck[k], dict):
            sd = ck[k]
            break
    else:
        # ίσως είναι ήδη καθαρό state_dict
        sd = ck
    # καθάρισε πιθανά "module." prefixes
    sd = {k.replace("module.",""):v for k,v in sd.items()}
    return sd

def imread_f32(p):
    im = Image.open(p).convert("L")
    x = np.asarray(im).astype(np.float32)/255.0
    return x

def tile_forward(net, x, tile=256, overlap=64):
    net.eval()
    with torch.no_grad():
        X = torch.from_numpy(x)[None,None]  # 1x1xH xW
        H,W = X.shape[-2:]
        step = tile - overlap
        out = torch.zeros_like(X)
        wsum = torch.zeros_like(X)
        # Hann
        wx = torch.hann_window(tile, periodic=False)
        wy = torch.hann_window(tile, periodic=False)
        ww = torch.outer(wy, wx)[None,None]  # 1x1xT xT
        for top in range(0, H, step):
            for left in range(0, W, step):
                bh = min(tile, H-top)
                bw = min(tile, W-left)
                patch = X[:,:,top:top+bh, left:left+bw]
                ph = tile - bh; pw = tile - bw
                # reflect padding safe
                if pw>0 or ph>0:
                    mode = "reflect" if (bw>1 and bh>1) else "replicate"
                    patch = F.pad(patch, (0,pw,0,ph), mode=mode)
                y = net(patch)  # residual raw
                out[:,:,top:top+tile, left:left+tile] += y * ww
                wsum[:,:,top:top+tile, left:left+tile] += ww
        out = out / torch.clamp(wsum, 1e-8)
        return out[0,0].cpu().numpy()[:H,:W]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)   # pseudo-noisy PNGs
    ap.add_argument("--out",   required=True)   # θα γράψει .npy residuals εδώ
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    sd = load_sd(args.ckpt)
    net = CompatNet(sd).to("cpu").eval()

    files = sorted(glob.glob(os.path.join(args.input, "*.png")))
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap}")

    for i,fp in enumerate(files,1):
        x = imread_f32(fp)        # [0,1]
        r = tile_forward(net, x, args.tile, args.overlap)  # raw residual (float, unnorm)
        outn = os.path.join(args.out, os.path.basename(fp).replace(".png",".npy"))
        np.save(outn, r.astype(np.float32))
        if i<=3:
            print(f"[{i}] {os.path.basename(fp)}  resid min/max/mean = {r.min():.4f}/{r.max():.4f}/{r.mean():.4f}")
    print("[DONE] raw residuals saved (.npy)")

if __name__ == "__main__":
    main()
