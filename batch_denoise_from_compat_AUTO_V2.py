# coding: utf-8
"""
AUTO denoise από fine-tuned checkpoint που ταιριάζει στο CompatNet(sd)
του denoise_debug_dualmode_COMPAT.py. Χωρίς build_compat_model·
χρησιμοποιεί απευθείας CompatNet(sd).
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from importlib.machinery import SourceFileLoader
from tqdm import tqdm

def imread_gray_float01(p: Path)->np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float32)/255.0
def imsave_uint01(a: np.ndarray, p: Path):
    a = np.clip(a,0,1); Image.fromarray((a*255.0+0.5).astype(np.uint8)).save(p)
def ptiles(x, lo, hi):
    lo=float(np.percentile(x,lo)); hi=float(np.percentile(x,hi))
    if hi-lo<1e-6: hi=lo+1e-6
    return lo,hi

def extract_patch(x, top, left, tile):
    _,_,H,W = x.shape
    h = min(tile, H - top)
    w = min(tile, W - left)
    patch = x[:, :, top:top+h, left:left+w]
    ph = tile - h
    pw = tile - w
    if ph > 0 or pw > 0:
        # reflect μόνο αν το pad είναι μικρότερο από τη διάσταση-1 και h,w>1,
        # αλλιώς replicate (πάντα ασφαλές).
        safe_reflect = (h > 1 and w > 1 and ph <= h-1 and pw <= w-1)
        mode = "reflect" if safe_reflect else "replicate"
        patch = F.pad(patch, (0, pw, 0, ph), mode=mode)
    return patch, h, w


def stitch(out, top,left, y, h,w):
    out[:,:,top:top+h,left:left+w]=y[:,:,:h,:w]

def forward_tiled(net, x, tile, overlap):
    step=tile-overlap; out=torch.zeros_like(x)
    for top in range(0,x.shape[2],step):
        for left in range(0,x.shape[3],step):
            patch,h,w=extract_patch(x,top,left,tile)
            with torch.no_grad(): y=net(patch)
            stitch(out,top,left,y,h,w)
    return out

def choose_mode(x0, y_clean, y_resid):
    def score(y):
        sat=float(((y<=0)|(y>=1)).mean())
        std=float(y.std()+1e-6); std_in=float(x0.std()+1e-6)
        return sat+0.5*abs(math.log((std)/(std_in)))
    return ("clean",y_clean) if score(y_clean)<=score(y_resid) else ("residual",y_resid)

def load_sd(ckpt_path, device):
    ck=torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict):
        if "model_state" in ck: return ck["model_state"]
        if "model" in ck and isinstance(ck["model"], dict): return ck["model"]
    return ck  # assume state_dict

def main():
    ap=argparse.ArgumentParser()
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
    args=ap.parse_args()

    # load compat module & model
    compat = SourceFileLoader("compat", args.compat).load_module()
    sd = load_sd(args.ckpt, args.device)
    net = compat.CompatNet(sd).to(args.device).eval()

    in_dir=Path(args.input); out_dir=Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    files=sorted(in_dir.glob("*.png")); 
    if args.limit>0: files=files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={args.device} mode={args.mode}")

    for p in tqdm(files):
        x0 = imread_gray_float01(p)                     # 0..1 input
        a,b = ptiles(x0, args.p_lo, args.p_hi)          # per-image norm
        x = np.clip((x0-a)/(b-a), 0.0, 1.0)

        xt = torch.from_numpy(x)[None,None,...].to(args.device)
        yt = forward_tiled(net, xt, args.tile, args.overlap)
        y_norm = np.clip(yt.squeeze().detach().cpu().numpy().astype(np.float32), 0.0, 1.0)

        # δύο υποψήφια στο original scale
        y_clean = np.clip(y_norm*(b-a)+a, 0.0, 1.0)
        y_resid = np.clip(x0 - y_norm*(b-a), 0.0, 1.0)

        if args.mode=="clean": picked,y="clean",y_clean
        elif args.mode=="residual": picked,y="residual",y_resid
        else: picked,y = choose_mode(x0,y_clean,y_resid)

        # ορατότητα για PNG
        v_hi = float(np.percentile(y, args.viz_percentile))
        if v_hi<1e-6: v_hi=1.0
        y_viz = np.clip(y / v_hi, 0.0, 1.0)
        imsave_uint01(y_viz, out_dir/p.name)

        sys.stdout.write(f"\r[{p.name}] {picked} in[{x0.min():.3f},{x0.max():.3f}] out[{y.min():.3f},{y.max():.3f}]   ")
        sys.stdout.flush()
    print(f"\n[DONE] Saved to {str(out_dir)}")

if __name__=="__main__":
    main()
