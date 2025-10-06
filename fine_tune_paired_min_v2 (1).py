#!/usr/bin/env python3
# fine_tune_paired_min_v2.py (fixed: Remove verbose= from ReduceLROnPlateau for older torch)
import argparse, math, random, time, importlib.util, sys, os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def imread_f32(p: Path):
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 255: arr = arr / 65535.0
    else:               arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)

def center_crop_pair(noisy, clean, size):
    h, w = noisy.shape
    ch = min(h, size); cw = min(w, size)
    y0 = (h - ch) // 2; x0 = (w - cw) // 2
    return noisy[y0:y0+ch, x0:x0+cw], clean[y0:y0+ch, x0:x0+cw]

def random_crop_pair(noisy, clean, size):
    h, w = noisy.shape
    if h < size or w < size:
        pad_h = max(0, size - h); pad_w = max(0, size - w)
        noisy = np.pad(noisy, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode="reflect")
        clean = np.pad(clean, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode="reflect")
        h, w = noisy.shape
    y0 = random.randint(0, h - size); x0 = random.randint(0, w - size)
    return noisy[y0:y0+size, x0:x0+size], clean[y0:y0+size, x0:y0+size]

def psnr(x, y):
    mse = F.mse_loss(x, y, reduction='mean').item()
    if mse == 0: return 99.0
    return 10.0 * math.log10(1.0 / mse)

class PairedESPIDataset(Dataset):
    def __init__(self, noisy_root, clean_root, split="train", val_ratio=0.2, seed=42,
                 ext=".png", crop=224, aug=False):
        self.noisy_root = Path(noisy_root); self.clean_root = Path(clean_root)
        self.ext = ext; self.crop = int(crop) if crop else 0; self.aug = bool(aug)
        clean_files = sorted(self.clean_root.glob(f"*{self.ext}"))
        pairs = []
        for c in clean_files:
            n = self.noisy_root / c.name
            if n.exists():
                pairs.append((str(n), str(c)))
        if not pairs:
            raise ValueError(f"No paired files found between {noisy_root} and {clean_root}")
        rng = np.random.RandomState(seed)
        idx = np.arange(len(pairs)); rng.shuffle(idx)
        n_val = max(1, int(len(pairs)*val_ratio))
        if split == "train": sel = idx[n_val:]
        else:                sel = idx[:n_val]
        self.pairs = [pairs[i] for i in sel]

    def __len__(self): return len(self.pairs)

    def _load_gray(self, p): return imread_f32(Path(p))

    def __getitem__(self, i):
        noisy_p, clean_p = self.pairs[i]
        noisy = self._load_gray(noisy_p); clean = self._load_gray(clean_p)
        if self.crop and self.crop > 0:
            if self.aug and random.random() < 0.5:
                noisy, clean = random_crop_pair(noisy, clean, self.crop)
            else:
                noisy, clean = center_crop_pair(noisy, clean, self.crop)
        if self.aug:
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=1).copy(); clean = np.flip(clean, axis=1).copy()
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=0).copy(); clean = np.flip(clean, axis=0).copy()
        noisy = torch.from_numpy(noisy).unsqueeze(0).float()
        clean = torch.from_numpy(clean).unsqueeze(0).float()
        return noisy, clean

class ECALite(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg(x)                        # [B,C,1,1]
        y = y.squeeze(-1).transpose(1,2)      # [B,1,C]
        y = self.conv(y)
        y = self.sigmoid(y).transpose(1,2).unsqueeze(-1)
        return x * y

class DnCNNLiteECA(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=64, depth=17, eca_every=4, eca_ks=3):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, features, 3, padding=1)
        layers = []
        for i in range(depth-2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
            if eca_every>0 and ((i+1) % eca_every == 0):
                layers += [ECALite(features, k_size=eca_ks)]
        self.mid = nn.Sequential(*layers)
        self.exit = nn.Conv2d(features, out_ch, 3, padding=1)
    def forward(self, x):
        y = self.entry(x)
        y = self.mid(y)
        noise = self.exit(y)
        return x - noise

def try_load_external_model(def_path: str, class_name: str, in_ch=1, out_ch=1, features=64, depth=17):
    spec = importlib.util.spec_from_file_location("ext_model_mod", def_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {def_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    cls = getattr(mod, class_name, None)
    if cls is None: raise RuntimeError(f"Class {class_name} not found in {def_path}")
    try:
        model = cls(in_channels=in_ch, out_channels=out_ch, features=features, num_layers=depth)
    except TypeError:
        try:    model = cls(in_ch, out_ch, depth, features)
        except Exception:
            model = cls()
    return model

def gaussian_window(size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size//2
    g = torch.exp(-(coords**2) / (2*sigma*sigma)); g /= g.sum()
    return (g[:,None] * g[None,:])

def ssim_torch(x, y, window=None, C1=0.01**2, C2=0.03**2):
    if window is None:
        window = gaussian_window(11, 1.5, x.device)
    w = window.expand(x.size(1),1,window.size(0),window.size(1))
    mu_x = F.conv2d(x, w, padding=window.size(0)//2, groups=x.size(1))
    mu_y = F.conv2d(y, w, padding=window.size(0)//2, groups=y.size(1))
    mu_x2 = mu_x*mu_x; mu_y2 = mu_y*mu_y; mu_xy = mu_x*mu_y
    sigma_x2 = F.conv2d(x*x, w, padding=window.size(0)//2, groups=x.size(1)) - mu_x2
    sigma_y2 = F.conv2d(y*y, w, padding=window.size(0)//2, groups=y.size(1)) - mu_y2
    sigma_xy = F.conv2d(x*y, w, padding=window.size(0)//2, groups=x.size(1)) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2))/((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map.mean()

def sobel_grad(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

class CombinedLoss(nn.Module):
    def __init__(self, w_mse=0.9, w_ssim=0.1, w_edge=0.0, clamp=True):
        super().__init__()
        self.w_mse = w_mse; self.w_ssim = w_ssim; self.w_edge = w_edge; self.clamp = clamp
    def forward(self, out, target):
        if self.clamp:
            out = torch.clamp(out, 0.0, 1.0)
        mse = F.mse_loss(out, target)
        ssim = ssim_torch(out, target)
        edge = F.l1_loss(sobel_grad(out), sobel_grad(target)) if self.w_edge>0 else torch.tensor(0.0, device=out.device)
        loss = self.w_mse*mse + self.w_ssim*(1.0-ssim) + self.w_edge*edge
        return loss, (mse.item(), ssim.item(), edge.item())

def freeze_bn_(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def run_epoch(model, dl, opt, criterion, device, train=True):
    model.train(train)
    tot = 0.0; n=0
    mse_sum=0.0; ssim_sum=0.0; edge_sum=0.0; psnr_sum=0.0
    for noisy, clean in dl:
        noisy = noisy.to(device); clean = clean.to(device)
        if train: opt.zero_grad(set_to_none=True)
        out = model(noisy)
        loss, (lm, ls, le) = criterion(out, clean)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            psnr_sum += psnr(torch.clamp(out,0,1), clean)
        tot += loss.item(); mse_sum += lm; ssim_sum += ls; edge_sum += le; n += 1
    return tot/n, mse_sum/n, ssim_sum/n, edge_sum/n, psnr_sum/n

def save_triplets(model, dl, device, outdir: Path, max_items=3):
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for noisy, clean in dl:
            noisy = noisy.to(device); clean = clean.to(device)
            out = torch.clamp(model(noisy), 0, 1)
            B = noisy.size(0)
            for i in range(B):
                n = noisy[i,0].cpu().numpy()
                o = out[i,0].cpu().numpy()
                c = clean[i,0].cpu().numpy()
                h, w = n.shape
                trip = np.concatenate([n, o, c], axis=1)
                img = (trip*255.0 + 0.5).astype(np.uint8)
                Image.fromarray(img).save(outdir / f"val_triplet_{saved:02d}.png")
                saved += 1
                if saved >= max_items: return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy-root", required=True)
    ap.add_argument("--clean-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-5)

    ap.add_argument("--w-mse", type=float, default=0.9)
    ap.add_argument("--w-ssim", type=float, default=0.1)
    ap.add_argument("--w-edge", type=float, default=0.0)
    ap.add_argument("--no-clamp", action="store_true", help="Disable clamp[0,1] before loss")
    ap.add_argument("--freeze-bn", action="store_true", help="Freeze BatchNorm stats/affine")

    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--model-def", type=str, default=None)
    ap.add_argument("--model-class", type=str, default=None)
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--out-ch", type=int, default=1)
    ap.add_argument("--features", type=int, default=64)
    ap.add_argument("--depth", type=int, default=17)
    ap.add_argument("--eca-every", type=int, default=4)
    ap.add_argument("--eca-ks", type=int, default=3)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    ds_tr = PairedESPIDataset(args.noisy_root, args.clean_root, split="train",
                              val_ratio=args.val_ratio, seed=args.seed, ext=".png",
                              crop=args.crop, aug=True)
    ds_va = PairedESPIDataset(args.noisy_root, args.clean_root, split="val",
                              val_ratio=args.val_ratio, seed=args.seed, ext=".png",
                              crop=args.crop, aug=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=(device.type=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=0, pin_memory=(device.type=="cuda"))

    # model
    if args.model_def and args.model_class:
        print(f"[MODEL] Importing {args.model_class} from {args.model_def}")
        model = try_load_external_model(args.model_def, args.model_class,
                                        in_ch=args.in_ch, out_ch=args.out_ch,
                                        features=args.features, depth=args.depth)
    else:
        print(f"[MODEL] Using built-in DnCNNLiteECA (features={args.features}, depth={args.depth}, eca_every={args.eca_every})")
        model = DnCNNLiteECA(in_ch=args.in_ch, out_ch=args.out_ch, features=args.features,
                             depth=args.depth, eca_every=args.eca_every, eca_ks=args.eca_ks)

    if args.resume and os.path.exists(args.resume):
        print(f"[RESUME] Loading checkpoint (strict=False): {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model_state", ckpt if isinstance(ckpt, dict) else None)
        if state is None: state = ckpt
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            new_state = {k.replace("module.",""): v for k,v in state.items()}
            model.load_state_dict(new_state, strict=False)
        print("[RESUME] Loaded.")

    model.to(device)

    if args.freeze_bn:
        print("[BN] Freezing BatchNorm (eval + no-grad).")
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Remove 'verbose' for compatibility with older torch
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    criterion = CombinedLoss(w_mse=args.w_mse, w_ssim=args.w_ssim, w_edge=args.w_edge, clamp=(not args.no_clamp))

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)

    best_val = 1e9; best_epoch = -1
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr, tr_m, tr_s, tr_e, tr_p = run_epoch(model, dl_tr, opt, criterion, device, train=True)
        va, va_m, va_s, va_e, va_p = run_epoch(model, dl_va, opt, criterion, device, train=False)
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | Train {tr:.4f} (MSE {tr_m:.4f} SSIM {tr_s:.4f} EDGE {tr_e:.4f})"
              f" | Val {va:.4f} (MSE {va_m:.4f} SSIM {va_s:.4f} EDGE {va_e:.4f})"
              f" | PSNR {va_p:.2f} | time {dt:.1f}s")
        sched.step(va)

        # save last
        ck = {"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(), "args": vars(args)}
        torch.save(ck, outdir / "checkpoints" / "last_finetune.pth")

        if va < best_val:
            best_val = va; best_epoch = epoch
            torch.save(ck, outdir / "checkpoints" / "best_finetune.pth")
            print(f"[BEST] epoch {epoch}  val={va:.6f}")
            save_triplets(model, dl_va, device, outdir / "samples", max_items=3)

    print(f"[DONE] Best val {best_val:.6f} at epoch {best_epoch}.")
    print(f"Checkpoints under: {outdir / 'checkpoints'}")
    print(f"Sample triplets under: {outdir / 'samples'}")

if __name__ == "__main__":
    main()
