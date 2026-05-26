#!/usr/bin/env python3
"""
Retrain DnCNN on REAL noisy-clean pairs.

Instead of synthetic noise, uses aligned real single-shot → averaged pairs.

Usage:
  # V4 no-ECA on real pairs
  py retrain_dncnn_real.py --clean-dir real_pairs/clean --noisy-dir real_pairs/noisy --use-eca 0 --tag v4r_noeca --version v4

  # V4 ECA on real pairs
  py retrain_dncnn_real.py --clean-dir real_pairs/clean --noisy-dir real_pairs/noisy --use-eca 1 --tag v4r_eca --version v4

  # V5 ECA on real pairs
  py retrain_dncnn_real.py --clean-dir real_pairs/clean --noisy-dir real_pairs/noisy --use-eca 1 --tag v5r_eca --version v5
"""
import argparse, csv, math, random, contextlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ─── Reproducibility ───
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ─── Image I/O ───
def imread_uint(path):
    """Read 8/16-bit PNG as float32 [0,1]. Matches V4 script."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(img.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

# ─── ECA modules (same as V4/V5) ───
def eca_kernel_for(C, gamma=2.0, b=1.0):
    return max(3, int(round(math.log2(max(C,1)) / gamma + b / gamma)) | 1)

class ECA_v4(nn.Module):
    def __init__(self, ch, k_size=0, temp=0.75, gain=0.5, centered=True):
        super().__init__()
        k = k_size if (k_size and k_size%2==1) else eca_kernel_for(ch)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,k,padding=k//2,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.temp=temp; self.gain=gain; self.centered=centered
    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(1,2)
        y = self.conv(y)
        g = self.sigmoid(y / max(self.temp,1e-6)).transpose(1,2).unsqueeze(-1)
        return x * ((1.0 + self.gain*(g-0.5)*2.0) if self.centered else g)

class ECA_v5(nn.Module):
    def __init__(self, ch, k_size=0, temp=1.0, gain=0.25, centered=True, use_maxpool=True, learnable=True):
        super().__init__()
        k = k_size if (k_size and k_size%2==1) else eca_kernel_for(ch)
        self.gap = nn.AdaptiveAvgPool2d(1); self.gmp = nn.AdaptiveMaxPool2d(1)
        self.use_maxpool = use_maxpool; self.centered = centered
        self.conv_avg = nn.Conv1d(1,1,k,padding=k//2,bias=False)
        self.conv_max = nn.Conv1d(1,1,k,padding=k//2,bias=False) if use_maxpool else None
        self.sigmoid = nn.Sigmoid()
        self.temp_min=1e-6; self.gain_min=0.01; self.gain_max=1.5
        if learnable:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(max(float(temp),1e-6))))
            frac = min(1-1e-4, max(1e-4, (float(gain)-self.gain_min)/(self.gain_max-self.gain_min)))
            self.raw_gain = nn.Parameter(torch.tensor(math.log(frac/(1-frac))))
            self.learnable = True
        else:
            self.register_buffer("temp_c", torch.tensor(float(temp)))
            self.register_buffer("gain_c", torch.tensor(float(gain)))
            self.learnable = False
    def _temp(self): return (F.softplus(self.log_temp)+self.temp_min) if self.learnable else self.temp_c
    def _gain(self): return (self.gain_min+(self.gain_max-self.gain_min)*torch.sigmoid(self.raw_gain)) if self.learnable else self.gain_c
    def forward(self, x):
        y = self.conv_avg(self.gap(x).squeeze(-1).transpose(1,2))
        if self.use_maxpool and self.conv_max is not None:
            y = 0.5*(y + self.conv_max(self.gmp(x).squeeze(-1).transpose(1,2)))
        g = self.sigmoid(y / torch.clamp(self._temp(), min=self.temp_min)).transpose(1,2).unsqueeze(-1)
        gain = torch.clamp(self._gain(), self.gain_min, self.gain_max)
        return x * ((1.0 + gain*(g-0.5)*2.0) if self.centered else g)

# ─── DnCNN Model ───
def make_norm(kind, ch, gn=8):
    if kind=="batch": return nn.BatchNorm2d(ch)
    if kind=="group": return nn.GroupNorm(min(gn,ch), ch)
    return nn.Identity()

class DnCNNLite(nn.Module):
    def __init__(self, version="v4", features=32, depth=17, norm="group", gn=8,
                 use_eca=False, eca_positions=None, eca_k=5):
        super().__init__()
        c = features
        self.head = nn.Sequential(nn.Conv2d(1,c,3,1,1,bias=False), nn.ReLU(True))
        blocks, ecas = [], []
        pos = set(eca_positions or [])
        for i in range(depth-2):
            blocks.append(nn.Sequential(nn.Conv2d(c,c,3,1,1,bias=False), make_norm(norm,c,gn), nn.ReLU(True)))
            if use_eca and i in pos:
                ecas.append(ECA_v5(c, k_size=eca_k, learnable=True) if version=="v5" else ECA_v4(c, k_size=eca_k))
            else:
                ecas.append(nn.Identity())
        self.blocks = nn.ModuleList(blocks); self.eca = nn.ModuleList(ecas)
        self.spa = nn.ModuleList([nn.Identity()]*(depth-2))
        self.tail = nn.Conv2d(c,1,3,1,1,bias=False)
    def forward(self, x):
        h = self.head(x)
        for b, e, s in zip(self.blocks, self.eca, self.spa):
            h = s(e(b(h)))
        return x - self.tail(h)

# ─── Dataset: Real Pairs ───
class RealPairDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, tile=256, augment=True):
        self.clean_dir = Path(clean_dir); self.noisy_dir = Path(noisy_dir)
        clean_files = {p.name for p in self.clean_dir.glob('*.png')}
        noisy_files = {p.name for p in self.noisy_dir.glob('*.png')}
        common = sorted(clean_files & noisy_files)
        self.names = common
        self.tile = tile; self.augment = augment
        print(f"[DATA] {len(common)} paired images found")

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        clean = imread_uint(self.clean_dir / name)
        noisy = imread_uint(self.noisy_dir / name)

        # Ensure same size
        h = min(clean.shape[0], noisy.shape[0])
        w = min(clean.shape[1], noisy.shape[1])
        clean = clean[:h,:w]; noisy = noisy[:h,:w]

        # Random crop
        t = self.tile
        if h >= t and w >= t:
            top = random.randint(0, h-t); left = random.randint(0, w-t)
            clean = clean[top:top+t, left:left+t]
            noisy = noisy[top:top+t, left:left+t]
        else:
            # Pad if too small
            clean = np.pad(clean, ((0,max(0,t-h)),(0,max(0,t-w))), mode='reflect')[:t,:t]
            noisy = np.pad(noisy, ((0,max(0,t-h)),(0,max(0,t-w))), mode='reflect')[:t,:t]

        c = torch.from_numpy(clean).unsqueeze(0).float()
        n = torch.from_numpy(noisy).unsqueeze(0).float()

        # Augment
        if self.augment:
            if random.random() < 0.5: c = torch.flip(c,[-1]); n = torch.flip(n,[-1])
            if random.random() < 0.2: c = torch.flip(c,[-2]); n = torch.flip(n,[-2])

        return n, c  # noisy input, clean target

# ─── Loss (stable for dark ESPI images) ───
class CharbonnierLoss(nn.Module):
    """Charbonnier (smooth L1) + MSE — NaN-safe for dark images."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        diff = pred - target
        charb = torch.sqrt(diff**2 + self.eps**2).mean()
        mse = F.mse_loss(pred, target)
        return 0.7 * charb + 0.3 * mse

# ─── PSNR ───
def psnr(x, y):
    mse = F.mse_loss(x, y).item()
    if mse < 1e-10: return 50.0
    return 10*math.log10(1.0/mse)

# ─── Training ───
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clean-dir', required=True)
    p.add_argument('--noisy-dir', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--version', default='v4', choices=['v4','v5'])
    p.add_argument('--use-eca', type=int, default=0)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', default=r'C:\ESPI\data\output')
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir) / args.tag
    ckpt_dir = out / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ECA positions
    if args.version == 'v4':
        eca_pos = [0,1,2]
    else:
        eca_pos = [0,1,2,3,6,10,14]

    model = DnCNNLite(
        version=args.version, features=32, depth=17,
        use_eca=bool(args.use_eca), eca_positions=eca_pos
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {args.version} use_eca={bool(args.use_eca)} params={n_params:,}")

    # Dataset
    ds = RealPairDataset(args.clean_dir, args.noisy_dir, tile=args.tile)
    n_val = max(200, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    # Disable augment for val
    # (val_ds still uses same underlying dataset, but augment is random so effectively light noise)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = CharbonnierLoss().to(device)

    log_csv = out / 'train_log.csv'
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_loss','val_psnr','lr'])

    best_val = float('inf')
    debug_done = False
    for ep in range(1, args.epochs+1):
        # Train
        model.train()
        tl, tb = 0.0, 0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            # Debug first batch
            if not debug_done:
                print(f"[DEBUG] noisy: shape={noisy.shape} range=[{noisy.min():.4f},{noisy.max():.4f}] mean={noisy.mean():.4f}")
                print(f"[DEBUG] clean: shape={clean.shape} range=[{clean.min():.4f},{clean.max():.4f}] mean={clean.mean():.4f}")
                with torch.no_grad():
                    test_pred = model(noisy)
                    print(f"[DEBUG] pred:  range=[{test_pred.min():.4f},{test_pred.max():.4f}] mean={test_pred.mean():.4f}")
                    test_loss = crit(test_pred, clean)
                    print(f"[DEBUG] loss={test_loss.item():.6f} finite={torch.isfinite(test_loss).item()}")
                debug_done = True
            opt.zero_grad()
            pred = model(noisy)
            loss = crit(pred, clean)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item(); tb += 1
            elif tb == 0 and ep == 1:
                print(f"[WARN] NaN loss at batch {tb}, loss={loss.item()}, pred range=[{pred.min():.4f},{pred.max():.4f}]")

        # Val
        model.eval()
        vl, vp, vb = 0.0, 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = torch.clamp(model(noisy), 0, 1)
                loss = crit(pred, clean)
                if torch.isfinite(loss):
                    vl += loss.item()
                    vp += psnr(pred, clean)
                    vb += 1

        train_loss = tl/max(tb,1)
        val_loss = vl/max(vb,1)
        val_psnr = vp/max(vb,1)
        lr = sched.get_last_lr()[0]
        sched.step()

        print(f"[{args.tag}] Ep {ep:02d}: TrainL={train_loss:.6f} ValL={val_loss:.6f} PSNR={val_psnr:.2f} LR={lr:.2e} (tb={tb} vb={vb})")

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{val_psnr:.2f}', f'{lr:.2e}'])

        # Save
        ckpt = {'model': model.state_dict(), 'epoch': ep, 'val_loss': val_loss, 'val_psnr': val_psnr}
        torch.save(ckpt, ckpt_dir / 'last.pth')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / 'best.pth')
            print(f"  ★ New best val_loss={val_loss:.4f}")

    print(f"\n[{args.tag}] DONE. Best val_loss={best_val:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best.pth'}")

if __name__ == '__main__':
    main()
