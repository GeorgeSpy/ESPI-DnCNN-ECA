#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-safe ESPI Denoising — DnCNN-Lite + ECA (Single File) with RESUME
- Works on CPU-only machines (Ryzen 5 5600G / 7 5700G) and on CUDA when available
- AMP/GradScaler auto-disabled on CPU (uses torch.amp if available)
- Tile-based validation/inference (Hann window) on full-res
- REAL evaluation (optional): single-shot noisy vs averaged pseudo-clean
- TensorBoard logging (optional): --tensorboard
- ONNX export of best checkpoint (optional): --export-onnx path
- NEW: --resume {auto|path/to/checkpoint.pth} to continue training
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
import random
import csv
import os
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Prefer torch.amp if available
try:
    from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
    AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast as cuda_autocast, GradScaler as CudaGradScaler
    AMP_NEW = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False
    SummaryWriter = None

# ----------------------- Determinism & Utils -----------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_pngs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.png"))

def imread_uint(path: Path) -> np.ndarray:
    """Read 8/16-bit PNG as float32 in [0,1]."""
    import PIL.Image as Image
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(img.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

# ----------------------- Model: DnCNN-Lite + ECA -----------------------

def kaiming_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        k = k_size if (k_size % 2 == 1) else (k_size + 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)                       # [B,C,1,1]
        y = y.squeeze(-1).transpose(1, 2)     # [B,1,C]
        y = self.conv1d(y)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y

def make_norm(kind: str, ch: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        for g in [8,4,2,1]:
            if ch % g == 0:
                return nn.GroupNorm(g, ch)
        return nn.GroupNorm(1, ch)
    return nn.Identity()

@dataclass
class DnCNNLiteECAConfig:
    in_channels: int = 1
    out_channels: int = 1
    features: int = 32
    depth: int = 17
    norm: str = "group"
    eca_interval: int = 4
    eca_k: int = 3
    residual_weight: float = 1.0

class ConvBlock(nn.Module):
    def __init__(self, ch: int, norm: str = "group"):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm = make_norm(norm, ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))

class DnCNNLiteECA(nn.Module):
    def __init__(self, cfg: DnCNNLiteECAConfig):
        super().__init__()
        c = cfg.features
        self.cfg = cfg
        self.head = nn.Sequential(nn.Conv2d(cfg.in_channels, c, 3, padding=1, bias=False), nn.ReLU(inplace=True))
        blocks, ecas = [], []
        for i in range(cfg.depth - 2):
            blocks.append(ConvBlock(c, cfg.norm))
            ecas.append(ECA(c, cfg.eca_k) if ((i + 1) % cfg.eca_interval == 0) else nn.Identity())
        self.blocks = nn.ModuleList(blocks)
        self.eca = nn.ModuleList(ecas)
        self.tail = nn.Conv2d(c, cfg.out_channels, 3, padding=1, bias=False)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = self.head(x)
        for blk, att in zip(self.blocks, self.eca):
            h = blk(h); h = att(h)
        noise = self.tail(h)
        return identity - self.cfg.residual_weight * noise

# ----------------------- Losses & Metrics -----------------------

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],[-2., 0., 2.],[-1., 0., 1.]]).view(1,1,3,3)
        ky = torch.tensor([[-1.,-2.,-1.],[ 0., 0., 0.],[ 1., 2., 1.]]).view(1,1,3,3)
        self.register_buffer('kx', kx); self.register_buffer('ky', ky)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1); gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-12)

def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    def gaussian_window(size, sigma=1.5):
        gauss = torch.tensor([math.exp(-(i - size//2)**2/(2*sigma**2)) for i in range(size)])
        gauss = gauss / gauss.sum()
        return (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).float()
    C1 = 0.01**2; C2 = 0.03**2
    b, c, h, w = x.shape
    device = x.device
    window = gaussian_window(window_size).to(device).view(1,1,window_size,window_size)
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=c)
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=c)
    sigma_x = F.conv2d(x*x, window, padding=window_size//2, groups=c) - mu_x*mu_x
    sigma_y = F.conv2d(y*y, window, padding=window_size//2, groups=c) - mu_y*mu_y
    sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=c) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-12)
    return 1.0 - ssim_map.mean()

class EdgeAwareLoss(nn.Module):
    def __init__(self, w_l1: float = 0.8, w_ssim: float = 0.2, w_edge: float = 0.1):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_edge = w_l1, w_ssim, w_edge
        self.sobel = Sobel()
    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, tgt); ssim = ssim_loss(pred, tgt)
        e = F.l1_loss(self.sobel(pred), self.sobel(tgt))
        return self.w_l1*l1 + self.w_ssim*ssim + self.w_edge*e

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    mse = F.mse_loss(x, y).item()
    if mse <= 0: return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse + eps))

def ssim_metric(x: torch.Tensor, y: torch.Tensor) -> float:
    return (1.0 - ssim_loss(x, y)).item()

def fringe_edge_f1(x: torch.Tensor, y: torch.Tensor, q: float = 0.85) -> float:
    sob = Sobel().to(x.device); ex = sob(x); ey = sob(y)
    def norm01(t):
        tmin = torch.amin(t, dim=(2,3), keepdim=True); tmax = torch.amax(t, dim=(2,3), keepdim=True)
        return (t - tmin) / (tmax - tmin + 1e-6)
    ex = norm01(ex); ey = norm01(ey)
    B = x.size(0); f1s = []
    for i in range(B):
        thr = torch.quantile(ey[i].flatten(), q)
        px = (ex[i] >= thr).flatten(); py = (ey[i] >= thr).flatten()
        tp = (px & py).sum().item(); fp = (px & ~py).sum().item(); fn = (~px & py).sum().item()
        prec = tp / (tp + fp + 1e-9); rec  = tp / (tp + fn + 1e-9)
        f1s.append(2*prec*rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))

# ----------------------- Tiled inference (Hann) -----------------------

def hann2d(h: int, w: int, eps: float = 1e-6) -> torch.Tensor:
    wx = torch.hann_window(w).unsqueeze(0); wy = torch.hann_window(h).unsqueeze(1)
    return (wy @ wx) + eps

def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    # Accept shapes like [B,1,1,H,W] (from DataLoader over already 4D samples)
    if x.dim() == 5 and x.size(1) == 1:
        x = x.squeeze(1)
    # Accept [1,H,W] -> add batch dim
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x

@torch.no_grad()
def denoise_tiled(model: nn.Module, image: torch.Tensor, tile: int = 256, overlap: int = 32) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    x = _ensure_nchw(image).to(device); _, _, H, W = x.shape
    out = torch.zeros_like(x); weight = torch.zeros_like(x)
    step = tile - overlap; win = hann2d(tile, tile).to(device).view(1,1,tile,tile)
    for top in range(0, H, step):
        for left in range(0, W, step):
            bottom = min(top + tile, H); right  = min(left + tile, W)
            pad_h = tile - (bottom - top); pad_w = tile - (right - left)
            patch = torch.zeros((1,1,tile,tile), device=device)
            patch[:,:,:tile-pad_h,:tile-pad_w] = x[:,:,top:bottom,left:right]
            pred = model(patch) * win
            out[:,:,top:bottom,left:right] += pred[:,:,:tile-pad_h,:tile-pad_w]
            weight[:,:,top:bottom,left:right] += win[:,:,:tile-pad_h,:tile-pad_w]
    return out / (weight + 1e-6)

# ----------------------- Synthetic Noise & Augs -----------------------

def add_speckle_and_gaussian(x: torch.Tensor, sigma_g: float = 0.02, speckle: float = 0.2) -> torch.Tensor:
    if speckle > 0: x = x * torch.clamp(1.0 + speckle * torch.randn_like(x), 0.0, 3.0)
    if sigma_g > 0: x = x + sigma_g * torch.randn_like(x)
    return torch.clamp(x, 0.0, 1.0)

def augment_espi(x: torch.Tensor) -> torch.Tensor:
    # Works for shapes [..., H, W] (3D or 4D): use last two dims as H,W
    Hdim, Wdim = -2, -1

    if torch.rand(1).item() < 0.5:
        dy = int(torch.randint(-4, 5, (1,)).item())
        dx = int(torch.randint(-4, 5, (1,)).item())
        x = torch.roll(x, shifts=(dy, dx), dims=(Hdim, Wdim))

    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[Wdim])  # horizontal flip

    if torch.rand(1).item() < 0.2:
        x = torch.flip(x, dims=[Hdim])  # vertical flip

    if torch.rand(1).item() < 0.7:
        scale = 0.9 + 0.2 * torch.rand(1).item()
        bias  = -0.05 + 0.1 * torch.rand(1).item()
        x = torch.clamp(x * scale + bias, 0.0, 1.0)

    if torch.rand(1).item() < 0.7:
        gamma = 0.9 + 0.2 * torch.rand(1).item()
        x = torch.clamp(x ** gamma, 0.0, 1.0)

    return x

# ----------------------- Splits & Datasets -----------------------

@dataclass
class DataSplit:
    train: List[Path]; val: List[Path]

def build_split(paths: List[Path], mode: str, val_ratio: float, seed: int, root: Optional[Path] = None,
                lofo_group: Optional[str] = None) -> DataSplit:
    assert mode in ("random","lofo")
    rng = random.Random(seed)
    if mode == "random":
        p = paths[:]; rng.shuffle(p); n_val = max(1, int(len(p)*val_ratio))
        return DataSplit(train=p[n_val:], val=p[:n_val])
    else:
        assert root is not None, "LOFO requires clean-root"
        groups: Dict[str, List[Path]] = {}
        for path in paths:
            rel = path.relative_to(root)
            g = rel.parts[0] if len(rel.parts)>1 else rel.parent.name
            groups.setdefault(g, []).append(path)
        keys = sorted(groups.keys())
        if lofo_group is None: lofo_group = keys[-1]
        val = groups[lofo_group]; train = []
        for k,v in groups.items():
            if k!=lofo_group: train+=v
        return DataSplit(train=train, val=val)

class TrainTileDataset(Dataset):
    def __init__(self, paths: List[Path], tile: int = 512, sigma_g: float = 0.02, speckle: float = 0.2, do_aug: bool = True):
        self.paths = paths; self.tile = tile; self.sigma=sigma_g; self.speckle=speckle; self.do_aug=do_aug
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx: int):
        arr = imread_uint(self.paths[idx]); h,w = arr.shape; th,tw = self.tile,self.tile
        if h<th or w<tw:
            arr = np.pad(arr, ((0,max(0,th-h)),(0,max(0,tw-w))), mode="reflect"); h,w = arr.shape
        top = np.random.randint(0,h-th+1); left = np.random.randint(0,w-tw+1)
        crop = arr[top:top+th, left:left+tw]
        x = torch.from_numpy(crop).unsqueeze(0).float()
        if self.do_aug: x = augment_espi(x)
        clean = x.clone(); noisy = add_speckle_and_gaussian(clean, self.sigma, self.speckle)
        return noisy, clean

class ValFullResSyntheticDataset(Dataset):
    def __init__(self, paths: List[Path], sigma_g: float = 0.02, speckle: float = 0.2):
        self.paths = paths; self.sigma=sigma_g; self.speckle=speckle
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx: int):
        arr = imread_uint(self.paths[idx])
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        clean = x.clone(); noisy = add_speckle_and_gaussian(clean, self.sigma, self.speckle)
        return noisy, clean

class RealPairDataset(Dataset):
    def __init__(self, clean_root: Path, noisy_root: Path):
        clean_paths = list_pngs(clean_root); pairs=[]
        for cp in clean_paths:
            rel = cp.relative_to(clean_root); npth = noisy_root / rel
            if npth.exists(): pairs.append((npth, cp))
        if not pairs: raise FileNotFoundError("No REAL pairs found (match folder structures).")
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx: int):
        noisy_p, clean_p = self.pairs[idx]
        n_arr = imread_uint(noisy_p); c_arr = imread_uint(clean_p)
        H = min(n_arr.shape[0], c_arr.shape[0]); W = min(n_arr.shape[1], c_arr.shape[1])
        n_arr = n_arr[:H,:W]; c_arr = c_arr[:H,:W]
        noisy = torch.from_numpy(n_arr).unsqueeze(0).unsqueeze(0).float()
        clean = torch.from_numpy(c_arr).unsqueeze(0).unsqueeze(0).float()
        return noisy, clean, str(noisy_p.relative_to(noisy_p.parents[1]))

# ----------------------- AMP Helpers (CPU-safe) -----------------------

def get_amp_objects(device: torch.device):
    use_cuda = (device.type == "cuda")
    if AMP_NEW:
        scaler = AmpGradScaler("cuda", enabled=use_cuda)
        if use_cuda:
            @contextlib.contextmanager
            def ctx():
                with amp_autocast("cuda"):
                    yield
        else:
            @contextlib.contextmanager
            def ctx():
                yield
        return scaler, ctx
    else:
        scaler = CudaGradScaler(enabled=use_cuda)
        if use_cuda:
            return scaler, cuda_autocast
        else:
            @contextlib.contextmanager
            def null_autocast():
                yield
            return scaler, null_autocast

# ----------------------- Training / Validation / REAL eval -----------------------

@dataclass
class Args:
    clean_root: str
    output_dir: str
    split_mode: str
    val_ratio: float
    lofo_group: Optional[str]
    real_noisy_root: Optional[str]
    tile: int
    overlap: int
    batch_size: int
    workers: int
    epochs: int
    lr: float
    features: int
    depth: int
    norm: str
    eca_interval: int
    sigma_g: float
    speckle: float
    seed: int
    patience: int
    device: str
    tensorboard: bool
    export_onnx: Optional[str]
    resume: Optional[str]

def build_loaders(a: Args, device: torch.device):
    root = Path(a.clean_root); paths = list_pngs(root)
    if not paths: raise FileNotFoundError(f"No PNGs under {root}")
    split = build_split(paths, mode=a.split_mode, val_ratio=a.val_ratio, seed=a.seed, root=root, lofo_group=a.lofo_group)
    pin = (device.type == "cuda")
    train_dl = DataLoader(TrainTileDataset(split.train, a.tile, a.sigma_g, a.speckle, True),
                          batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                          pin_memory=pin, drop_last=True)
    val_dl = DataLoader(ValFullResSyntheticDataset(split.val, a.sigma_g, a.speckle),
                        batch_size=1, shuffle=False, num_workers=0, pin_memory=pin)
    return train_dl, val_dl

def save_ckpt(state: Dict, ckpt_dir: Path, is_best: bool):
    ensure_dir(ckpt_dir); torch.save(state, ckpt_dir/"last.pth")
    if is_best: torch.save(state, ckpt_dir/"best.pth")

def run_epoch_train(model, dl, criterion, opt, scaler, autocast_ctx, device):
    model.train(); total=0.0
    for noisy, clean in dl:
        noisy=noisy.to(device, non_blocking=True); clean=clean.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            den = model(noisy); loss = criterion(den, clean)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        total += loss.item()
    return total / max(1,len(dl))

@torch.no_grad()
def run_validation_fullres(model, dl, criterion, device, tile, overlap, autocast_ctx):
    model.eval(); vloss=0.0; ps_list=[]; ss_list=[]; fe_list=[]
    for noisy, clean in dl:
        noisy=noisy.to(device); clean=clean.to(device)
        noisy = _ensure_nchw(noisy); clean = _ensure_nchw(clean)
        with autocast_ctx():
            den = denoise_tiled(model, noisy, tile=tile, overlap=overlap)
            loss = criterion(den, clean)
        vloss += loss.item(); ps_list.append(psnr(den,clean)); ss_list.append(ssim_metric(den,clean)); fe_list.append(fringe_edge_f1(den,clean))
    n=max(1,len(dl))
    return vloss/n, float(np.mean(ps_list)), float(np.mean(ss_list)), float(np.mean(fe_list))

@torch.no_grad()
def run_real_evaluation(model, clean_root: Optional[Path], noisy_root: Optional[Path], device, tile, overlap, out_csv: Path):
    if clean_root is None or noisy_root is None or not noisy_root.exists():
        return None
    try:
        ds = RealPairDataset(clean_root, noisy_root)
    except Exception as e:
        print(f"[REAL] Skipping: {e}"); return None
    rows=[("rel_path","PSNR","SSIM","EdgeF1")]; ps_list=[]; ss_list=[]; fe_list=[]
    for i in range(len(ds)):
        noisy, clean, rel = ds[i]
        noisy=noisy.to(device); clean=clean.to(device)
        den = denoise_tiled(model, noisy, tile=tile, overlap=overlap)
        ps=psnr(den,clean); ss=ssim_metric(den,clean); fe=fringe_edge_f1(den,clean)
        rows.append((rel, f"{ps:.4f}", f"{ss:.6f}", f"{fe:.6f}"))
        ps_list.append(ps); ss_list.append(ss); fe_list.append(fe)
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return {"real_psnr": float(np.mean(ps_list)), "real_ssim": float(np.mean(ss_list)), "real_edgef1": float(np.mean(fe_list)), "count": len(ds)}

# ----------------------- Main -----------------------

def main(a: Args):
    set_seed(a.seed, deterministic=True)
    device = torch.device(a.device if (a.device=="cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(a.output_dir); ckpt_dir = out_dir/"checkpoints"; log_csv = out_dir/"train_log.csv"
    ensure_dir(out_dir)

    # Data
    train_dl, val_dl = build_loaders(a, device)

    # Model/Optim
    model = DnCNNLiteECA(DnCNNLiteECAConfig(1,1,a.features,a.depth,a.norm,a.eca_interval,3,1.0)).to(device)
    criterion = EdgeAwareLoss(0.8,0.2,0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    # Resume support
    start_epoch = 0
    if a.resume:
        ckpt_path = Path(a.resume) if a.resume != "auto" else (ckpt_dir / "last.pth")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            try:
                model.load_state_dict(ckpt["model"])
                opt.load_state_dict(ckpt["optimizer"])
                sched.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[RESUME] Warning while loading states: {e}")
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[RESUME] Loaded '{ckpt_path}' at epoch {start_epoch}.")
        else:
            print(f"[RESUME] Requested but checkpoint not found: {ckpt_path}")

    # AMP objects (CPU-safe or CUDA)
    scaler, autocast_ctx = get_amp_objects(device)

    # TensorBoard
    writer = None
    if a.tensorboard and TB_AVAILABLE:
        tb_dir = out_dir/"tb"; ensure_dir(tb_dir)
        writer = SummaryWriter(str(tb_dir))

    # CSV header (append if exists and resuming)
    write_header = not Path(log_csv).exists() or not a.resume
    if write_header:
        with open(log_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_psnr,val_ssim,val_edgeF1,lr\n")

    best_val=float("inf"); best_epoch=start_epoch
    for epoch in range(start_epoch + 1, start_epoch + a.epochs + 1):
        tr = run_epoch_train(model, train_dl, criterion, opt, scaler, autocast_ctx, device)
        vl, vp, vs, vf = run_validation_fullres(model, val_dl, criterion, device, a.tile, a.overlap, autocast_ctx)
        sched.step(); lr_now = sched.get_last_lr()[0]

        print(f"Epoch {epoch:03d}/{start_epoch + a.epochs} | Train {tr:.4f} | Val {vl:.4f} | PSNR {vp:.2f} | SSIM {vs:.4f} | EdgeF1 {vf:.4f} | LR {lr_now:.2e}")
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr:.6f},{vl:.6f},{vp:.4f},{vs:.6f},{vf:.6f},{lr_now:.8f}\n")

        if writer is not None:
            writer.add_scalar("loss/train", tr, epoch)
            writer.add_scalar("loss/val", vl, epoch)
            writer.add_scalar("val/psnr", vp, epoch)
            writer.add_scalar("val/ssim", vs, epoch)
            writer.add_scalar("val/edgeF1", vf, epoch)
            writer.add_scalar("opt/lr", lr_now, epoch)

        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "args": vars(a)}
        is_best = vl < best_val
        save_ckpt(state, ckpt_dir, is_best)
        if is_best: best_val=vl; best_epoch=epoch
        if (epoch - best_epoch) >= a.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {a.patience} epochs).")
            break

    print(f"Best val loss so far: {best_val:.4f} (best epoch {best_epoch})")

    # REAL evaluation (optional)
    if a.real_noisy_root:
        stats = run_real_evaluation(model, Path(a.clean_root), Path(a.real_noisy_root), device, a.tile, a.overlap, out_dir/"real_eval_final.csv")
        print(f"[REAL] {stats}")

    # Export ONNX (optional)
    if a.export_onnx:
        best_path = ckpt_dir/"best.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["model"]); model.eval().cpu()
        dummy = torch.randn(1,1,512,512)
        torch.onnx.export(
            model, dummy, a.export_onnx,
            input_names=["input"], output_names=["output"],
            opset_version=17,
            dynamic_axes={"input": {2:"H", 3:"W"}, "output": {2:"H", 3:"W"}}
        )
        print(f"Exported ONNX to {a.export_onnx}")

    if writer is not None: writer.close()

# ----------------------- CLI -----------------------

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-root", type=str, required=True)
    p.add_argument("--real-noisy-root", type=str, default="")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--split-mode", type=str, default="random", choices=["random","lofo"])
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--lofo-group", type=str, default=None)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)  # safe for CPU
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--depth", type=int, default=17)
    p.add_argument("--norm", type=str, default="group", choices=["none","batch","group"])
    p.add_argument("--eca-interval", type=int, default=4)
    p.add_argument("--sigma-g", type=float, default=0.02)
    p.add_argument("--speckle", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--export-onnx", type=str, default="")
    p.add_argument("--resume", type=str, default="", help="path to ckpt or 'auto' to load outputs/checkpoints/last.pth")
    a = p.parse_args()
    return Args(
        clean_root=a.clean_root, output_dir=a.output_dir, split_mode=a.split_mode, val_ratio=a.val_ratio,
        lofo_group=a.lofo_group, real_noisy_root=a.real_noisy_root if a.real_noisy_root else None,
        tile=a.tile, overlap=a.overlap, batch_size=a.batch_size, workers=a.workers, epochs=a.epochs, lr=a.lr,
        features=a.features, depth=a.depth, norm=a.norm, eca_interval=a.eca_interval,
        sigma_g=a.sigma_g, speckle=a.speckle, seed=a.seed, patience=a.patience, device=a.device,
        tensorboard=bool(a.tensorboard), export_onnx=(a.export_onnx if a.export_onnx else None),
        resume=(a.resume if a.resume else None)
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
