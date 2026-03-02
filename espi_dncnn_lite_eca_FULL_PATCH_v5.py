#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESPI Denoising β€” DnCNN-Lite + ECA (+ optional Spatial Lite Attention) β€” FULL PATCH V5
CPU-safe single-file trainer/inferencer with:
- Resume (--resume auto|path)
- Tile-based full-res validation/inference (Hann blending)
- REAL evaluation (single-shot noisy vs averaged pseudo-clean) β€” periodic + final
- TensorBoard logging (--tensorboard) incl. ECA gate diagnostics
- ONNX export (--export-onnx)
- Best-by-loss and Best-by-SSIM checkpoints
- Optional freeze of Norm layers after warm-up
- ECA shallow placement via --eca-positions, adaptive kernel, temp/gain
- Optional Burr-XII speckle for more realistic synthetic noise
Tested on CPU (Ryzen 5600G/5700G) and CUDA if available.
"""
from __future__ import annotations

import argparse, contextlib, csv, math, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------- Optional AMP (auto-disabled on CPU) ----------
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

# ----------------------- Utils & I/O -----------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_pngs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.png"))

def imread_uint(path: Path) -> np.ndarray:
    """Read 8/16-bit PNG as float32 [0,1]."""
    from PIL import Image
    img = Image.open(path)
    if img.mode in ("I;16","I;16B","I"):
        arr = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(img.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

# ----------------------- Model -----------------------

def kaiming_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None: nn.init.zeros_(m.bias)

def eca_kernel_for(C:int, gamma:float=2.0, b:float=1.0)->int:
    """Adaptive ECA kernel size rule-of-thumb; returns odd >=3."""
    val = int(round(math.log2(max(C,1))/gamma + b/gamma))
    k = max(3, val | 1)  # ensure odd
    return k

class ECA(nn.Module):
    """ECA v5: dual pooling, optional multi-scale kernels, optional learnable temp/gain."""
    def __init__(
        self,
        channels: int,
        k_size: int = 0,
        temp: float = 0.75,
        gain: float = 0.5,
        centered: bool = True,
        use_maxpool: bool = True,
        multi_scale: bool = False,
        learnable_temp_gain: bool = False,
        temp_min: float = 1e-6,
        gain_min: float = 0.01,
        gain_max: float = 1.5,
    ):
        super().__init__()
        k = k_size if (k_size and (k_size % 2 == 1)) else eca_kernel_for(channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.use_maxpool = bool(use_maxpool)
        self.multi_scale = bool(multi_scale)
        self.centered = centered
        self.learnable_temp_gain = bool(learnable_temp_gain)
        self.temp_min = float(temp_min)
        self.gain_min = float(gain_min)
        self.gain_max = float(max(gain_max, gain_min + 1e-6))
        self._last_gate_mean: Optional[float] = None

        if self.multi_scale:
            self.conv_ms = nn.ModuleList([
                nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False),
                nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False),
            ])
            self.conv_avg = None
            self.conv_max = None
        else:
            self.conv_avg = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
            self.conv_max = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False) if self.use_maxpool else None
            self.conv_ms = None
        self.sigmoid = nn.Sigmoid()

        if self.learnable_temp_gain:
            init_temp = float(max(temp, self.temp_min))
            self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp, dtype=torch.float32)))
            frac = (float(gain) - self.gain_min) / (self.gain_max - self.gain_min)
            frac = float(min(1.0 - 1e-4, max(1e-4, frac)))
            raw = math.log(frac / (1.0 - frac))
            self.raw_gain = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
            self.temp_const = None
            self.gain_const = None
        else:
            self.register_buffer("temp_const", torch.tensor(float(max(temp, self.temp_min)), dtype=torch.float32))
            self.register_buffer("gain_const", torch.tensor(float(gain), dtype=torch.float32))
            self.log_temp = None
            self.raw_gain = None

    def current_temp(self) -> torch.Tensor:
        if self.learnable_temp_gain:
            return F.softplus(self.log_temp) + self.temp_min
        return self.temp_const

    def current_gain(self) -> torch.Tensor:
        if self.learnable_temp_gain:
            return self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(self.raw_gain)
        return self.gain_const

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_avg = self.gap(x).squeeze(-1).transpose(1, 2)  # [B,1,C]

        if self.multi_scale:
            y_base = y_avg
            if self.use_maxpool:
                y_max = self.gmp(x).squeeze(-1).transpose(1, 2)
                y_base = 0.5 * (y_avg + y_max)
            y = torch.stack([conv(y_base) for conv in self.conv_ms], dim=0).mean(dim=0)
        else:
            y = self.conv_avg(y_avg)
            if self.use_maxpool and (self.conv_max is not None):
                y_max = self.gmp(x).squeeze(-1).transpose(1, 2)
                y = 0.5 * (y + self.conv_max(y_max))

        temp = self.current_temp()
        g = self.sigmoid(y / torch.clamp(temp, min=self.temp_min))
        self._last_gate_mean = g.mean().item()
        g = g.transpose(1, 2).unsqueeze(-1)  # [B,C,1,1]

        if self.centered:
            gain = torch.clamp(self.current_gain(), min=self.gain_min, max=self.gain_max)
            scale = 1.0 + gain * (g - 0.5) * 2.0
        else:
            scale = g
        return x * scale

class SpatialLiteAttention(nn.Module):
    """Very light spatial attention: conv over [avg,max] maps."""
    def __init__(self, k: int = 5):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mxx, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.cat([avg, mxx], dim=1)   # [B,2,H,W]
        a = self.sigmoid(self.conv(m))     # [B,1,H,W]
        return x * a

def make_norm(kind: str, ch: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "batch": return nn.BatchNorm2d(ch)
    if kind == "group":
        for g in [8,4,2,1]:
            if ch % g == 0: return nn.GroupNorm(g, ch)
        return nn.GroupNorm(1, ch)
    return nn.Identity()

def make_norm_with_groups(kind: str, ch: int, gn_groups: int = 0) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        if gn_groups and gn_groups > 0:
            g = max(1, min(int(gn_groups), int(ch)))
            while g > 1 and (ch % g) != 0:
                g -= 1
            return nn.GroupNorm(g, ch)
        return make_norm(kind, ch)
    return nn.Identity()

@dataclass
class DnCNNLiteECAConfig:
    in_channels: int = 1
    out_channels: int = 1
    features: int = 32
    depth: int = 17
    norm: str = "group"
    gn_groups: int = 0               # 0=auto [8,4,2,1], >0 force max group count
    use_eca: bool = True
    # --- ECA controls
    eca_interval: int = 4              # used if eca_positions is None
    eca_k: int = 3                     # 0 => auto via eca_kernel_for
    eca_temp: float = 0.75
    eca_gain: float = 0.5
    eca_centered: bool = True
    eca_use_maxpool: bool = True
    eca_multi_scale: bool = False
    eca_learnable_temp_gain: bool = False
    eca_temp_min: float = 1e-6
    eca_gain_min: float = 0.01
    eca_gain_max: float = 1.5
    eca_order: str = "post"            # post | pre
    eca_preset: str = "none"           # none | shallow3 | dense_shallow
    eca_positions: Optional[List[int]] = None  # explicit shallow placements, e.g., [0,1,2]
    residual_weight: float = 1.0
    # --- Spatial lite attention
    spa_interval: int = 0              # 0 = disabled
    spa_k: int = 5

def resolve_eca_positions(depth: int, preset: str, manual: Optional[List[int]]) -> Optional[List[int]]:
    max_block = max(0, int(depth) - 3)
    if manual is not None and len(manual) > 0:
        return sorted([i for i in manual if 0 <= i <= max_block])
    preset = (preset or "none").lower()
    if preset == "shallow3":
        raw = [0, 1, 2]
    elif preset == "dense_shallow":
        raw = [0, 1, 2, 3, 6, 10, 14]
    else:
        return None
    return sorted([i for i in raw if 0 <= i <= max_block])

class ConvBlock(nn.Module):
    def __init__(self, ch: int, norm: str = "group", gn_groups: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm = make_norm_with_groups(norm, ch, gn_groups=gn_groups)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.norm(self.conv(x)))

class DnCNNLiteECA(nn.Module):
    def __init__(self, cfg: DnCNNLiteECAConfig):
        super().__init__()
        c = cfg.features
        self.cfg = cfg
        self.head = nn.Sequential(nn.Conv2d(cfg.in_channels, c, 3, padding=1, bias=False),
                                  nn.ReLU(inplace=True))
        blocks, ecas, spas = [], [], []
        resolved_positions = resolve_eca_positions(cfg.depth, cfg.eca_preset, cfg.eca_positions)
        pos_set = set(resolved_positions) if resolved_positions is not None else None
        for i in range(cfg.depth - 2):
            blocks.append(ConvBlock(c, cfg.norm, cfg.gn_groups))
            use_eca = False
            if cfg.use_eca:
                use_eca = (i in pos_set) if (pos_set is not None) else ((i+1) % cfg.eca_interval == 0)
            if use_eca:
                ecas.append(
                    ECA(
                        c,
                        k_size=cfg.eca_k,
                        temp=cfg.eca_temp,
                        gain=cfg.eca_gain,
                        centered=cfg.eca_centered,
                        use_maxpool=cfg.eca_use_maxpool,
                        multi_scale=cfg.eca_multi_scale,
                        learnable_temp_gain=cfg.eca_learnable_temp_gain,
                        temp_min=cfg.eca_temp_min,
                        gain_min=cfg.eca_gain_min,
                        gain_max=cfg.eca_gain_max,
                    )
                )
            else:
                ecas.append(nn.Identity())
            if cfg.spa_interval and ((i+1) % cfg.spa_interval == 0):
                spas.append(SpatialLiteAttention(cfg.spa_k))
            else:
                spas.append(nn.Identity())
        self.blocks = nn.ModuleList(blocks)
        self.eca = nn.ModuleList(ecas)
        self.spa = nn.ModuleList(spas)
        self.tail = nn.Conv2d(c, cfg.out_channels, 3, padding=1, bias=False)
        self.apply(kaiming_init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x; h = self.head(x)
        if (self.cfg.eca_order or "post").lower() == "pre":
            for blk, att_c, att_s in zip(self.blocks, self.eca, self.spa):
                h = att_c(h); h = blk(h); h = att_s(h)
        else:
            for blk, att_c, att_s in zip(self.blocks, self.eca, self.spa):
                h = blk(h); h = att_c(h); h = att_s(h)
        noise = self.tail(h)
        return identity - self.cfg.residual_weight * noise

# ----------------------- Losses & Metrics -----------------------

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
        ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3)
        self.register_buffer("kx", kx); self.register_buffer("ky", ky)
    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1); gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-12)

def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    def gaussian_window(size, sigma=1.5):
        g = torch.tensor([math.exp(-(i-size//2)**2/(2*sigma**2)) for i in range(size)])
        g = g / g.sum()
        return (g.unsqueeze(1) @ g.unsqueeze(0)).float()
    C1 = 0.01**2; C2 = 0.03**2
    b,c,h,w = x.shape; device=x.device
    wdw = gaussian_window(window_size).to(device).view(1,1,window_size,window_size)
    mu_x = F.conv2d(x, wdw, padding=window_size//2, groups=c)
    mu_y = F.conv2d(y, wdw, padding=window_size//2, groups=c)
    sigma_x = F.conv2d(x*x, wdw, padding=window_size//2, groups=c) - mu_x*mu_x
    sigma_y = F.conv2d(y*y, wdw, padding=window_size//2, groups=c) - mu_y*mu_y
    sigma_xy= F.conv2d(x*y, wdw, padding=window_size//2, groups=c) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x*mu_x + mu_y*mu_y + C1)*(sigma_x + sigma_y + C2) + 1e-12)
    return 1.0 - ssim_map.mean()

class EdgeAwareLoss(nn.Module):
    def __init__(self, w_l1: float = 0.8, w_ssim: float = 0.2, w_edge: float = 0.1):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_edge = w_l1, w_ssim, w_edge
        self.sobel = Sobel()
    def forward(self, pred, tgt):
        l1 = F.l1_loss(pred, tgt); ss = ssim_loss(pred, tgt)
        e = F.l1_loss(self.sobel(pred), self.sobel(tgt))
        return self.w_l1*l1 + self.w_ssim*ss + self.w_edge*e

def psnr(x, y, eps: float = 1e-12) -> float:
    mse = F.mse_loss(x, y).item()
    return 99.0 if mse <= 0 else 20.0 * math.log10(1.0 / math.sqrt(mse + eps))

def ssim_metric(x, y) -> float: return (1.0 - ssim_loss(x, y)).item()

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
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1s.append(2*prec*rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))

# ---------------- Tiled inference (Hann) ----------------

def hann2d(h: int, w: int, eps: float = 1e-6) -> torch.Tensor:
    wx = torch.hann_window(w).unsqueeze(0); wy = torch.hann_window(h).unsqueeze(1)
    return (wy @ wx) + eps

def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5 and x.size(1) == 1: x = x.squeeze(1)
    if x.dim() == 3: x = x.unsqueeze(0)
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
            bottom = min(top + tile, H); right = min(left + tile, W)
            pad_h = tile - (bottom - top); pad_w = tile - (right - left)
            patch = torch.zeros((1,1,tile,tile), device=device)
            patch[:,:,:tile-pad_h,:tile-pad_w] = x[:,:,top:bottom,left:right]
            pred = model(patch) * win
            out[:,:,top:bottom,left:right] += pred[:,:,:tile-pad_h,:tile-pad_w]
            weight[:,:,top:bottom,left:right] += win[:,:,:tile-pad_h,:tile-pad_w]
    return out / (weight + 1e-6)

# ---------------- Synthetic Noise & Augmentations ----------------

def add_burr_speckle(x, c=4.687, k=1189.53, strength=0.25):
    """Multiplicative Burr-XII speckle; meanβ‰1 scaling."""
    u = torch.rand_like(x).clamp(1e-6, 1-1e-6)
    m = ((1 - u)**(-1.0/k) - 1.0).clamp(min=0) ** (1.0/c)
    m = m / (m.mean() + 1e-9)
    return torch.clamp(x * (1.0 + strength*(m-1.0)), 0.0, 3.0)

def add_speckle_and_gaussian(x: torch.Tensor, sigma_g: float = 0.02, speckle: float = 0.2, kind: str = "gauss") -> torch.Tensor:
    kind = (kind or "gauss").lower()
    if speckle > 0:
        if kind == "burr":
            x = add_burr_speckle(x, strength=speckle)
        else:
            x = x * torch.clamp(1.0 + speckle * torch.randn_like(x), 0.0, 3.0)
    if sigma_g > 0: x = x + sigma_g * torch.randn_like(x)
    return torch.clamp(x, 0.0, 1.0)

def augment_espi(x: torch.Tensor) -> torch.Tensor:
    Hdim, Wdim = -2, -1
    if torch.rand(1).item() < 0.5:
        dy = int(torch.randint(-4,5,(1,)).item()); dx = int(torch.randint(-4,5,(1,)).item())
        x = torch.roll(x, shifts=(dy,dx), dims=(Hdim,Wdim))
    if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[Wdim])
    if torch.rand(1).item() < 0.2: x = torch.flip(x, dims=[Hdim])
    if torch.rand(1).item() < 0.7:
        scale = 0.9 + 0.2*torch.rand(1).item(); bias = -0.05 + 0.1*torch.rand(1).item()
        x = torch.clamp(x*scale + bias, 0.0, 1.0)
    if torch.rand(1).item() < 0.7:
        gamma = 0.9 + 0.2*torch.rand(1).item(); x = torch.clamp(x**gamma, 0.0, 1.0)
    return x

# ---------------- Splits & Datasets ----------------

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
    def __init__(self, paths: List[Path], tile: int = 512, sigma_g: float = 0.02, speckle: float = 0.2, do_aug: bool = True, speckle_kind: str = "gauss"):
        self.paths = paths; self.tile = tile; self.sigma=sigma_g; self.speckle=speckle; self.do_aug=do_aug; self.speckle_kind=speckle_kind
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx: int):
        arr = imread_uint(self.paths[idx]); h,w = arr.shape; th,tw = self.tile,self.tile
        if h<th or w<tw:
            arr = np.pad(arr, ((0,max(0,th-h)),(0,max(0,tw-w))), mode="reflect"); h,w = arr.shape
        top = np.random.randint(0,h-th+1); left = np.random.randint(0,w-tw+1)
        crop = arr[top:top+th, left:left+tw]
        x = torch.from_numpy(crop).unsqueeze(0).float()
        if self.do_aug: x = augment_espi(x)
        clean = x.clone(); noisy = add_speckle_and_gaussian(clean, self.sigma, self.speckle, kind=self.speckle_kind)
        return noisy, clean

class ValFullResSyntheticDataset(Dataset):
    def __init__(self, paths: List[Path], sigma_g: float = 0.02, speckle: float = 0.2, speckle_kind: str = "gauss"):
        self.paths = paths; self.sigma=sigma_g; self.speckle=speckle; self.speckle_kind=speckle_kind
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx: int):
        arr = imread_uint(self.paths[idx])
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        clean = x.clone(); noisy = add_speckle_and_gaussian(clean, self.sigma, self.speckle, kind=self.speckle_kind)
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

# ---------------- AMP helpers ----------------

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
        if use_cuda: return scaler, cuda_autocast
        @contextlib.contextmanager
        def nullctx(): yield
        return scaler, nullctx

# ---------------- Train / Val / REAL eval ----------------

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
    weight_decay: float
    features: int
    depth: int
    norm: str
    gn_groups: int
    use_eca: bool
    eca_interval: int
    eca_k: int
    eca_temp: float
    eca_gain: float
    eca_centered: bool
    eca_use_maxpool: bool
    eca_multi_scale: bool
    eca_learnable_temp_gain: bool
    eca_temp_min: float
    eca_gain_min: float
    eca_gain_max: float
    eca_order: str
    eca_preset: str
    eca_positions: Optional[List[int]]
    spa_interval: int
    spa_ks: int
    sigma_g: float
    speckle: float
    speckle_kind: str
    seed: int
    patience: int
    device: str
    tensorboard: bool
    export_onnx: Optional[str]
    resume: Optional[str]
    resume_strict: bool
    w_edge: float
    eca_param_lr_scale: float
    eca_param_weight_decay: float
    separate_eca_param_optim: bool
    grad_clip: float
    max_nonfinite_batches: int
    nan_action: str
    log_grad_norm: bool
    freeze_norm_epoch: int
    real_eval_every: int

def build_loaders(a: Args, device: torch.device):
    root = Path(a.clean_root); paths = list_pngs(root)
    if not paths: raise FileNotFoundError(f"No PNGs under {root}")
    split = build_split(paths, mode=a.split_mode, val_ratio=a.val_ratio, seed=a.seed, root=root, lofo_group=a.lofo_group)
    pin = (device.type == "cuda")
    train_dl = DataLoader(TrainTileDataset(split.train, a.tile, a.sigma_g, a.speckle, True, a.speckle_kind),
                          batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                          pin_memory=pin, drop_last=True)
    val_dl = DataLoader(ValFullResSyntheticDataset(split.val, a.sigma_g, a.speckle, a.speckle_kind),
                        batch_size=1, shuffle=False, num_workers=0, pin_memory=pin)
    return train_dl, val_dl

def save_ckpt(state: Dict, ckpt_dir: Path, is_best: bool, tag: str = "best"):
    ensure_dir(ckpt_dir); torch.save(state, ckpt_dir/"last.pth")
    if is_best: torch.save(state, ckpt_dir/f"{tag}.pth")

def freeze_norm_layers(m: nn.Module):
    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        for p in m.parameters(): p.requires_grad = False

def eca_gate_means(model: nn.Module) -> List[float]:
    means = []
    for m in getattr(model, "eca", []):
        if isinstance(m, ECA) and (m._last_gate_mean is not None):
            means.append(float(m._last_gate_mean))
    return means

def eca_temp_gain_means(model: nn.Module) -> Tuple[float, float]:
    temps: List[float] = []
    gains: List[float] = []
    for m in getattr(model, "eca", []):
        if isinstance(m, ECA):
            try:
                t = float(m.current_temp().detach().cpu().item())
                g = float(m.current_gain().detach().cpu().item())
                if math.isfinite(t):
                    temps.append(t)
                if math.isfinite(g):
                    gains.append(g)
            except Exception:
                continue
    t_mean = float(np.mean(temps)) if temps else float("nan")
    g_mean = float(np.mean(gains)) if gains else float("nan")
    return t_mean, g_mean

def _is_finite_tensor(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())

def _global_grad_norm(params) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not _is_finite_tensor(g):
            return float("nan")
        n = g.norm(2).item()
        total_sq += n * n
    return math.sqrt(total_sq)

def _stop_on_nonfinite(nan_action: str, bad_count: int, max_bad: int) -> bool:
    return (nan_action == "stop") and (bad_count > max_bad)

def run_epoch_train(model, dl, criterion, opt, scaler, autocast_ctx, device,
                    grad_clip: float = 1.0, max_nonfinite_batches: int = 3,
                    nan_action: str = "stop", log_grad_norm: bool = False):
    model.train()
    total = 0.0
    used = 0
    seen = 0
    bad = 0
    grad_sum = 0.0

    for noisy, clean in dl:
        seen += 1
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        if (not _is_finite_tensor(noisy)) or (not _is_finite_tensor(clean)):
            bad += 1
            if _stop_on_nonfinite(nan_action, bad, max_nonfinite_batches):
                break
            continue

        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            den = model(noisy)
            loss = criterion(den, clean)

        if not _is_finite_tensor(loss):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if _stop_on_nonfinite(nan_action, bad, max_nonfinite_batches):
                break
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        else:
            grad_norm = float(_global_grad_norm(model.parameters()))

        if not math.isfinite(grad_norm):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if _stop_on_nonfinite(nan_action, bad, max_nonfinite_batches):
                break
            continue

        scaler.step(opt)
        scaler.update()
        total += float(loss.item())
        used += 1
        if log_grad_norm:
            grad_sum += grad_norm

    return {
        "loss": (total / used) if used > 0 else float("nan"),
        "nonfinite_batches": bad,
        "batches_seen": seen,
        "batches_used": used,
        "grad_norm": (grad_sum / used) if (log_grad_norm and used > 0) else float("nan"),
    }

@torch.no_grad()
def run_validation_fullres(model, dl, criterion, device, tile, overlap, autocast_ctx,
                           max_nonfinite_batches: int = 3, nan_action: str = "stop"):
    model.eval()
    vloss = 0.0
    ps_list, ss_list, fe_list = [], [], []
    used = 0
    seen = 0
    bad = 0

    for noisy, clean in dl:
        seen += 1
        noisy = noisy.to(device)
        clean = clean.to(device)
        noisy = _ensure_nchw(noisy)
        clean = _ensure_nchw(clean)

        if (not _is_finite_tensor(noisy)) or (not _is_finite_tensor(clean)):
            bad += 1
            if _stop_on_nonfinite(nan_action, bad, max_nonfinite_batches):
                break
            continue

        with autocast_ctx():
            den = denoise_tiled(model, noisy, tile=tile, overlap=overlap)
            den = torch.clamp(den, 0.0, 1.0)  # clamp only for metrics
            loss = criterion(den, clean)

        if not _is_finite_tensor(loss):
            bad += 1
            if _stop_on_nonfinite(nan_action, bad, max_nonfinite_batches):
                break
            continue

        pv = psnr(den, clean)
        sv = ssim_metric(den, clean)
        fv = fringe_edge_f1(den, clean)
        if not (math.isfinite(pv) and math.isfinite(sv) and math.isfinite(fv)):
            bad += 1
            if _stop_on_nonfinite(nan_action, bad, max_nonfinite_batches):
                break
            continue

        vloss += float(loss.item())
        ps_list.append(float(pv))
        ss_list.append(float(sv))
        fe_list.append(float(fv))
        used += 1

    if used == 0:
        return {
            "loss": float("nan"),
            "psnr": float("nan"),
            "ssim": float("nan"),
            "edgef1": float("nan"),
            "nonfinite_batches": bad,
            "samples_seen": seen,
            "samples_used": used,
        }

    return {
        "loss": vloss / used,
        "psnr": float(np.mean(ps_list)),
        "ssim": float(np.mean(ss_list)),
        "edgef1": float(np.mean(fe_list)),
        "nonfinite_batches": bad,
        "samples_seen": seen,
        "samples_used": used,
    }

@torch.no_grad()
def run_real_evaluation(model, clean_root: Optional[Path], noisy_root: Optional[Path], device, tile, overlap, out_csv: Path):
    if clean_root is None or noisy_root is None or not noisy_root.exists():
        return None
    try:
        ds = RealPairDataset(clean_root, noisy_root)
    except Exception as e:
        print(f"[REAL] Skipping: {e}"); return None
    rows=[("rel_path","PSNR","SSIM","EdgeF1")]; ps_list=[]; ss_list=[]; fe_list=[]; skipped=0
    for i in range(len(ds)):
        noisy, clean, rel = ds[i]
        noisy=noisy.to(device); clean=clean.to(device)
        if (not _is_finite_tensor(noisy)) or (not _is_finite_tensor(clean)):
            skipped += 1
            continue
        den = denoise_tiled(model, noisy, tile=tile, overlap=overlap)
        den = torch.clamp(den, 0.0, 1.0)
        ps=psnr(den,clean); ss=ssim_metric(den,clean); fe=fringe_edge_f1(den,clean)
        if not (math.isfinite(ps) and math.isfinite(ss) and math.isfinite(fe)):
            skipped += 1
            continue
        rows.append((rel, f"{ps:.4f}", f"{ss:.6f}", f"{fe:.6f}"))
        ps_list.append(ps); ss_list.append(ss); fe_list.append(fe)
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    if not ps_list:
        return {"real_psnr": float("nan"), "real_ssim": float("nan"), "real_edgef1": float("nan"), "count": len(ds), "used": 0, "skipped": skipped}
    return {
        "real_psnr": float(np.mean(ps_list)),
        "real_ssim": float(np.mean(ss_list)),
        "real_edgef1": float(np.mean(fe_list)),
        "count": len(ds),
        "used": len(ps_list),
        "skipped": skipped,
    }

# ---------------- Main ----------------

def main(a: Args):
    set_seed(a.seed, deterministic=True)
    device = torch.device(a.device if (a.device=="cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(a.output_dir); ckpt_dir = out_dir/"checkpoints"; log_csv = out_dir/"train_log.csv"
    ensure_dir(out_dir)

    # Data
    train_dl, val_dl = build_loaders(a, device)

    # Model/Optim
    cfg = DnCNNLiteECAConfig(
        in_channels=1, out_channels=1, features=a.features, depth=a.depth, norm=a.norm,
        gn_groups=a.gn_groups, use_eca=a.use_eca,
        eca_interval=a.eca_interval, eca_k=a.eca_k, eca_temp=a.eca_temp, eca_gain=a.eca_gain,
        eca_centered=a.eca_centered,
        eca_use_maxpool=a.eca_use_maxpool,
        eca_multi_scale=a.eca_multi_scale,
        eca_learnable_temp_gain=a.eca_learnable_temp_gain,
        eca_temp_min=a.eca_temp_min,
        eca_gain_min=a.eca_gain_min,
        eca_gain_max=a.eca_gain_max,
        eca_order=a.eca_order,
        eca_preset=a.eca_preset,
        eca_positions=a.eca_positions,
        residual_weight=1.0, spa_interval=a.spa_interval, spa_k=a.spa_ks
    )
    print(
        f"[CFG] use_eca={a.use_eca} norm={a.norm} gn_groups={a.gn_groups} "
        f"eca_order={a.eca_order} eca_preset={a.eca_preset} "
        f"eca_positions={a.eca_positions} eca_interval={a.eca_interval} "
        f"eca_k={a.eca_k} eca_multi_scale={a.eca_multi_scale} "
        f"eca_use_maxpool={a.eca_use_maxpool} "
        f"eca_temp={a.eca_temp} eca_gain={a.eca_gain} learnable_tg={a.eca_learnable_temp_gain} "
        f"freeze_norm_epoch={a.freeze_norm_epoch} nan_action={a.nan_action}"
    )
    model = DnCNNLiteECA(cfg).to(device)
    criterion = EdgeAwareLoss(0.8,0.2,a.w_edge).to(device)
    if a.separate_eca_param_optim:
        eca_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if ("log_temp" in name) or ("raw_gain" in name):
                eca_params.append(p)
            else:
                other_params.append(p)
        if eca_params:
            opt = torch.optim.AdamW(
                [
                    {"params": other_params, "weight_decay": a.weight_decay, "lr": a.lr},
                    {
                        "params": eca_params,
                        "weight_decay": a.eca_param_weight_decay,
                        "lr": a.lr * a.eca_param_lr_scale,
                    },
                ],
                lr=a.lr,
            )
            print(
                f"[OPT] separate ECA params: n={len(eca_params)} "
                f"lr_scale={a.eca_param_lr_scale} wd={a.eca_param_weight_decay}"
            )
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    # Resume
    start_epoch = 0
    if a.resume:
        ckpt_path = Path(a.resume) if a.resume != "auto" else (ckpt_dir / "last.pth")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            try:
                model.load_state_dict(ckpt["model"], strict=bool(a.resume_strict))
                opt.load_state_dict(ckpt["optimizer"])
                sched.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                if a.resume_strict:
                    raise RuntimeError(
                        f"[RESUME] Strict loading failed for '{ckpt_path}'. "
                        "Use --resume-nonstrict only if you intentionally changed architecture."
                    ) from e
                print(f"[RESUME] Non-strict fallback: {e}")
                model.load_state_dict(ckpt["model"], strict=False)
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[RESUME] Loaded '{ckpt_path}' at epoch {start_epoch}.")
        else:
            print(f"[RESUME] Requested but checkpoint not found: {ckpt_path}")

    # AMP
    scaler, autocast_ctx = get_amp_objects(device)

    # TensorBoard
    writer = None
    if a.tensorboard and TB_AVAILABLE:
        tb_dir = out_dir/"tb"; ensure_dir(tb_dir)
        writer = SummaryWriter(str(tb_dir))

    # CSV header
    write_header = not Path(log_csv).exists() or not a.resume
    if write_header:
        with open(log_csv, "w", encoding="utf-8") as f:
            f.write(
                "epoch,train_loss,val_loss,val_psnr,val_ssim,val_edgeF1,"
                "real_psnr,real_ssim,real_edgeF1,lr,"
                "train_nonfinite,val_nonfinite,train_batches,val_samples,grad_norm\n"
            )

    best_val=float("inf"); best_epoch=start_epoch; best_ssim=-1.0
    for epoch in range(start_epoch + 1, start_epoch + a.epochs + 1):
        train_stats = run_epoch_train(
            model, train_dl, criterion, opt, scaler, autocast_ctx, device,
            grad_clip=a.grad_clip, max_nonfinite_batches=a.max_nonfinite_batches,
            nan_action=a.nan_action, log_grad_norm=a.log_grad_norm
        )
        val_stats = run_validation_fullres(
            model, val_dl, criterion, device, a.tile, a.overlap, autocast_ctx,
            max_nonfinite_batches=a.max_nonfinite_batches, nan_action=a.nan_action
        )
        tr = float(train_stats["loss"])
        vl = float(val_stats["loss"])
        vp = float(val_stats["psnr"])
        vs = float(val_stats["ssim"])
        vf = float(val_stats["edgef1"])

        # Hard stop if configured and non-finite instability persists
        if a.nan_action == "stop":
            if train_stats["batches_used"] == 0:
                print(f"[STOP] Epoch {epoch}: no valid training batches (non-finite data/loss).")
                break
            if val_stats["samples_used"] == 0:
                print(f"[STOP] Epoch {epoch}: no valid validation samples (non-finite metrics/loss).")
                break

        # Optional norm freeze after warm-up
        if a.freeze_norm_epoch > 0 and epoch == a.freeze_norm_epoch:
            model.apply(freeze_norm_layers); print("[DEBUG] Norm layers frozen at epoch", epoch)

        # Periodic REAL evaluation
        real_stats = None
        if a.real_noisy_root and (a.real_eval_every > 0) and (epoch % a.real_eval_every == 0):
            real_stats = run_real_evaluation(model, Path(a.clean_root), Path(a.real_noisy_root), device, a.tile, a.overlap, out_dir/f"real_eval_e{epoch:03d}.csv")
            if real_stats:
                print(f"[REAL@e{epoch}] {real_stats}")

        # LR step
        sched.step(); lr_now = sched.get_last_lr()[0]

        # ECA diagnostics
        eca_means = eca_gate_means(model)
        eca_temp_mean, eca_gain_mean = eca_temp_gain_means(model)
        if eca_means:
            msg = "[ECA] gate means: " + ", ".join(f"{v:.3f}" for v in eca_means)
            if math.isfinite(eca_temp_mean) and math.isfinite(eca_gain_mean):
                msg += f" | temp={eca_temp_mean:.4f} gain={eca_gain_mean:.4f}"
            print(msg)

        # Logging
        real_ps = real_stats["real_psnr"] if (real_stats and "real_psnr" in real_stats) else float("nan")
        real_ss = real_stats["real_ssim"] if (real_stats and "real_ssim" in real_stats) else float("nan")
        real_fe = real_stats["real_edgef1"] if (real_stats and "real_edgef1" in real_stats) else float("nan")
        grad_norm_mean = float(train_stats["grad_norm"])
        print(
            f"Epoch {epoch:03d}/{start_epoch + a.epochs} | "
            f"Train {tr:.4f} (bad={train_stats['nonfinite_batches']}, used={train_stats['batches_used']}/{train_stats['batches_seen']}) | "
            f"Val {vl:.4f} (bad={val_stats['nonfinite_batches']}, used={val_stats['samples_used']}/{val_stats['samples_seen']}) | "
            f"PSNR {vp:.2f} | SSIM {vs:.4f} | EdgeF1 {vf:.4f} | LR {lr_now:.2e}"
        )
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{tr:.6f},{vl:.6f},{vp:.4f},{vs:.6f},{vf:.6f},"
                f"{real_ps:.4f},{real_ss:.6f},{real_fe:.6f},{lr_now:.8f},"
                f"{train_stats['nonfinite_batches']},{val_stats['nonfinite_batches']},"
                f"{train_stats['batches_used']},{val_stats['samples_used']},{grad_norm_mean:.6f}\n"
            )

        if writer is not None:
            writer.add_scalar("loss/train", tr, epoch)
            writer.add_scalar("loss/val", vl, epoch)
            writer.add_scalar("val/psnr", vp, epoch)
            writer.add_scalar("val/ssim", vs, epoch)
            writer.add_scalar("val/edgeF1", vf, epoch)
            writer.add_scalar("opt/lr", lr_now, epoch)
            writer.add_scalar("debug/train_nonfinite", float(train_stats["nonfinite_batches"]), epoch)
            writer.add_scalar("debug/val_nonfinite", float(val_stats["nonfinite_batches"]), epoch)
            if math.isfinite(grad_norm_mean):
                writer.add_scalar("opt/grad_norm_mean", grad_norm_mean, epoch)
            if real_stats:
                writer.add_scalar("real/psnr", real_ps, epoch)
                writer.add_scalar("real/ssim", real_ss, epoch)
                writer.add_scalar("real/edgeF1", real_fe, epoch)
            if eca_means:
                writer.add_scalar("eca/gate_mean_avg", float(np.mean(eca_means)), epoch)
            if math.isfinite(eca_temp_mean):
                writer.add_scalar("eca/temp_avg", eca_temp_mean, epoch)
            if math.isfinite(eca_gain_mean):
                writer.add_scalar("eca/gain_avg", eca_gain_mean, epoch)

        # Checkpoints
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "args": vars(a)}
        is_best_loss = math.isfinite(vl) and (vl < best_val)
        is_best_ssim = math.isfinite(vs) and (vs > best_ssim)
        save_ckpt(state, ckpt_dir, is_best_loss, tag="best")     # by loss (compat)
        if is_best_loss:
            best_val = vl
            best_epoch = epoch
        if is_best_ssim:
            save_ckpt(state, ckpt_dir, True, tag="best_ssim")
            best_ssim = vs

        if (epoch - best_epoch) >= a.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {a.patience} epochs).")
            break

    print(f"Best val loss so far: {best_val:.4f} (best epoch {best_epoch}); best val SSIM: {best_ssim:.4f}")

    # Final REAL evaluation
    if a.real_noisy_root:
        stats = run_real_evaluation(model, Path(a.clean_root), Path(a.real_noisy_root), device, a.tile, a.overlap, out_dir/"real_eval_final.csv")
        print(f"[REAL] {stats}")

    # Export ONNX
    if a.export_onnx:
        best_path = ckpt_dir/"best_ssim.pth"
        if not best_path.exists():
            best_path = ckpt_dir/"best.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True); model.eval().cpu()
        dummy = torch.randn(1,1,512,512)
        torch.onnx.export(model, dummy, a.export_onnx,
                          input_names=["input"], output_names=["output"],
                          opset_version=17,
                          dynamic_axes={"input": {2:"H", 3:"W"}, "output": {2:"H", 3:"W"}})
        print(f"Exported ONNX to {a.export_onnx}")

    if writer is not None: writer.close()

# ---------------- CLI ----------------

def parse_int_list(s: str) -> Optional[List[int]]:
    if not s: return None
    parts = [p.strip() for p in s.split(",") if p.strip()!=""]
    if not parts: return None
    out = []
    for p in parts:
        try: out.append(int(p))
        except: pass
    return out if out else None

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
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--depth", type=int, default=17)
    p.add_argument("--norm", type=str, default="group", choices=["none","batch","group"])
    p.add_argument("--gn-groups", type=int, default=0, help="for --norm group: 0=auto [8,4,2,1], >0 force max group count")
    p.add_argument("--use-eca", dest="use_eca", action="store_true", help="enable ECA blocks")
    p.add_argument("--no-eca", dest="use_eca", action="store_false", help="disable ECA blocks for fair baseline")
    p.set_defaults(use_eca=True)
    # --- ECA controls
    p.add_argument("--eca-interval", type=int, default=4, help="if --eca-positions empty, insert ECA every N blocks")
    p.add_argument("--eca-k", type=int, default=0, help="0=auto; odd>=3 for fixed")
    p.add_argument("--eca-temp", type=float, default=0.75)
    p.add_argument("--eca-gain", type=float, default=0.5)
    p.add_argument("--eca-centered", action="store_true", help="use centered multiplicative band [1-gain,1+gain]")
    p.add_argument("--eca-use-maxpool", dest="eca_use_maxpool", action="store_true", help="use dual pooling (GAP+GMP) in ECA")
    p.add_argument("--eca-no-maxpool", dest="eca_use_maxpool", action="store_false", help="disable GMP branch, use GAP only")
    p.set_defaults(eca_use_maxpool=True)
    p.add_argument("--eca-multi-scale", action="store_true", help="use parallel ECA kernels k3/k5/k7")
    p.add_argument("--eca-learnable-temp-gain", action="store_true", help="make ECA temp/gain learnable")
    p.add_argument("--eca-temp-min", type=float, default=1e-6, help="minimum temperature for learnable/static temp")
    p.add_argument("--eca-gain-min", type=float, default=0.01, help="minimum centered gain clamp")
    p.add_argument("--eca-gain-max", type=float, default=1.5, help="maximum centered gain clamp")
    p.add_argument("--eca-order", type=str, default="post", choices=["post","pre"], help="apply ECA after or before ConvBlock")
    p.add_argument("--eca-preset", type=str, default="none", choices=["none","shallow3","dense_shallow"], help="preset eca positions if --eca-positions not provided")
    p.add_argument("--eca-positions", type=str, default="", help="comma list of block indices (0-based) for ECA (e.g., '0,1,2')")
    # --- Spatial lite attention
    p.add_argument("--spa-interval", type=int, default=0, help="0=off; otherwise add spatial-lite attention every N blocks")
    p.add_argument("--spa-ks", type=int, default=5, help="spatial attention kernel size")
    # --- Noise & aug
    p.add_argument("--sigma-g", type=float, default=0.02)
    p.add_argument("--speckle", type=float, default=0.2)
    p.add_argument("--speckle-kind", type=str, default="gauss", choices=["gauss","burr"])
    # --- Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--export-onnx", type=str, default="")
    p.add_argument("--resume", type=str, default="", help="path to ckpt or 'auto' (looks in output_dir/checkpoints/last.pth)")
    p.add_argument("--resume-strict", dest="resume_strict", action="store_true", help="strict checkpoint loading (recommended)")
    p.add_argument("--resume-nonstrict", dest="resume_strict", action="store_false", help="allow partial checkpoint loading")
    p.set_defaults(resume_strict=True)
    p.add_argument("--w-edge", type=float, default=0.10, help="edge-preserving weight in loss")
    p.add_argument("--separate-eca-param-optim", action="store_true", help="use separate optimizer group for ECA temp/gain params")
    p.add_argument("--eca-param-lr-scale", type=float, default=0.5, help="LR scale for ECA temp/gain param group")
    p.add_argument("--eca-param-weight-decay", type=float, default=1e-3, help="weight decay for ECA temp/gain param group")
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; <=0 disables clipping")
    p.add_argument("--max-nonfinite-batches", type=int, default=3, help="max skipped non-finite batches before stop when nan-action=stop")
    p.add_argument("--nan-action", type=str, default="stop", choices=["skip","stop"], help="policy for non-finite loss/metrics")
    p.add_argument("--log-grad-norm", action="store_true", help="log mean gradient norm per epoch")
    p.add_argument("--freeze-norm-epoch", type=int, default=0, help="0=never; otherwise freeze BN/GN at this epoch")
    p.add_argument("--real-eval-every", type=int, default=5, help="0=off; otherwise run REAL evaluation every N epochs")
    a = p.parse_args()
    return Args(
        clean_root=a.clean_root, output_dir=a.output_dir, split_mode=a.split_mode, val_ratio=a.val_ratio,
        lofo_group=a.lofo_group, real_noisy_root=a.real_noisy_root if a.real_noisy_root else None,
        tile=a.tile, overlap=a.overlap, batch_size=a.batch_size, workers=a.workers, epochs=a.epochs, lr=a.lr,
        weight_decay=a.weight_decay, features=a.features, depth=a.depth, norm=a.norm,
        gn_groups=a.gn_groups, use_eca=bool(a.use_eca),
        eca_interval=a.eca_interval, eca_k=a.eca_k, eca_temp=a.eca_temp, eca_gain=a.eca_gain,
        eca_centered=bool(a.eca_centered),
        eca_use_maxpool=bool(a.eca_use_maxpool),
        eca_multi_scale=bool(a.eca_multi_scale),
        eca_learnable_temp_gain=bool(a.eca_learnable_temp_gain),
        eca_temp_min=a.eca_temp_min,
        eca_gain_min=a.eca_gain_min,
        eca_gain_max=a.eca_gain_max,
        eca_order=a.eca_order,
        eca_preset=a.eca_preset,
        eca_positions=parse_int_list(a.eca_positions),
        spa_interval=a.spa_interval, spa_ks=a.spa_ks, sigma_g=a.sigma_g, speckle=a.speckle, speckle_kind=a.speckle_kind,
        seed=a.seed, patience=a.patience, device=a.device,
        tensorboard=bool(a.tensorboard), export_onnx=(a.export_onnx if a.export_onnx else None),
        resume=(a.resume if a.resume else None), resume_strict=bool(a.resume_strict), w_edge=a.w_edge,
        eca_param_lr_scale=a.eca_param_lr_scale, eca_param_weight_decay=a.eca_param_weight_decay,
        separate_eca_param_optim=bool(a.separate_eca_param_optim),
        grad_clip=a.grad_clip, max_nonfinite_batches=a.max_nonfinite_batches,
        nan_action=a.nan_action, log_grad_norm=bool(a.log_grad_norm),
        freeze_norm_epoch=a.freeze_norm_epoch, real_eval_every=a.real_eval_every
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)

