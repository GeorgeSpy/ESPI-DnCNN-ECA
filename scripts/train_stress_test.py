#!/usr/bin/env python3
import os
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

import torchvision
from torchvision import models
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# EXACT ARCHITECTURES FROM retrain_dncnn_real.py (MATCHING CHECKPOINTS 100%)
# ---------------------------------------------------------

def eca_kernel_for(C: int, gamma: float = 2.0, b: float = 1.0) -> int:
    val = int(round(math.log2(max(C, 1)) / gamma + b / gamma))
    return max(3, val | 1)

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

def load_denoiser(ckpt_path, version, device):
    if version == "v4":
        m = DnCNNLite(version="v4", use_eca=True, eca_positions=[0,1,2])
    else:
        m = DnCNNLite(version="v5", use_eca=True, eca_positions=[0,1,2,3,6,10,14])
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    m.load_state_dict(state, strict=True) # Strict now because architectures should match 100%
    return m.to(device).eval()

# ---------------------------------------------------------
# STRESS TEST DATASET
# ---------------------------------------------------------

class ESPICNNStressDataset(Dataset):
    def __init__(self, df, root_dir, noise_sigma=0.0, augment=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root_dir)
        self.noise_sigma = noise_sigma / 255.0
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(self.root / row['filename']) as im:
            im = im.convert('L')
            if im.size != (256, 256): im = im.resize((256, 256), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        
        if self.augment:
            if random.random() > 0.5: x = torch.flip(x, [1])
            if random.random() > 0.5: x = torch.flip(x, [2])
            if random.random() > 0.5: 
                angle = random.uniform(-10, 10)
                x = torchvision.transforms.functional.rotate(x, angle)
                
        if self.noise_sigma > 0:
            x = x + torch.randn_like(x) * self.noise_sigma
            x = torch.clamp(x, 0, 1)
        return x, int(row['label'])

class ResNet18Gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(512, 5)
    def forward(self, x): return self.model(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--denoiser-ckpt', default='')
    p.add_argument('--denoiser-version', default='v4')
    p.add_argument('--noise-sigma', type=float, default=25.0)
    p.add_argument('--tag', required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path("results/stress_test") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    denoiser = load_denoiser(args.denoiser_ckpt, args.denoiser_version, device) if args.denoiser_ckpt else None

    df = pd.read_csv(args.csv)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    train_ds = ESPICNNStressDataset(train_df, args.root, noise_sigma=args.noise_sigma, augment=True)
    val_ds = ESPICNNStressDataset(val_df, args.root, noise_sigma=args.noise_sigma, augment=False)

    counts = train_df['label'].value_counts().sort_index().values
    weights = 1.0 / counts
    sample_weights = [weights[l] for l in train_df['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = ResNet18Gray().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler(args.device)

    best_acc = 0.0
    with open(out_dir/"log.csv", "w") as log:
        log.write("epoch,acc,f1\n")
        for ep in range(1, args.epochs+1):
            model.train()
            tl = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                if denoiser: 
                    with torch.no_grad(): x = torch.clamp(denoiser(x), 0, 1)
                with autocast(args.device):
                    loss = crit(model(x), y)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tl += loss.item()

            model.eval()
            all_p, all_y = [], []
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    if denoiser: x = torch.clamp(denoiser(x), 0, 1)
                    all_p.extend(model(x).argmax(1).cpu().numpy())
                    all_y.extend(y.cpu().numpy())
            
            rep = classification_report(all_y, all_p, output_dict=True, zero_division=0)
            acc, mf1 = rep['accuracy'], rep['macro avg']['f1-score']
            print(f"[{args.tag}] Ep {ep}: Acc {acc:.4f} | F1 {mf1:.4f}")
            log.write(f"{ep},{acc:.4f},{mf1:.4f}\n"); log.flush()
            if acc > best_acc: best_acc = acc; torch.save(model.state_dict(), out_dir/"best.pth")

if __name__ == "__main__": main()
