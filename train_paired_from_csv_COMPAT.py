#!/usr/bin/env python3
"""Pair-driven fine-tuning wrapper around the ESPI DnCNN model.

Loads explicit (noisy, clean) mappings from a CSV so we can mix real single-shots
and pseudo-noisy averages without renaming files on disk.
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import espi_dncnn_lite_eca_full_cpu_safe_FIXED_PATCHED_v2 as base

Array = np.ndarray
Pair = Tuple[Path, Path]


def pad_to_min(x: Array, min_h: int, min_w: int) -> Array:
    h, w = x.shape
    if h >= min_h and w >= min_w:
        return x
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        x = np.pad(x, ((top, bottom), (left, right)), mode="reflect")
    return x


def random_crop_pair(noisy: Array, clean: Array, size: int) -> Tuple[Array, Array]:
    noisy = pad_to_min(noisy, size, size)
    clean = pad_to_min(clean, size, size)
    h, w = noisy.shape
    if h == size and w == size:
        return noisy, clean
    y0 = random.randint(0, h - size)
    x0 = random.randint(0, w - size)
    return noisy[y0:y0 + size, x0:x0 + size], clean[y0:y0 + size, x0:x0 + size]


def center_crop_pair(noisy: Array, clean: Array, size: int) -> Tuple[Array, Array]:
    noisy = pad_to_min(noisy, size, size)
    clean = pad_to_min(clean, size, size)
    h, w = noisy.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return noisy[y0:y0 + size, x0:x0 + size], clean[y0:y0 + size, x0:x0 + size]


class CSVPatchDataset(Dataset):
    def __init__(self, pairs: Sequence[Pair], crop: int, augment: bool):
        self.pairs = list(pairs)
        self.crop = int(crop)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_p, clean_p = self.pairs[idx]
        noisy = base.imread_uint(noisy_p)
        clean = base.imread_uint(clean_p)
        h = min(noisy.shape[0], clean.shape[0])
        w = min(noisy.shape[1], clean.shape[1])
        noisy = noisy[:h, :w]
        clean = clean[:h, :w]
        if self.crop > 0:
            if self.augment and random.random() < 0.5:
                noisy, clean = random_crop_pair(noisy, clean, self.crop)
            else:
                noisy, clean = center_crop_pair(noisy, clean, self.crop)
        if self.augment:
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=1).copy()
                clean = np.flip(clean, axis=1).copy()
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=0).copy()
                clean = np.flip(clean, axis=0).copy()
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()
        return noisy_t, clean_t


class CSVFullImageDataset(Dataset):
    def __init__(self, pairs: Sequence[Pair]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_p, clean_p = self.pairs[idx]
        noisy = base.imread_uint(noisy_p)
        clean = base.imread_uint(clean_p)
        h = min(noisy.shape[0], clean.shape[0])
        w = min(noisy.shape[1], clean.shape[1])
        noisy = torch.from_numpy(noisy[:h, :w]).unsqueeze(0).unsqueeze(0).float()
        clean = torch.from_numpy(clean[:h, :w]).unsqueeze(0).unsqueeze(0).float()
        return noisy, clean


def read_pairs(csv_path: Path) -> List[Pair]:
    rows: List[Pair] = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            noisy = Path(row["noisy"])
            clean = Path(row["clean"])
            if noisy.exists() and clean.exists():
                rows.append((noisy, clean))
            else:
                skipped += 1
    if skipped:
        print(f"[WARN] skipped {skipped} rows with missing files")
    if not rows:
        raise FileNotFoundError(f"No valid pairs in {csv_path}")
    return rows


def split_pairs(pairs: Sequence[Pair], val_ratio: float, seed: int) -> Tuple[List[Pair], List[Pair]]:
    idx = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


@dataclass
class Args:
    pairs_csv: Path
    output_dir: Path
    epochs: int
    batch_size: int
    workers: int
    lr: float
    tile: int
    overlap: int
    val_ratio: float
    seed: int
    device: str
    resume: Path | None
    init: Path | None
    w_edge: float


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, type=Path, help="CSV with columns noisy,clean")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory for checkpoints/logs")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tile", type=int, default=256, help="Crop size for training patches and val tiles")
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--resume", type=Path)
    p.add_argument("--init", type=Path, help="Optional checkpoint to initialize model weights")
    p.add_argument("--w-edge", type=float, default=0.10)
    args = p.parse_args()
    return Args(
        pairs_csv=args.pairs,
        output_dir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch,
        workers=args.workers,
        lr=args.lr,
        tile=args.tile,
        overlap=args.overlap,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        init=args.init,
        w_edge=args.w_edge,
    )


def main():
    raw = parse_args()
    base.set_seed(raw.seed, deterministic=True)
    device = torch.device(raw.device if (raw.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = raw.output_dir
    ckpt_dir = out_dir / "checkpoints"
    log_csv = out_dir / "train_log.csv"
    base.ensure_dir(ckpt_dir)

    pairs = read_pairs(raw.pairs_csv)
    train_pairs, val_pairs = split_pairs(pairs, raw.val_ratio, raw.seed)
    print(f"[DATA] train={len(train_pairs)} val={len(val_pairs)}")

    train_ds = CSVPatchDataset(train_pairs, crop=raw.tile, augment=True)
    val_ds = CSVFullImageDataset(val_pairs)

    train_dl = DataLoader(train_ds, batch_size=raw.batch_size, shuffle=True,
                          num_workers=raw.workers, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=False)

    cfg = base.DnCNNLiteECAConfig(features=32, depth=17, norm="group",
                                  eca_interval=4, residual_weight=1.0)
    model = base.DnCNNLiteECA(cfg).to(device)
    criterion = base.EdgeAwareLoss(0.8, 0.2, raw.w_edge).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=raw.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=raw.epochs)

    start_epoch = 0
    if raw.init and raw.init.exists():
        ckpt = torch.load(raw.init, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            print(f"[INIT] loaded weights from {raw.init}")
    if raw.resume and raw.resume.exists():
        ckpt = torch.load(raw.resume, map_location=device)
        try:
            model.load_state_dict(ckpt["model"], strict=False)
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[RESUME] restored state from {raw.resume} (epoch {start_epoch})")
        except KeyError:
            print(f"[RESUME] malformed checkpoint: {raw.resume}")

    scaler, autocast_ctx = base.get_amp_objects(device)

    if not log_csv.exists():
        base.ensure_dir(log_csv.parent)
        with open(log_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_psnr,val_ssim,val_edgeF1,lr\n")

    best_val = float("inf")
    best_epoch = start_epoch
    for epoch in range(start_epoch + 1, start_epoch + raw.epochs + 1):
        train_loss = base.run_epoch_train(model, train_dl, criterion, opt, scaler, autocast_ctx, device)
        val_loss, val_psnr, val_ssim, val_edge = base.run_validation_fullres(
            model, val_dl, criterion, device, raw.tile, raw.overlap, autocast_ctx)
        sched.step()
        lr_now = sched.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | PSNR {val_psnr:.2f} | SSIM {val_ssim:.4f} | EdgeF1 {val_edge:.4f} | LR {lr_now:.2e}")
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_psnr:.4f},{val_ssim:.6f},{val_edge:.6f},{lr_now:.8f}\n")
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "args": vars(raw),
        }
        is_best = val_loss < best_val
        base.save_ckpt(state, ckpt_dir, is_best)
        if is_best:
            best_val = val_loss
            best_epoch = epoch
        if epoch - best_epoch >= 10:
            print(f"Early stop at epoch {epoch} (no improvement for 10 epochs)")
            break

    print(f"Best val loss {best_val:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
