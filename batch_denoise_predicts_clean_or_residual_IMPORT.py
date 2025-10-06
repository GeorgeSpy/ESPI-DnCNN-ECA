# -*- coding: utf-8 -*-
# batch_denoise_predicts_clean_or_residual_IMPORT.py
"""
Batch denoise with tile+Hann. Tries to import the exact training model class
from espi_dncnn_lite_eca_full_cpu_safe_FIXED_PATCHED_v2.py (same folder);
falls back to a plain DnCNN-lite if import fails.
"""

from __future__ import annotations
import argparse, inspect, importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- IO ----------
def imread_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0

def imsave_uint01(x01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x01, 0.0, 1.0)
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(str(path))


# ---------- dynamic import of training model ----------
def try_import_training_model() -> tuple[type[nn.Module] | None, dict]:
    base = Path(__file__).resolve().parent
    candidates = [
        "espi_dncnn_lite_eca_full_cpu_safe_FIXED_PATCHED_v2.py",
        "espi_dncnn_lite_eca_full_cpu_safe_FIXED.py",
        "espi_dncnn_lite_eca_full_cpu_safe.py",
    ]
    for name in candidates:
        p = base / name
        if not p.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(p.stem, str(p))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(mod)                 # type: ignore

            cls = None
            for nm, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, nn.Module):
                    low = nm.lower()
                    if ("dncnn" in low) or ("eca" in low) or ("lite" in low):
                        cls = obj
                        break
            if cls is None:
                continue

            sig = inspect.signature(cls.__init__)
            par = sig.parameters
            defkw = {}
            if "in_ch" in par: defkw["in_ch"] = 1
            if "in_channels" in par: defkw["in_channels"] = 1
            if "out_ch" in par: defkw["out_ch"] = 1
            if "out_channels" in par: defkw["out_channels"] = 1
            if "features" in par: defkw["features"] = 64
            if "num_features" in par: defkw["num_features"] = 64
            if "depth" in par: defkw["depth"] = 17
            if "num_layers" in par: defkw["num_layers"] = 17
            if "eca_every" in par: defkw["eca_every"] = 4
            if "eca_interval" in par: defkw["eca_interval"] = 4
            return cls, defkw
        except Exception:
            continue
    return None, {}


def extract_state_dict(ckpt: dict) -> dict:
    for k in ("model_state", "model", "state_dict", "net", "weights"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in ckpt.items()):
        return ckpt
    raise RuntimeError("No state_dict found in checkpoint.")


def build_model_from_ckpt(ckpt: dict) -> nn.Module:
    sd = extract_state_dict(ckpt)

    cls, defkw = try_import_training_model()
    if cls is not None:
        try:
            sig = inspect.signature(cls.__init__)
            kwargs = {k: v for k, v in defkw.items() if k in sig.parameters}
            model = cls(**kwargs)  # type: ignore
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[LOAD] training-class load_state_dict -> missing:{len(missing)} unexpected:{len(unexpected)}")
            return model
        except Exception as e:
            print(f"[WARN] training-class load failed ({e}); using fallback.")

    class SimpleDnCNN(nn.Module):
        def __init__(self, depth=17, ch=64):
            super().__init__()
            layers = [nn.Conv2d(1, ch, 3, 1, 1), nn.ReLU(inplace=True)]
            for _ in range(depth - 2):
                layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(ch, 1, 3, 1, 1)]
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    model = SimpleDnCNN(depth=17, ch=64)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD:FALLBACK] missing:{len(missing)} unexpected:{len(unexpected)}")
    return model


# ---------- tiling with Hann + replicate pad ----------
def hann2d(h: int, w: int) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False)
    wx = torch.hann_window(w, periodic=False)
    return torch.outer(wy, wx)

@torch.no_grad()
def denoise_tiled(model: nn.Module, img: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    _, _, H, W = img.shape
    out = torch.zeros_like(img)
    acc = torch.zeros_like(img)

    step = tile - overlap
    if step <= 0:
        step = tile

    for top in range(0, H, step):
        for left in range(0, W, step):
            h0 = min(tile, H - top)
            w0 = min(tile, W - left)
            patch = img[:, :, top: top + h0, left: left + w0]
            ph = tile - h0
            pw = tile - w0
            if ph > 0 or pw > 0:
                patch = F.pad(patch, (0, pw, 0, ph), mode="replicate")

            pred = model(patch)
            pred = pred[:, :, :h0, :w0]

            w2d = hann2d(h0, w0).to(img.device)[None, None, :, :]
            out[:, :, top: top + h0, left: left + w0] += pred * w2d
            acc[:, :, top: top + h0, left: left + w0] += w2d

    out = out / torch.clamp(acc, min=1e-6)
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--predicts-clean", action="store_true")
    ap.add_argument("--residual-weight", type=float, default=1.0)
    ap.add_argument("--percentile-norm", type=float, default=0.0)  # 0=off, e.g. 99.0 to enable
    ap.add_argument("--prefilter", choices=["none", "median3"], default="none")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input); outp = Path(args.output)
    files = sorted([p for p in inp.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]])
    if args.limit > 0:
        files = files[:args.limit]

    print(f"[INFO] Files: {len(files)} | tile={args.tile} overlap={args.overlap} device={args.device}")

    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model_from_ckpt(ck).to(args.device)
    model.eval()

    for i, f in enumerate(files, 1):
        x = imread_gray(f)

        if args.percentile_norm and args.percentile_norm > 0.0:
            p_lo = 100.0 - float(args.percentile_norm)
            p_hi = float(args.percentile_norm)
            lo = float(np.percentile(x, p_lo))
            hi = float(np.percentile(x, p_hi))
            if hi > lo + 1e-6:
                x = (x - lo) / (hi - lo)
                x = np.clip(x, 0.0, 1.0)

        if args.prefilter == "median3":
            try:
                import cv2
                x = cv2.medianBlur((x * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
            except Exception:
                pass

        xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(args.device)

        with torch.no_grad():
            pred = denoise_tiled(model, xt, args.tile, args.overlap)

        if args.predicts_clean:
            clean = pred
        else:
            clean = xt - args.residual_weight * pred

        clean_np = clean.clamp(0, 1).cpu().squeeze().numpy()
        imsave_uint01(clean_np, outp / f.name)

        if i % 20 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {f.name}")

    print(f"[DONE] Saved to {outp}")


if __name__ == "__main__":
    main()
