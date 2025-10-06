#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx_from_ckpt.py
Export DnCNN-Lite+ECA(+optional SpatialLite) to ONNX directly from a training checkpoint.

Usage:
python export_onnx_from_ckpt.py ^
  --ckpt "C:\...\outputs_W01\checkpoints\best.pth" ^
  --onnx "C:\...\outputs_W01\dncnn_lite_eca.onnx"
"""
import argparse, torch, torch.nn as nn

def make_norm(kind: str, ch: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "batch": return nn.BatchNorm2d(ch)
    if kind == "group":
        for g in [8,4,2,1]:
            if ch % g == 0: return nn.GroupNorm(g, ch)
        return nn.GroupNorm(1, ch)
    return nn.Identity()

class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        k = k_size if (k_size % 2 == 1) else (k_size + 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).transpose(1,2)
        y = self.conv1d(y)
        y = self.sigmoid(y).transpose(1,2).unsqueeze(-1)
        return x * y

class SpatialLiteAttention(nn.Module):
    def __init__(self, k: int = 5):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        import torch
        avg = torch.mean(x, dim=1, keepdim=True)
        mxx, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.cat([avg, mxx], dim=1)
        a = self.sigmoid(self.conv(m))
        return x * a

class DnCNNLiteECA(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=32, depth=17, norm="group",
                 eca_interval=4, eca_k=3, spa_interval=0, spa_k=5, residual_weight=1.0):
        super().__init__()
        c = features
        self.residual_weight = residual_weight
        self.head = nn.Sequential(nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
                                  nn.ReLU(inplace=True))
        blocks, ecas, spas = [], [], []
        for i in range(depth - 2):
            blocks.append(nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                make_norm(norm, c),
                nn.ReLU(inplace=True)
            ))
            ecas.append(ECA(c, eca_k) if ((i+1) % eca_interval == 0) else nn.Identity())
            if spa_interval and ((i+1) % spa_interval == 0):
                spas.append(SpatialLiteAttention(spa_k))
            else:
                spas.append(nn.Identity())
        self.blocks = nn.ModuleList(blocks)
        self.eca = nn.ModuleList(ecas)
        self.spa = nn.ModuleList(spas)
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1, bias=False)
    def forward(self, x):
        idt = x; h = self.head(x)
        for b, ec, sp in zip(self.blocks, self.eca, self.spa):
            h = b(h); h = ec(h); h = sp(h)
        noise = self.tail(h)
        return idt - self.residual_weight * noise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("args", {})
    model = DnCNNLiteECA(
        in_ch=1, out_ch=1,
        features=cfg.get("features", 32),
        depth=cfg.get("depth", 17),
        norm=cfg.get("norm", "group"),
        eca_interval=cfg.get("eca_interval", 4),
        eca_k=3,
        spa_interval=cfg.get("spa_interval", 0),
        spa_k=cfg.get("spa_ks", 5),
        residual_weight=1.0
    )
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    dummy = torch.randn(1,1,512,512)  # dynamic axes will allow any H,W
    torch.onnx.export(model, dummy, args.onnx,
                      input_names=["input"], output_names=["output"],
                      opset_version=17,
                      dynamic_axes={"input": {2:"H",3:"W"}, "output": {2:"H",3:"W"}})
    print("Exported ONNX to", args.onnx)

if __name__ == "__main__":
    main()
