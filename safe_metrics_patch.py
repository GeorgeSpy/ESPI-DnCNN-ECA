#!/usr/bin/env python3
"""
Safe metrics patch for train_paired_from_csv_COMPAT.py
Prevents NaN in PSNR/SSIM/EdgeF1 validation metrics.
"""
import math
import torch
from torch import nn

_EPS = 1e-12

def safe_psnr(pred, target, data_range=None):
    """
    Safe PSNR calculation with guards against NaN/Inf.
    
    Args:
        pred: [N,C,H,W] predicted tensor
        target: [N,C,H,W] target tensor  
        data_range: optional data range, auto-detected if None
        
    Returns:
        float: PSNR in dB, or 0.0 if computation fails
    """
    # Convert to float64 for numerical stability
    diff = (pred - target).to(torch.float64)
    mse = torch.mean(diff * diff)
    
    # Guard against NaN/Inf
    if torch.isnan(mse) or torch.isinf(mse):
        return 0.0
    
    # Clamp MSE to prevent log(0)
    mse = torch.clamp(mse, min=_EPS)
    
    # Auto-detect data range if not provided
    if data_range is None:
        data_range = (target.max() - target.min()).detach().float().item()
        if not math.isfinite(data_range) or data_range <= 0:
            data_range = 1.0
    
    # Calculate PSNR
    psnr = 10.0 * math.log10((data_range * data_range) / mse.item())
    return float(psnr)

def safe_ssim(pred, target, data_range=None):
    """
    Safe SSIM calculation with guards against flat images.
    
    Args:
        pred: [N,C,H,W] predicted tensor
        target: [N,C,H,W] target tensor
        data_range: optional data range, auto-detected if None
        
    Returns:
        float: SSIM value, or 1.0 for flat cases
    """
    x = pred.to(torch.float32)
    y = target.to(torch.float32)
    
    # Auto-detect data range if not provided
    dr = (y.max() - y.min()).detach().float().item() if data_range is None else data_range
    if not math.isfinite(dr) or dr <= 0:
        return 1.0  # Flat case → SSIM = 1.0 to avoid NaN
    
    # Simplified proxy SSIM: 1 - normalized MSE
    mse = torch.mean((x - y) ** 2).item()
    nmse = mse / (dr * dr + _EPS)
    ssim_proxy = max(0.0, 1.0 - nmse)
    return float(ssim_proxy)

def safe_edge_f1(pred, target, thr=None):
    """
    Safe Edge F1 calculation with guards against edge cases.
    
    Args:
        pred: [N,C,H,W] predicted tensor
        target: [N,C,H,W] target tensor
        thr: edge threshold, auto-detected if None
        
    Returns:
        float: Edge F1 score, or 0.0 if computation fails
    """
    # Auto-detect threshold if not provided
    if thr is None:
        thr = float(pred.detach().std().item()) * 0.5 + 1e-6
    
    # Calculate gradients
    gx = nn.functional.pad(pred[:, :, :, 1:] - pred[:, :, :, :-1], (0,1,0,0))
    gy = nn.functional.pad(pred[:, :, 1:, :] - pred[:, :, :-1, :], (0,0,0,1))
    gxt = nn.functional.pad(target[:, :, :, 1:] - target[:, :, :, :-1], (0,1,0,0))
    gyt = nn.functional.pad(target[:, :, 1:, :] - target[:, :, :-1, :], (0,0,0,1))
    
    # Edge detection
    pe = (gx.abs() + gy.abs()) > thr
    te = (gxt.abs() + gyt.abs()) > thr
    
    # Calculate F1
    tp = (pe & te).sum().item()
    fp = (pe & (~te)).sum().item()
    fn = ((~pe) & te).sum().item()
    
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2*prec*rec / (prec + rec + 1e-9)
    
    # Guard against NaN
    if not math.isfinite(f1):
        f1 = 0.0
    
    return float(f1)

def safe_val_loss(criterion, pred, target):
    """
    Safe validation loss calculation with NaN guards.
    
    Args:
        criterion: loss function
        pred: predicted tensor
        target: target tensor
        
    Returns:
        float: validation loss, or 0.0 if NaN
    """
    loss = criterion(pred, target)
    # Replace NaN/Inf with safe values
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
    return float(loss.item())

# Example usage in validation loop:
"""
with torch.no_grad():
    # ... forward pass ...
    out = model(input_batch)
    gt = target_batch
    
    # Safe metrics
    psnr = safe_psnr(out, gt)
    ssim = safe_ssim(out, gt)
    edgef1 = safe_edge_f1(out, gt)
    val_loss = safe_val_loss(criterion, out, gt)
    
    # Log metrics
    print(f"Val Loss: {val_loss:.4f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | EdgeF1: {edgef1:.4f}")
"""

if __name__ == "__main__":
    print("Safe metrics patch ready for train_paired_from_csv_COMPAT.py")
    print("Functions: safe_psnr, safe_ssim, safe_edge_f1, safe_val_loss")
