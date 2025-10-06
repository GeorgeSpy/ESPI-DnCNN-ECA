#!/usr/bin/env python3
# bench_cli_denoise.py — Time PyTorch vs ONNX using your existing CLI scripts.
# Works on Windows CMD or PowerShell.

import argparse, shutil, subprocess, time, sys
from pathlib import Path

def collect_pngs(root: Path, limit: int):
    files = sorted(root.rglob("*.png"))
    if limit > 0:
        files = files[:limit]
    return files

def copy_subset(files, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)

def run_cmd(cmd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.perf_counter()
    return (t1-t0), p.returncode, p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch", required=True, help="path to batch_denoise_pytorch_v3.py")
    ap.add_argument("--onnx", required=True, help="path to batch_denoise_onnx.py")
    ap.add_argument("--ckpt", required=True, help="best.pth")
    ap.add_argument("--onnx-model", required=True, help="onnx file")
    ap.add_argument("--input", required=True, help="averaged input dir with PNGs")
    ap.add_argument("--tmp-root", required=True, help="temporary root for subset")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERR] input not found: {inp}")
        sys.exit(2)

    tmp = Path(args.tmp_root)
    pt_in = tmp / "pt_in"; onnx_in = tmp / "onnx_in"
    pt_out = tmp / "pt_out"; onnx_out = tmp / "onnx_out"
    for d in [pt_in, onnx_in, pt_out, onnx_out]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    files = collect_pngs(inp, args.limit)
    if not files:
        print(f"[ERR] No PNGs found under {inp}")
        sys.exit(3)
    copy_subset(files, pt_in)
    copy_subset(files, onnx_in)

    # PyTorch
    cmd_pt = ["python", args.pytorch, "--ckpt", args.ckpt, "--input", str(pt_in),
              "--output", str(pt_out), "--tile", str(args.tile), "--overlap", str(args.overlap),
              "--device", args.device]
    t_pt, rc_pt, out_pt = run_cmd(cmd_pt)

    # ONNX
    cmd_ox = ["python", args.onnx, "--onnx", args.onnx_model, "--input", str(onnx_in),
              "--output", str(onnx_out), "--tile", str(args.tile), "--overlap", str(args.overlap)]
    t_ox, rc_ox, out_ox = run_cmd(cmd_ox)

    n = len(files)
    print("=== PyTorch ===")
    print(f"files: {n}, time: {t_pt:.2f}s, {(t_pt/n)*1000:.1f} ms/img, {n/t_pt:.2f} FPS, rc={rc_pt}")
    print("=== ONNX ===")
    print(f"files: {n}, time: {t_ox:.2f}s, {(t_ox/n)*1000:.1f} ms/img, {n/t_ox:.2f} FPS, rc={rc_ox}")
    # To see raw outputs:
    # print(out_pt)
    # print(out_ox)

if __name__ == "__main__":
    main()
