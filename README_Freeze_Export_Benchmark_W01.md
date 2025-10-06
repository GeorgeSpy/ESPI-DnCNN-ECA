# ESPI Modes Project – Freeze, Presets, Export & Benchmark

## 1) Freeze περιβάλλοντος
Εκτέλεσε ΜΕΣΑ στο virtualenv (venv). Δουλεύει το ίδιο από **CMD** ή **PowerShell**.

### CMD
```
cd C:\ESPI_DnCNN
python -V > environment_info.txt
pip freeze > requirements.txt
python - <<PY > env_extra.txt
import platform, torch, sys
print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
type environment_info.txt env_extra.txt > env_all.txt
del env_extra.txt
```

### PowerShell
```
cd C:\ESPI_DnCNN
python -V > environment_info.txt
pip freeze > requirements.txt
python - <<'PY' | Out-File env_extra.txt -Encoding utf8
import platform, torch, sys
print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
Get-Content environment_info.txt, env_extra.txt | Set-Content env_all.txt
Remove-Item env_extra.txt
```

## 2) Presets (Phase & Denoiser)
Άνοιξε/προσαρμόσε το `project_config_W01.yaml` (paths). Προτεινόμενα defaults:
- ROI mask: `C:\\Users\\Geiorge HTPC\\Desktop\\Mpouzoukia\\roi_mask.png`
- Phase STRICT: band=18, center_suppress=16, flatfield=80, unwrap=auto
- Denoiser: tile 224, overlap 32, device cpu
- Checkpoint: `C:\\Users\\Geiorge HTPC\\Desktop\\Mpouzoukia\\outputs_W01\\checkpoints\\best.pth`

## 3) Export σε ONNX
**CMD ή PowerShell (ίδια εντολή σε μία γραμμή):**
```
python C:\ESPI_DnCNN\espi_dncnn_lite_eca_full_cpu_safe_FIXED_PATCHED_v2.py --device cpu --clean-root "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_ESPI_90db-Averaged" --output-dir "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\outputs_W01" --epochs 1 --resume auto --export-onnx "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\outputs_W01\dncnn_lite_eca.onnx"
```

## 4) Benchmark PyTorch vs ONNX
Χρησιμοποίησε το `bench_cli_denoise.py` που δίνεται εδώ.

**CMD ή PowerShell (μία γραμμή):**
```
python C:\ESPI_DnCNN\bench_cli_denoise.py --pytorch "C:\ESPI_DnCNN\batch_denoise_pytorch_v3.py" --onnx "C:\ESPI_DnCNN\batch_denoise_onnx.py" --ckpt "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\outputs_W01\checkpoints\best.pth" --onnx-model "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\outputs_W01\dncnn_lite_eca.onnx" --input "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_ESPI_90db-Averaged" --tmp-root "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\bench_tmp" --tile 224 --overlap 32 --limit 30 --device cpu
```

**Τι εκτυπώνει:** συνολικό χρόνο, ms/εικόνα, FPS για PyTorch και ONNX.
