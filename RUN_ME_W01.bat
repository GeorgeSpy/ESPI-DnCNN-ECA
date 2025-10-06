@echo off
setlocal ENABLEDELAYEDEXPANSION
title ESPI W01 - Freeze, ONNX Export, Benchmark

rem ====== EDITABLE PATHS (check once) ======
set "PROJ=C:\ESPI_DnCNN"
set "ROOT=C:\Users\Geiorge HTPC\Desktop\Mpouzoukia"
set "AVERAGED=%ROOT%\W01_ESPI_90db-Averaged"
set "CKPT=%ROOT%\outputs_W01\checkpoints\best.pth"
set "ONNX=%ROOT%\outputs_W01\dncnn_lite_eca.onnx"

rem scripts inside project
set "TRAIN=%PROJ%\espi_dncnn_lite_eca_full_cpu_safe_FIXED_PATCHED_v2.py"
set "BENCH=%PROJ%\bench_cli_denoise.py"
set "PTCLI=%PROJ%\batch_denoise_pytorch_v3.py"
set "ONNXCLI=%PROJ%\batch_denoise_onnx.py"

rem virtualenv python (fallback to system python if missing)
set "PYTHON=C:\ESPI_VENV2\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [WARN] venv python not found: "%PYTHON%"
  set "PYTHON=python"
)

rem ====== sanity checks ======
if not exist "%PROJ%"  (echo [ERR] Project folder not found: "%PROJ%"& pause & exit /b 1)
if not exist "%AVERAGED%" (echo [ERR] Averaged folder not found: "%AVERAGED%"& pause & exit /b 1)
if not exist "%TRAIN%" (echo [ERR] Training/Export script not found: "%TRAIN%"& pause & exit /b 1)
if not exist "%PTCLI%" (echo [ERR] PyTorch batch CLI not found: "%PTCLI%"& pause & exit /b 1)
if not exist "%ONNXCLI%" (echo [ERR] ONNX batch CLI not found: "%ONNXCLI%"& pause & exit /b 1)
if not exist "%BENCH%" (echo [ERR] Benchmark script not found: "%BENCH%"& pause & exit /b 1)

cd /d "%PROJ%"

rem ====== 1) FREEZE ENV ======
echo.
echo === [1/3] FREEZE ENVIRONMENT ===
"%PYTHON%" -V > "%PROJ%\environment_info.txt"
"%PYTHON%" -m pip freeze > "%PROJ%\requirements.txt"
"%PYTHON%" -c "import platform, sys; print('Python:', platform.python_version()); print('Platform:', platform.platform())" >> "%PROJ%\environment_info.txt"
"%PYTHON%" -c "import importlib, sys; 
try:
 import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())
except Exception as e:
 print('Torch: not installed or failed to import:', e)" >> "%PROJ%\environment_info.txt"
echo [OK] Wrote environment_info.txt and requirements.txt
echo.

rem ====== 2) EXPORT ONNX ======
echo === [2/3] EXPORT ONNX ===
if not exist "%CKPT%" (
  echo [WARN] best.pth not found at "%CKPT%". Will try --resume auto; ensure checkpoints exist under outputs_W01\checkpoints.
)
echo Running export via training script (resume auto)...
"%PYTHON%" "%TRAIN%" --device cpu --clean-root "%AVERAGED%" --output-dir "%ROOT%\outputs_W01" --epochs 1 --resume auto --export-onnx "%ONNX%"
if errorlevel 1 (
  echo [WARN] Export command returned non-zero exit code. Check messages above.
) else (
  echo [OK] Export attempted. ONNX target: "%ONNX%"
)
echo.

rem ====== 3) BENCHMARK ======
echo === [3/3] BENCHMARK PyTorch vs ONNX ===
set "TMP=%ROOT%\bench_tmp"
if exist "%TMP%" rmdir /s /q "%TMP%"
mkdir "%TMP%"
echo Running benchmark on a subset (limit=30) with tile=224, overlap=32 ...
"%PYTHON%" "%BENCH%" --pytorch "%PTCLI%" --onnx "%ONNXCLI%" --ckpt "%CKPT%" --onnx-model "%ONNX%" --input "%AVERAGED%" --tmp-root "%TMP%" --tile 224 --overlap 32 --limit 30 --device cpu 1> "%ROOT%\bench_results.txt" 2>&1
type "%ROOT%\bench_results.txt"

echo.
echo === DONE ===
echo - Env files:        %PROJ%\environment_info.txt , %PROJ%\requirements.txt
echo - ONNX (target):    %ONNX%
echo - Bench results:    %ROOT%\bench_results.txt
echo.
pause
