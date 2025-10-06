@echo off
setlocal ENABLEDELAYEDEXPANSION
title ESPI PseudoNoise v3 + BaselineSafe + Fine-Tune v4 (One-Click)

rem ======= EDIT THESE PATHS IF NEEDED =======
set "PROJ=C:\ESPI_DnCNN"
set "ROOT=C:\Users\Geiorge HTPC\Desktop\Mpouzoukia"

set "AVERAGED=%ROOT%\W01_ESPI_90db-Averaged"
set "PSEUDO=%ROOT%\W01_ESPI_90db-PseudoNoisy_v3"
set "OUTDIR=%ROOT%\outputs_W01_paired_v4"
set "CKPT=%ROOT%\outputs_W01\checkpoints\best.pth"   rem optional resume

set "PYTHON=C:\ESPI_VENV2\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "MAKE=%PROJ%\make_pseudo_noisy_v3.py"
set "SANITY=%PROJ%\paired_sanity_check.py"
set "BASEPSNR=%PROJ%\dataset_baseline_psnr_safe.py"
set "FINETUNE=%PROJ%\fine_tune_paired_min_v4.py"

rem ======= CHECK FILES =======
if not exist "%PROJ%"   (echo [ERR] Project folder not found: "%PROJ%" & pause & exit /b 1)
if not exist "%AVERAGED%" (echo [ERR] Averaged folder not found: "%AVERAGED%" & pause & exit /b 1)

for %%F in ("%MAKE%" "%SANITY%" "%BASEPSNR%" "%FINETUNE%") do (
  if not exist "%%~F" (
    echo [ERR] Missing script: %%~F
    echo Copy the .py files into %PROJ% and run again.
    pause
    exit /b 1
  )
)

echo.
echo ===== [1/3] Generate Pseudo-Noisy (Gentle) =====
if not exist "%PSEUDO%" mkdir "%PSEUDO%"
"%PYTHON%" "%MAKE%" --input "%AVERAGED%" --output "%PSEUDO%" --speckle-k 25.0 --speckle-theta 0.04 --spk-min 0.92 --spk-max 1.10 --poisson-gain 0.03 --gauss-sigma 0.0015 --vignette 0.01 --jitter 0
if errorlevel 1 (echo [WARN] Pseudo generator returned non-zero rc.)

echo.
echo ===== [1.5/3] Pair Sanity ^& Baseline PSNR (safe) =====
"%PYTHON%" "%SANITY%" --clean "%AVERAGED%" --noisy "%PSEUDO%"
"%PYTHON%" "%BASEPSNR%" --noisy "%PSEUDO%" --clean "%AVERAGED%" --limit 200

echo.
echo ===== [2/3] Fine-Tune v4 (CPU-safe, BN frozen) =====
set "RESUME_ARG="
if exist "%CKPT%" set RESUME_ARG=--resume "%CKPT%"
"%PYTHON%" "%FINETUNE%" --noisy-root "%PSEUDO%" --clean-root "%AVERAGED%" --outdir "%OUTDIR%" --epochs 20 --batch-size 1 --workers 0 --crop 192 --lr 3e-5 --device cpu --freeze-bn %RESUME_ARG%
if errorlevel 1 echo [WARN] Fine-tune returned non-zero rc.

echo.
echo ===== DONE =====
echo Outputs:
echo  - Pseudo noisy:   %PSEUDO%
echo  - Fine-tune out:  %OUTDIR%
echo  - Samples:        %OUTDIR%\samples
pause
