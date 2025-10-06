@echo off
setlocal ENABLEDELAYEDEXPANSION
title ESPI PseudoNoise + Baseline + Fine-Tune (One-Click)

rem ======= EDIT THESE PATHS IF NEEDED =======
set "PROJ=C:\ESPI_DnCNN"
set "ROOT=C:\Users\Geiorge HTPC\Desktop\Mpouzoukia"

set "AVERAGED=%ROOT%\W01_ESPI_90db-Averaged"
set "PSEUDO=%ROOT%\W01_ESPI_90db-PseudoNoisy"
set "OUTDIR=%ROOT%\outputs_W01_paired_v2"
set "CKPT=%ROOT%\outputs_W01\checkpoints\best.pth"   rem optional resume

set "PYTHON=C:\ESPI_VENV2\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "MAKE=%PROJ%\make_pseudo_noisy_v2.py"
set "SANITY=%PROJ%\paired_sanity_check.py"
set "BASEPSNR=%PROJ%\dataset_baseline_psnr.py"
set "FINETUNE=%PROJ%\fine_tune_paired_min_v2.py"

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
echo ===== [1/3] Generate Pseudo-Noisy (Light) =====
if not exist "%PSEUDO%" mkdir "%PSEUDO%"
"%PYTHON%" "%MAKE%" --input "%AVERAGED%" --output "%PSEUDO%" --speckle-k 6.0 --speckle-theta 0.3 --poisson-gain 0.7 --gauss-sigma 0.01 --vignette 0.05 --jitter 0
if errorlevel 1 (echo [WARN] Pseudo generator returned non-zero rc.)

echo.
echo ===== [1.5/3] Pair Sanity & Baseline PSNR =====
"%PYTHON%" "%SANITY%" --clean "%AVERAGED%" --noisy "%PSEUDO%"
"%PYTHON%" "%BASEPSNR%" --noisy "%PSEUDO%" --clean "%AVERAGED%" --limit 200

echo.
echo ===== [2/3] Fine-Tune v2 (CPU-safe, BN frozen) =====
set "RESUME_ARG="
if exist "%CKPT%" set RESUME_ARG=--resume "%CKPT%"
"%PYTHON%" "%FINETUNE%" --noisy-root "%PSEUDO%" --clean-root "%AVERAGED%" --outdir "%OUTDIR%" --epochs 30 --batch-size 2 --workers 0 --crop 224 --lr 3e-5 --device cpu --freeze-bn %RESUME_ARG%
if errorlevel 1 echo [WARN] Fine-tune returned non-zero rc.

echo.
echo ===== DONE =====
echo Outputs:
echo  - Pseudo noisy:   %PSEUDO%
echo  - Fine-tune out:  %OUTDIR%
echo  - Samples:        %OUTDIR%\samples
echo Re-run phase/QC on a few frames to verify downstream improvement.
pause
