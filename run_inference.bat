@echo off
setlocal enabledelayedexpansion

REM ====== CONFIG ======
set VENV_PY=C:\ESPI_VENV2\Scripts\python.exe
set MASK=C:\ESPI_TEMP\roi_mask.png
set MODEL=C:\ESPI_TEMP\features\baseline_nodal_model_merged.joblib

if "%~1"=="" (
  echo Usage: %~nx0 "C:\path\to\INTENSITY_DIR"
  exit /b 1
)
set INP=%~1

REM make run folder with timestamp-safe name
for /f "tokens=1-4 delims=/:. " %%a in ("%date% %time%") do set TS=%%d%%b%%c_%%e%%f
set RUN=C:\ESPI_TEMP\run_%TS%
mkdir "%RUN%" 2>nul
mkdir "%RUN%\masked_v2" "%RUN%\phase" "%RUN%\features" "%RUN%\infer" 2>nul

echo [1/5] mask+norm -> masked_v2
"%VENV_PY%" C:\ESPI_DnCNN\mask_and_norm_v2.py --in-dir "%INP%" --out-dir "%RUN%\masked_v2" --mask "%MASK%"
if errorlevel 1 goto :fail

echo [2/5] phase extraction (Preset A) -> phase
"%VENV_PY%" C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py ^
  --input-dir  "%RUN%\masked_v2" ^
  --output-dir "%RUN%\phase" ^
  --band 18 --center-suppress 16 --flatfield 100 --annulus 8 300 ^
  --roi-mask "%MASK%" --unwrap auto
if errorlevel 1 goto :fail

echo [3/5] features -> CSV
"%VENV_PY%" C:\ESPI_DnCNN\espi_features_nodal.py ^
  --phase-root "%RUN%\phase" ^
  --roi-mask   "%MASK%" ^
  --out-csv    "%RUN%\features\features.csv"
if errorlevel 1 goto :fail

echo [4/5] dedup features (png+npy) -> features_dedup.csv
"%VENV_PY%" C:\ESPI_DnCNN\dedup_features_by_name.py --in "%RUN%\features\features.csv" --out "%RUN%\features\features_dedup.csv"
if errorlevel 1 goto :fail

echo [5/5] predict -> predictions.csv (+probabilities)
"%VENV_PY%" C:\ESPI_DnCNN\predict_from_features.py ^
  --feats   "%RUN%\features\features_dedup.csv" ^
  --model   "%MODEL%" ^
  --out-csv "%RUN%\infer\predictions.csv"
if errorlevel 1 goto :fail

echo.
echo [OK] Done. Predictions: "%RUN%\infer\predictions.csv"
start notepad "%RUN%\infer\predictions.csv"
exit /b 0

:fail
echo.
echo [ERROR] Pipeline failed. Check the last step's message above.
exit /b 2
