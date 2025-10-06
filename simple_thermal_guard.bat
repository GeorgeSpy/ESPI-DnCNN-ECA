@echo off
REM Simple Thermal Guard - CPU Usage Based Estimation
title Simple Thermal Guard - CPU Usage Estimation

echo =====================================
echo   SIMPLE THERMAL GUARD (CPU USAGE)
echo =====================================
echo Temperature limit: 77°C
echo Estimation: 20°C + (CPU%% × 0.4)
echo Check interval: 5 seconds
echo.
echo Compare readings with Core Temp for calibration!
echo.
echo Starting monitor...
echo Press Ctrl+C to stop
echo.

C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\simple_thermal_guard.py --temp-limit 77 --check-interval 5

echo.
echo Thermal guard stopped.
pause