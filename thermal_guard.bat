@echo off
REM CPU Thermal Guard - Windows Batch Launcher (Updated)
REM This script starts the thermal monitoring with Core Temp integration

title CPU Thermal Guard - Core Temp Integration
echo ===================================
echo   CPU THERMAL GUARD - CORE TEMP
echo ===================================
echo Temperature limit: 77°C
echo Check interval: 5 seconds
echo.
echo [1/3] Installing required packages...
C:\ESPI_VENV2\Scripts\pip.exe install psutil wmi pywin32
echo.
echo [2/3] Checking Core Temp...
echo Make sure Core Temp is running for accurate readings!
echo.
echo [3/3] Starting thermal monitor...
echo Press Ctrl+C to stop monitoring
echo.

C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\cpu_thermal_guard.py --temp-limit 77 --check-interval 5

echo.
echo Thermal guard stopped.
pause