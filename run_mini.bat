@echo off
set PY=C:\ESPI_VENV2\Scripts\python.exe
%PY% C:\ESPI_DnCNN\run_espi_pipeline.py --stages features,train --run
