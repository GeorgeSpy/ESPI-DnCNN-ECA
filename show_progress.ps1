# Live progress & mini-ETA
function Show-Progress($root){
  if(!(Test-Path $root)){ Write-Host "${root}: NOT FOUND"; return }
  
  $all = (Get-ChildItem -Directory $root | Where-Object { $_.Name -notlike "*_viz" }).Count
  $done = (Get-ChildItem -Directory $root | Where-Object {
    Test-Path (Join-Path $_.FullName "phase_unwrapped_npy")
  }).Count
  
  $pct = if($all -gt 0){ ($done/[double]$all) } else { 0 }
  "{0}: {1}/{2} ({3:P1})" -f (Split-Path $root -Leaf),$done,$all,$pct
}

Write-Host "=== PHASE EXTRACTION PROGRESS ==="
Show-Progress "C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100"
Show-Progress "C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100"
Show-Progress "C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"

# QC/IRLS/Features progress
function Show-QCProgress($root){
  if(!(Test-Path $root)){ Write-Host "${root}: NOT FOUND"; return }
  
  $phasesDone = (Get-ChildItem -Directory $root | Where-Object {
    Test-Path (Join-Path $_.FullName "phase_unwrapped_npy")
  }).Count
  
  $qcDone = (Get-ChildItem -Directory $root | Where-Object {
    Test-Path (Join-Path $_.FullName ".qc.done")
  }).Count
  
  $pct = if($phasesDone -gt 0){ ($qcDone/[double]$phasesDone) } else { 0 }
  "{0}: {1}/{2} QC complete ({3:P1})" -f (Split-Path $root -Leaf),$qcDone,$phasesDone,$pct
}

Write-Host "`n=== QC/IRLS/FEATURES PROGRESS ==="
Show-QCProgress "C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100"
Show-QCProgress "C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100"
Show-QCProgress "C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"

# Disk space check
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "`n=== DISK SPACE ==="
Write-Host "C drive Free: $freeGB GB"
if($freeGB -lt 10){
  Write-Host "WARNING: Low disk space!" -ForegroundColor Yellow
}

