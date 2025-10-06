<# 
  disk_autoguard.ps1
  Auto cleanup for ESPI_TEMP space management
  Usage:
    powershell -ExecutionPolicy Bypass -File C:\ESPI_DnCNN\disk_autoguard.ps1 `
      -Roots "C:\ESPI_TEMP\GPU_FULL2" -MinFreeGB 8 -TargetFreeGB 15 -IntervalSec 120

  Actions (in order):
    1) Delete phase_*_png / quality_png & *_viz dirs
    2) Delete CLEAN_u16 freq dirs that have .qc.done
    3) Run NTFS compact on NPY for ~30-60% compression
#>

param(
  [string[]]$Roots = @("C:\ESPI_TEMP\GPU_FULL2"),
  [int]$MinFreeGB = 8,
  [int]$TargetFreeGB = 15,
  [int]$IntervalSec = 120,
  [switch]$Once,
  [switch]$DryRun
)

$ErrorActionPreference = "SilentlyContinue"
Set-StrictMode -Version 2.0

$LogDir  = "C:\ESPI_TEMP\autoguard_logs"
$LogFile = Join-Path $LogDir "disk_autoguard.log"
$LockFile = "C:\ESPI_TEMP\.disk_autoguard.lock"

function New-Dir($p){ if(!(Test-Path $p)){ New-Item -ItemType Directory -Path $p | Out-Null } }
New-Dir $LogDir

function Write-Log([string]$msg){
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $line = "[$ts] $msg"
  $line | Out-File -FilePath $LogFile -Encoding UTF8 -Append
  Write-Host $line
}

function Get-FreeGB(){
  try {
    return [math]::Round((Get-PSDrive -Name C).Free/1GB,2)
  } catch { return 0 }
}

function Remove-IfExists([string]$path){
  if(Test-Path $path){
    if($DryRun){ Write-Log "DRYRUN: Remove-Item $path" }
    else{ Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue }
  }
}

function Sweep-PNG-and-Viz([string]$root){
  $freed = 0
  Write-Log "Sweep-PNG-and-Viz @ $root"
  $pngDirs = Get-ChildItem $root -Directory -Recurse |
    Where-Object { $_.Name -in @("phase_wrapped_png","phase_unwrapped_png","quality_png") }
  foreach($d in $pngDirs){
    $size = (Get-ChildItem $d.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    Remove-IfExists $d.FullName
    $freed += [math]::Round($size/1GB,3)
  }
  $vizDirs = Get-ChildItem $root -Directory -Recurse |
    Where-Object { $_.Name -like "*_viz" }
  foreach($d in $vizDirs){
    $size = (Get-ChildItem $d.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    Remove-IfExists $d.FullName
    $freed += [math]::Round($size/1GB,3)
  }
  Write-Log ("Sweep-PNG-and-Viz freed ~{0:N3} GB" -f $freed)
  return $freed
}

function Get-PrefixFromPhaseOutName([string]$phaseDirName){
  if($phaseDirName -match "^(W\d{2})_"){ return $Matches[1] } else { return $null }
}

function Sweep-CLEAN-after-QC([string]$root){
  $freed = 0
  Write-Log "Sweep-CLEAN-after-QC @ $root"

  $phaseOuts = Get-ChildItem $root -Directory |
    Where-Object { $_.Name -like "W*_PhaseOut_*" -and $_.Name -notlike "*_viz" }

  foreach($po in $phaseOuts){
    $prefix = Get-PrefixFromPhaseOutName $po.Name
    if(-not $prefix){ continue }
    $cleanRoot = Join-Path $root ($prefix + "_CLEAN_u16")
    if(!(Test-Path $cleanRoot)){ continue }

    $freqDirs = Get-ChildItem $po.FullName -Directory | Where-Object {
      Test-Path (Join-Path $_.FullName ".qc.done")
    }
    foreach($fd in $freqDirs){
      $target = Join-Path $cleanRoot $fd.Name
      if(Test-Path $target){
        $size = (Get-ChildItem $target -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        if($DryRun){ Write-Log "DRYRUN: Remove CLEAN_u16 '$target'" }
        else{ Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue }
        $freed += [math]::Round($size/1GB,3)
      }
    }
  }
  Write-Log ("Sweep-CLEAN-after-QC freed ~{0:N3} GB" -f $freed)
  return $freed
}

function Compact-NPY([string]$root){
  Write-Log "Compact-NPY @ $root (NTFS)"
  if($DryRun){ Write-Log "DRYRUN: compact /c /s:`"$root`" /i" ; return 0 }
  & compact /c /s:"$root" /i | Out-Null
  return 0
}

function Single-Pass(){
  $freeBefore = Get-FreeGB
  Write-Log ("Free before: {0:N2} GB" -f $freeBefore)

  if($freeBefore -ge $TargetFreeGB){
    Write-Log "Target already met. Nothing to do."
    return
  }

  $totalFreed = 0.0

  foreach($r in $Roots){
    if(!(Test-Path $r)){ continue }

    $totalFreed += Sweep-PNG-and-Viz $r
    if((Get-FreeGB) -ge $TargetFreeGB){ break }

    $totalFreed += Sweep-CLEAN-after-QC $r
    if((Get-FreeGB) -ge $TargetFreeGB){ break }

    $pre = Get-FreeGB
    Compact-NPY $r
    $post = Get-FreeGB
    $delta = [math]::Max(0, $post - $pre)
    if($delta -gt 0){ Write-Log ("Compact gain ~{0:N2} GB" -f $delta) ; $totalFreed += $delta }
    if($post -ge $TargetFreeGB){ break }
  }

  $freeAfter = Get-FreeGB
  Write-Log ("Free after : {0:N2} GB  (net +{1:N2} GB)" -f $freeAfter, ($freeAfter - $freeBefore))
}

if(Test-Path $LockFile){
  $ageMin = ((Get-Date) - (Get-Item $LockFile).LastWriteTime).TotalMinutes
  if($ageMin -lt 10){
    Write-Log "Another instance running (lock age less than 10 min). Exit."
    exit 0
  } else {
    Write-Log "Found old lock file (age greater than 10 min). Will replace."
    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
  }
}
Set-Content -Path $LockFile -Value ([System.Diagnostics.Process]::GetCurrentProcess().Id)

try{
  if($Once){
    Single-Pass
  } else {
    while($true){
      $free = Get-FreeGB
      if($free -lt $MinFreeGB){
        Write-Log ("Low space: {0:N2} GB less than {1} GB. Running sweeps..." -f $free, $MinFreeGB)
        Single-Pass
      } else {
        Write-Log ("OK space: {0:N2} GB (waiting {1}s)..." -f $free, $IntervalSec)
      }
      Start-Sleep -Seconds $IntervalSec
    }
  }
}
finally{
  Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
  Write-Log "disk_autoguard: completed."
}
