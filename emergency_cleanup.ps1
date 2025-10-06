# Emergency Cleanup - Run when disk is critically low
# This script follows the priority order for safe deletion

param([switch]$DryRun)

Write-Host "=== EMERGENCY CLEANUP ===" -ForegroundColor Red
if ($DryRun) { Write-Host "DRY RUN MODE - No files will be deleted" -ForegroundColor Yellow }

$freed = 0

# Priority 1: Delete _viz directories
Write-Host "`n[Priority 1] Deleting _viz directories..." -ForegroundColor Cyan
$vizDirs = Get-ChildItem "C:\ESPI_TEMP\GPU_FULL2" -Directory -Recurse | Where-Object { $_.Name -like "*_viz" }
foreach ($d in $vizDirs) {
    $size = (Get-ChildItem $d.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($DryRun) {
        Write-Host "  Would delete: $($d.FullName) (~$([math]::Round($size/1GB,2)) GB)"
    } else {
        Remove-Item $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $freed += $size
    }
}

# Priority 2: Delete PNG directories
Write-Host "`n[Priority 2] Deleting PNG directories..." -ForegroundColor Cyan
$pngDirs = Get-ChildItem "C:\ESPI_TEMP\GPU_FULL2" -Directory -Recurse |
    Where-Object { $_.Name -in @("phase_wrapped_png","phase_unwrapped_png","quality_png") }
foreach ($d in $pngDirs) {
    $size = (Get-ChildItem $d.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($DryRun) {
        Write-Host "  Would delete: $($d.FullName) (~$([math]::Round($size/1GB,2)) GB)"
    } else {
        Remove-Item $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $freed += $size
    }
}

# Priority 3: Delete CLEAN_u16 dirs with .qc.done
Write-Host "`n[Priority 3] Deleting completed CLEAN_u16 directories..." -ForegroundColor Cyan
$phaseOuts = Get-ChildItem "C:\ESPI_TEMP\GPU_FULL2" -Directory | Where-Object { $_.Name -like "W*_PhaseOut_*" }
foreach ($po in $phaseOuts) {
    $prefix = if ($po.Name -match "^(W\d{2})_") { $Matches[1] } else { $null }
    if (-not $prefix) { continue }
    
    $cleanRoot = "C:\ESPI_TEMP\GPU_FULL2\$($prefix)_CLEAN_u16"
    if (!(Test-Path $cleanRoot)) { continue }
    
    $doneDirs = Get-ChildItem $po.FullName -Directory | Where-Object {
        Test-Path (Join-Path $_.FullName ".qc.done")
    }
    
    foreach ($fd in $doneDirs) {
        $target = Join-Path $cleanRoot $fd.Name
        if (Test-Path $target) {
            $size = (Get-ChildItem $target -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
            if ($DryRun) {
                Write-Host "  Would delete: $target (~$([math]::Round($size/1GB,2)) GB)"
            } else {
                Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
                $freed += $size
            }
        }
    }
}

$freedGB = [math]::Round($freed/1GB,2)
$free = [math]::Round((Get-PSDrive -Name C).Free/1GB,2)

Write-Host "`n=== CLEANUP SUMMARY ===" -ForegroundColor Green
if ($DryRun) {
    Write-Host "Would free: ~$freedGB GB" -ForegroundColor Yellow
} else {
    Write-Host "Freed: $freedGB GB" -ForegroundColor Green
}
Write-Host "Current free space: $free GB" -ForegroundColor Green

