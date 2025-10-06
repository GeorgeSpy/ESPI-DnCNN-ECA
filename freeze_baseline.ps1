# Usage:
#   powershell -ExecutionPolicy Bypass -File .\freeze_baseline.ps1
$ts = Get-Date -Format "yyyyMMdd-HHmm"
$dst = "C:\ESPI_TEMP\FREEZE_$ts"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
New-Item -ItemType Directory -Force -Path "$dst\models" | Out-Null
New-Item -ItemType Directory -Force -Path "$dst\features" | Out-Null

# probable locations
$ck1 = "C:\ESPI_TEMP\denoise_finetune_GPU_FULLSET\checkpoints\best.pth"
$f_all = "C:\ESPI_TEMP\features\FINAL_all_features_merged.csv"
if (!(Test-Path $f_all)) { $f_all = "C:\ESPI_TEMP\features\all_features_merged.csv" }
$labels = "C:\ESPI_TEMP\features\FINAL_labels_fixed_bins.csv"
if (!(Test-Path $labels)) { $labels = "C:\ESPI_TEMP\features\labels_fixed_bins.csv" }
$labelmap = "C:\ESPI_TEMP\label_map.json"
$report = "C:\ESPI_TEMP\THESIS_PACKAGE\FINAL_THESIS_REPORT.md"

Copy-Item -Force -ErrorAction SilentlyContinue $ck1 "$dst\models\best.pth"
Copy-Item -Force -ErrorAction SilentlyContinue $f_all "$dst\features\all_features_merged.csv"
Copy-Item -Force -ErrorAction SilentlyContinue $labels "$dst\features\labels_fixed_bins.csv"
Copy-Item -Force -ErrorAction SilentlyContinue $labelmap "$dst\label_map.json"
if (Test-Path $report) { Copy-Item -Force $report "$dst\FINAL_THESIS_REPORT.md" }

# write a tiny manifest
$manifest = @{}
$manifest.ts = $ts
$manifest.best_exists = Test-Path "$dst\models\best.pth"
$manifest.features_rows = (Get-Content "$dst\features\all_features_merged.csv").Length - 1
$manifest.labels_rows = (Get-Content "$dst\features\labels_fixed_bins.csv").Length - 1
$manifest_json = $manifest | ConvertTo-Json
$manifest_json | Out-File -Encoding ASCII "$dst\FREEZE_MANIFEST.json"

Write-Host "Baseline frozen at: $dst"
