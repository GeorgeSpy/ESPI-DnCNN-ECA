param(
  [string]$SrcRoot = "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia",
  [string]$DestRoot = "D:\temp",
  [switch]$DoIt
)

# ---- config ----
$DryRun = -not $DoIt
function New-ArchiveRoot {
  $ts = Get-Date -Format "yyyyMMdd_HHmm"
  $arch = Join-Path $DestRoot "ESPI_ARCHIVE_$ts"
  New-Item -ItemType Directory -Force -Path $arch | Out-Null
  return $arch
}
function Ensure-Parent($path) {
  $parent = Split-Path $path -Parent
  if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}
function Move-Safe($srcPath, $archRoot) {
  if (-not (Test-Path $srcPath)) { Write-Host "  [skip] missing: $srcPath"; return }
  $abs = (Resolve-Path $srcPath).Path
  $rel = $abs.Substring($SrcRoot.Length).TrimStart('\')
  if ([string]::IsNullOrWhiteSpace($rel)) { $rel = Split-Path $abs -Leaf }
  $dst = Join-Path $archRoot $rel
  Ensure-Parent $dst
  if ((Get-Item $srcPath) -is [System.IO.DirectoryInfo]) {
    Write-Host "  [dir]  $rel"
  } else {
    Write-Host "  [file] $rel"
  }
  if ($DryRun) {
    Move-Item -Path $srcPath -Destination $dst -Force -ErrorAction Continue -WhatIf
  } else {
    Move-Item -Path $srcPath -Destination $dst -Force -ErrorAction Continue
  }
}

# --- what we KEEP (μην τα πειράξεις) ---
$keep = @(
  "W01_PhaseOut_DEN_b18_cs16_ff100",
  "roi_mask.png",
  "W01_modes",
  "features",
  "README.md","environment_info.txt","config.yaml"
)

# --- folders to ARCHIVE (ενδιάμεσα / βαριά / debug) ---
$foldersToArchive = @(
  # denoise outputs / intermediates
  "W01_ESPI_90db-CLEAN_fromRAW",
  "W01_ESPI_90db-CLEAN_fromRAW_masked",
  "W01_ESPI_90db-CLEAN_fromRAW_masked_v2",
  "W01_ESPI_90db-CLEAN_RAW_minus",
  "_RAW_RES_NPY",
  # παλιά phase variants (δεν χρησιμοποιούνται)
  "W01_PhaseOut_STRICT",
  "W01_PhaseOut_STRICT_DEN",
  "W01_PhaseOut_STRICT_DEN_CLEAN",
  "W01_PhaseOut_STRICT_DEN_CLEAN_v2",
  "W01_PhaseOut_masked_grid",
  # training outputs / debug / tmp
  "outputs_W01",
  "outputs_W01_paired",
  "outputs_W01_paired_v3",
  "outputs_W01_paired_v4",
  "outputs_W01_paired_v4_b",
  "outputs_W01_paired_v4_MATCH_mix",
  "PAIR_DEBUG_v3b",
  "bench_tmp",
  "FFT_Check",
  "FFT_Check_520",
  "FFT_Check_ROI",
  "_DEBUG_dualmode",
  "_DEBUG_identity",
  "_QC_maps",
  # masked averaged (ξαναγίνονται εύκολα)
  "W01_ESPI_90db-Averaged_masked",
  "W01_ESPI_90db-Averaged_masked_v2"
) | Where-Object { $_ -and (Test-Path (Join-Path $SrcRoot $_)) }

# --- loose file patterns to ARCHIVE (παντού κάτω από $SrcRoot) ---
$filePatterns = @(
  "*wrapped*.png", "*wrapped*.npy",
  "*.tmp", "*.bak", "*.log"
)

# --- averaged duplicates to ARCHIVE ---
$avgRoots = Get-ChildItem $SrcRoot -Directory -Recurse | Where-Object { $_.Name -like "*Averaged*" }
$avgDupPatterns = @("*db1.png","*_1075.png","*1005.png")

Write-Host "=== ESPI cleanup mover ==="
Write-Host ("Source:      {0}" -f $SrcRoot)
Write-Host ("Destination: {0}" -f $DestRoot)
$mode = if ($DryRun) { "DRY-RUN (preview)" } else { "EXECUTE" }
Write-Host ("Mode:        {0}" -f $mode)

$archRoot = New-ArchiveRoot

# 1) move listed folders
if ($foldersToArchive.Count -gt 0) {
  Write-Host "`n[Step 1] Folders to archive:"
  foreach ($rel in $foldersToArchive) {
    $p = Join-Path $SrcRoot $rel
    if ($keep | Where-Object { $rel -ieq $_ }) { Write-Host "  [keep] $rel"; continue }
    Move-Safe $p $archRoot
  }
} else {
  Write-Host "`n[Step 1] No known folders to archive were found under $SrcRoot."
}

# 2) move wrapped/QC/etc files by pattern
Write-Host "`n[Step 2] File patterns:"
foreach ($pat in $filePatterns) {
  $items = Get-ChildItem -Path $SrcRoot -Recurse -File -Filter $pat -ErrorAction SilentlyContinue
  foreach ($it in $items) { Move-Safe $it.FullName $archRoot }
}

# 3) averaged duplicates
Write-Host "`n[Step 3] Averaged duplicates:"
foreach ($avg in $avgRoots) {
  foreach ($pat in $avgDupPatterns) {
    $dups = Get-ChildItem -Path $avg.FullName -Recurse -File -Filter $pat -ErrorAction SilentlyContinue
    foreach ($f in $dups) { Move-Safe $f.FullName $archRoot }
  }
}

# 4) summary
Write-Host "`n[Summary] Archive folder prepared at: $archRoot"
if ($DryRun) {
  Write-Host "Dry-run completed. Re-run with -DoIt to actually move the files."
}
