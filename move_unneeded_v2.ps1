param(
  [string]$SrcRoot = "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia",
  [string]$DestRoot = "D:\temp",
  [string]$ArchiveRoot,         # προαιρετικά: χρησιμοποίησε συγκεκριμένο ESPI_ARCHIVE_*
  [switch]$DoIt,                # όταν μπει, εκτελεί μεταφορές
  [switch]$ReportOnly,          # μόνο αναφορά, καμία μεταφορά
  [int]$MinFreeMB = 2048        # stop αν ο προορισμός πέσει κάτω από αυτό το free space
)

# ---------- helpers ----------
function New-ArchiveRoot {
  if ($ArchiveRoot) { return $ArchiveRoot }
  $ts = Get-Date -Format "yyyyMMdd_HHmm"
  $arch = Join-Path $DestRoot "ESPI_ARCHIVE_$ts"
  New-Item -ItemType Directory -Force -Path $arch | Out-Null
  return $arch
}
function Ensure-Parent([string]$path) {
  $parent = Split-Path $path -Parent
  if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}
function Get-FreeBytes([string]$path) {
  $root = (Split-Path $path -Qualifier)
  if (-not $root.EndsWith("\")) { $root = $root + "\" }
  $di = New-Object System.IO.DriveInfo($root)
  return [int64]$di.AvailableFreeSpace
}
function Size-OfItem([string]$p) {
  if (Test-Path $p) {
    $it = Get-Item $p -ErrorAction SilentlyContinue
    if ($it -is [System.IO.DirectoryInfo]) {
      $s = (Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
      return [int64]($s)
    } else {
      return [int64]$it.Length
    }
  } else { return 0 }
}
function Rel-FromSrc([string]$abs, [string]$src) {
  $abs = (Resolve-Path $abs).Path
  if ($abs.StartsWith($src)) { return $abs.Substring($src.Length).TrimStart('\') }
  return (Split-Path $abs -Leaf)
}
function Move-One([string]$srcPath, [string]$archRoot, [bool]$execute) {
  if (-not (Test-Path $srcPath)) { return $false }
  $rel = Rel-FromSrc $srcPath $SrcRoot
  $dst = Join-Path $archRoot $rel
  Ensure-Parent $dst
  if ($execute) {
    Move-Item -Path $srcPath -Destination $dst -Force -ErrorAction Stop
  } else {
    Move-Item -Path $srcPath -Destination $dst -Force -WhatIf -ErrorAction SilentlyContinue
  }
  return $true
}

# ---------- WHAT stays / WHAT goes (όπως στο παλιό script) ----------
$keep = @(
  "W01_PhaseOut_DEN_b18_cs16_ff100",
  "roi_mask.png",
  "W01_modes",
  "features",
  "README.md","environment_info.txt","config.yaml"
)

$foldersToArchive = @(
  "W01_ESPI_90db-CLEAN_fromRAW",
  "W01_ESPI_90db-CLEAN_fromRAW_masked",
  "W01_ESPI_90db-CLEAN_fromRAW_masked_v2",
  "W01_ESPI_90db-CLEAN_RAW_minus",
  "_RAW_RES_NPY",
  "W01_PhaseOut_STRICT",
  "W01_PhaseOut_STRICT_DEN",
  "W01_PhaseOut_STRICT_DEN_CLEAN",
  "W01_PhaseOut_STRICT_DEN_CLEAN_v2",
  "W01_PhaseOut_masked_grid",
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
  "W01_ESPI_90db-Averaged_masked",
  "W01_ESPI_90db-Averaged_masked_v2"
) | Where-Object { $_ -and (Test-Path (Join-Path $SrcRoot $_)) }

$filePatterns = @(
  "*wrapped*.png", "*wrapped*.npy",
  "*.tmp", "*.bak", "*.log"
)

$avgRoots = Get-ChildItem $SrcRoot -Directory -Recurse | Where-Object { $_.Name -like "*Averaged*" }
$avgDupPatterns = @("*db1.png","*_1075.png","*1005.png")

Write-Host "=== ESPI mover v2 ==="
Write-Host ("Source:      {0}" -f $SrcRoot)
Write-Host ("Destination: {0}" -f $DestRoot)
Write-Host ("Mode:        {0}" -f ($(if ($ReportOnly) {"REPORT-ONLY"} elseif ($DoIt) {"EXECUTE"} else {"DRY-RUN"})))
Write-Host ("MinFreeMB:   {0}" -f $MinFreeMB)

# ---------- build candidate list ----------
$candidates = @()

# 1) listed folders
foreach ($rel in $foldersToArchive) {
  if ($keep | Where-Object { $rel -ieq $_ }) { continue }
  $p = Join-Path $SrcRoot $rel
  if (Test-Path $p) {
    $candidates += [pscustomobject]@{ Type="Dir"; Src=$p; Size= (Size-OfItem $p) }
  }
}

# 2) file patterns
foreach ($pat in $filePatterns) {
  $items = Get-ChildItem -Path $SrcRoot -Recurse -File -Filter $pat -ErrorAction SilentlyContinue
  foreach ($it in $items) {
    $candidates += [pscustomobject]@{ Type="File"; Src=$it.FullName; Size=[int64]$it.Length }
  }
}

# 3) averaged duplicates
foreach ($avg in $avgRoots) {
  foreach ($pat in $avgDupPatterns) {
    $dups = Get-ChildItem -Path $avg.FullName -Recurse -File -Filter $pat -ErrorAction SilentlyContinue
    foreach ($f in $dups) {
      $candidates += [pscustomobject]@{ Type="File"; Src=$f.FullName; Size=[int64](Get-Item $f.FullName).Length }
    }
  }
}

# de-dup by Src
$candidates = $candidates | Group-Object Src | ForEach-Object { $_.Group | Select-Object -First 1 }

# pending = όσοι υπάρχουν ακόμα στη πηγή
$pending = $candidates | Where-Object { Test-Path $_.Src }

# archive root (για logs/outputs)
$archRoot = if ($ArchiveRoot) { $ArchiveRoot } else {
  if ($ReportOnly -or -not $DoIt) {
    # Αν υπάρχει ήδη archive από πριν, πάρε το πιο πρόσφατο για report
    $existing = Get-ChildItem $DestRoot -Directory -Filter "ESPI_ARCHIVE_*" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($existing) { $existing.FullName } else { New-ArchiveRoot }
  } else { New-ArchiveRoot }
}

# paths for reports
$repDir = Join-Path $archRoot "_reports"
New-Item -ItemType Directory -Force -Path $repDir | Out-Null
$candCsv   = Join-Path $repDir "candidate_list.csv"
$movedCsv  = Join-Path $repDir "moved_list.csv"
$pendingCsv= Join-Path $repDir "pending_list.csv"

# write candidate list
$candidates | Select-Object Type, Src, Size |
  Export-Csv -NoTypeInformation -Encoding UTF8 $candCsv

# compute moved (ό,τι υπάρχει ήδη κάτω από $archRoot)
$moved = @()
if (Test-Path $archRoot) {
  $movedFiles = Get-ChildItem -Path $archRoot -Recurse -File -ErrorAction SilentlyContinue
  foreach ($mf in $movedFiles) {
    $moved += [pscustomobject]@{ Path=$mf.FullName; Size=[int64]$mf.Length }
  }
  $moved | Export-Csv -NoTypeInformation -Encoding UTF8 $movedCsv
}

# ---- REPORT ONLY ----
if ($ReportOnly) {
  $movedBytes   = ($moved | Measure-Object Size -Sum).Sum
  $pendingBytes = ($pending | Measure-Object Size -Sum).Sum
  $pending | Select-Object Type, Src, Size | Export-Csv -NoTypeInformation -Encoding UTF8 $pendingCsv
  Write-Host ("[REPORT] Candidates: {0}, Pending: {1}, Pending MB: {2:N1}" -f $candidates.Count, $pending.Count, ($pendingBytes/1MB))
  Write-Host ("[REPORT] Moved files in {0}: {1}, Moved MB: {2:N1}" -f $archRoot, $moved.Count, ($movedBytes/1MB))
  Write-Host ("[REPORT] CSVs: `n  - {0}`n  - {1}`n  - {2}" -f $candCsv,$movedCsv,$pendingCsv)
  exit 0
}

# ---- MOVE MODE (DRY-RUN or EXECUTE) ----
$minFreeBytes = [int64]$MinFreeMB * 1MB
$freeBefore = Get-FreeBytes $archRoot
Write-Host ("[INFO] Free space before: {0:N1} MB" -f ($freeBefore/1MB))

$executed = $false
$stopReason = $null

foreach ($it in $pending) {
  $free = Get-FreeBytes $archRoot
  if ($free -lt ($it.Size + $minFreeBytes)) {
    $stopReason = "OUT_OF_SPACE"
    Write-Host ("[STOP] Not enough free space to move next item ({0:N1} MB needed incl. margin)." -f (($it.Size+$minFreeBytes)/1MB)) -ForegroundColor Yellow
    break
  }
  Write-Host ("[MOVE] {0}  {1}  ({2:N1} MB)" -f $it.Type, (Rel-FromSrc $it.Src $SrcRoot), ($it.Size/1MB))
  try {
    $ok = Move-One $it.Src $archRoot $DoIt.IsPresent
    if ($ok) { $executed = $true }
  } catch {
    Write-Host "[ERR] $($_.Exception.Message)" -ForegroundColor Red
  }
}

# recompute pending after move
$pending2 = $candidates | Where-Object { Test-Path $_.Src }
$pending2 | Select-Object Type, Src, Size | Export-Csv -NoTypeInformation -Encoding UTF8 $pendingCsv

$moved = @()
if (Test-Path $archRoot) {
  $movedFiles = Get-ChildItem -Path $archRoot -Recurse -File -ErrorAction SilentlyContinue
  foreach ($mf in $movedFiles) { $moved += [pscustomobject]@{ Path=$mf.FullName; Size=[int64]$mf.Length } }
  $moved | Export-Csv -NoTypeInformation -Encoding UTF8 $movedCsv
}

$freeAfter = Get-FreeBytes $archRoot
$movedBytes = ($moved | Measure-Object Size -Sum).Sum
$pendingBytes = ($pending2 | Measure-Object Size -Sum).Sum

Write-Host ("[SUMMARY] Moved MB: {0:N1}   Pending MB: {1:N1}   Free after: {2:N1} MB" -f ($movedBytes/1MB), ($pendingBytes/1MB), ($freeAfter/1MB))
if ($stopReason) { Write-Host ("[SUMMARY] Stop reason: {0}" -f $stopReason) -ForegroundColor Yellow }
Write-Host ("[FILES] Reports: `n  - {0}`n  - {1}`n  - {2}" -f $candCsv,$movedCsv,$pendingCsv)
