param([switch]$WhatIf)

$Py   = "C:\ESPI_VENV2\Scripts\python.exe"
$Gen  = "C:\ESPI_DnCNN\make_pseudo_noisy_plus.py"   # patched έκδοση με --profile/--bitdepth
$PseudoRoot  = "C:\ESPI_TEMP\pseudo"
$MetricsRoot = "C:\ESPI_TEMP\pseudo\metrics"

$Jobs = @(
  @{Inst="W01"; Avg="C:\ESPI\data\wood_Averaged\W01_ESPI_90db-Averaged"},
  @{Inst="W02"; Avg="C:\ESPI\data\wood_Averaged\W02_ESPI_90db-Averaged"},
  @{Inst="W03"; Avg="C:\ESPI\data\wood_Averaged\W03_ESPI_90db-Averaged"}
)
$Profiles = @("lite","mid","heavy")
$Seeds    = 1..3

function Is-RunDone($outFolder,$metricsCsv){
  if (Test-Path $metricsCsv){
    try {
      $n=(Get-Content $metricsCsv -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
      if ($n -ge 200){ return $true }
    } catch {}
  }
  if (Test-Path $outFolder){
    $n=(Get-ChildItem $outFolder -Filter *.png -File -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($n -ge 200){ return $true }
  }
  return $false
}

# Preflight
New-Item -ItemType Directory -Force -Path $PseudoRoot,$MetricsRoot | Out-Null
foreach ($j in $Jobs){
  if (!(Test-Path $j.Avg)){
    Write-Host "[ERROR] Missing averaged root: $($j.Avg)" -ForegroundColor Red
  } else {
    $cnt=(Get-ChildItem $j.Avg -Filter *.png -File | Measure-Object).Count
    Write-Host ("[OK] {0} averaged: {1} png" -f $j.Inst,$cnt)
  }
}

# Resume loop
foreach ($j in $Jobs){
  foreach ($p in $Profiles){
    foreach ($s in $Seeds){
      $Out  = Join-Path $PseudoRoot ("{0}\Pseudo_{1}_s{2}" -f $j.Inst,$p,$s)
      $Mcsv = Join-Path $MetricsRoot ("{0}_{1}_s{2}.csv" -f $j.Inst,$p,$s)

      if (Is-RunDone $Out $Mcsv){
        Write-Host "[SKIP] $($j.Inst) $p s$s (already done)" -ForegroundColor Yellow
        continue
      }

      Write-Host "[RUN]  $($j.Inst) $p s$s -> $Out" -ForegroundColor Cyan
      if ($WhatIf){ continue }

      New-Item -ItemType Directory -Force -Path $Out | Out-Null
      $log = Join-Path $Out "generate.log"

      try {
        & $Py $Gen --input $j.Avg --output $Out --profile $p --seed $s --bitdepth 16 --export-metrics $Mcsv *>&1 |
          Tee-Object -FilePath $log
      } catch {
        Write-Host "[ERR]  $($j.Inst) $p s$s : $($_.Exception.Message)" -ForegroundColor Red
      }
    }
  }
}

""; "=== SUMMARY ==="
Get-ChildItem "$PseudoRoot\W*\Pseudo_*" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $n=(Get-ChildItem $_ -Filter *.png -File -ErrorAction SilentlyContinue | Measure-Object).Count
  "{0} -> {1} png" -f $_.FullName,$n
}
