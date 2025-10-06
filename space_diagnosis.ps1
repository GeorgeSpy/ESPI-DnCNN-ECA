# Space diagnosis - Find what's using the most space
Write-Host "=== TOP 25 SPACE CONSUMERS ===" -ForegroundColor Cyan

Get-ChildItem C:\ESPI_TEMP\GPU_FULL2 -Directory -Recurse |
  Where-Object { $_.Name -notlike "*_viz" } |
  ForEach-Object {
    $size=(Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue |
           Measure-Object Length -Sum).Sum
    [pscustomobject]@{Path=$_.FullName; GB=[math]::Round($size/1GB,3)}
  } | Sort-Object GB -Descending | Select-Object -First 25 | Format-Table -AutoSize

$free = [math]::Round((Get-PSDrive -Name C).Free/1GB,2)
Write-Host "`nC: Free space: $free GB" -ForegroundColor Green

