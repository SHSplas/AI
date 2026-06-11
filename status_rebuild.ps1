$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$metrics = "cpu_gpt_jarvis_rebuild_l6_v2048_metrics.csv"
$best = "cpu_gpt_jarvis_rebuild_l6_v2048_best.pth"
$main = "cpu_gpt_jarvis_rebuild_l6_v2048.pth"

Write-Host "=== Jarvis Rebuild Status ==="
Write-Host ("Time: {0}" -f (Get-Date))

$trainProcs = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { ([string]$_.CommandLine) -like "*train.py*cpu_gpt_jarvis_rebuild_l6_v2048*" }

if ($trainProcs) {
    Write-Host "Training process: RUNNING"
    $trainProcs | Select-Object ProcessId, CommandLine | Format-Table -AutoSize
} else {
    Write-Host "Training process: NOT RUNNING"
}

if (Test-Path $metrics) {
    $item = Get-Item $metrics
    Write-Host ("Metrics file: {0} bytes (updated {1})" -f $item.Length, $item.LastWriteTime)
    $lines = Get-Content $metrics
    if ($lines.Count -gt 1) {
        Write-Host "Last metrics rows:"
        $lines | Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
    }
} else {
    Write-Host "Metrics file missing."
}

$py = @'
import os
import torch
for p in ["cpu_gpt_jarvis_rebuild_l6_v2048_best.pth", "cpu_gpt_jarvis_rebuild_l6_v2048.pth"]:
    if not os.path.exists(p):
        print(f"{p}: missing")
        continue
    ckpt = torch.load(p, map_location="cpu")
    print(f"{p}: step={ckpt.get('step')} best_val={ckpt.get('best_val')}")
'@
$py | python -
