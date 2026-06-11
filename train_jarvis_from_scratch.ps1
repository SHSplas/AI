param(
    [int]$Steps = 1000,
    [int]$BatchSize = 4,
    [int]$AccumSteps = 4,
    [double]$LearningRate = 0.00003
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $Root

python .\scripts\train.py `
    --prepare-data `
    --run-steps $Steps `
    --batch-size $BatchSize `
    --accum-steps $AccumSteps `
    --lr $LearningRate `
    --n-embd 192 `
    --n-head 6 `
    --n-layer 6 `
    --block-size 256 `
    --ckpt-path .\Models\cpu_gpt_jarvis_local_scratch.pth `
    --best-path .\Models\cpu_gpt_jarvis_local_scratch_best.pth `
    --metrics-csv .\metrics\cpu_gpt_jarvis_local_scratch_metrics.csv `
    --reset-optimizer `
    --reset-best-val
