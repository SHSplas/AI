$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Resolve-Python {
    $candidates = @(
        "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe",
        "$env:LOCALAPPDATA\Python\bin\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path -LiteralPath $cmd.Source)) {
        $probe = & $cmd.Source --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$probe" -match 'Python') {
            return $cmd.Source
        }
    }

    throw "Could not find a usable Python interpreter. Install Python or disable the Windows Store python alias."
}

$python = Resolve-Python
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = Join-Path $PSScriptRoot 'temp'
$metricsDir = Join-Path $PSScriptRoot 'metrics'
$modelsDir = Join-Path $PSScriptRoot 'Models'
New-Item -ItemType Directory -Force -Path $logDir, $metricsDir, $modelsDir | Out-Null

$watchLog = Join-Path $logDir "v8_neurons_$runStamp.watch.log"
$trainLog = Join-Path $logDir "v8_neurons_$runStamp.train.log"
$trainErr = Join-Path $logDir "v8_neurons_$runStamp.train.err"
$smokeLog = Join-Path $logDir "v8_neurons_$runStamp.smoke.log"
$smokeErr = Join-Path $logDir "v8_neurons_$runStamp.smoke.err"

$baseBest = Join-Path $modelsDir 'cpu_gpt_jarvis_v6_guarded_best.pth'
$ckpt = Join-Path $modelsDir 'cpu_gpt_jarvis_v8_neurons.pth'
$best = Join-Path $modelsDir 'cpu_gpt_jarvis_v8_neurons_best.pth'
$metrics = Join-Path $metricsDir 'cpu_gpt_jarvis_v8_neurons_metrics.csv'

"[$(Get-Date -Format s)] v8 neuron pipeline started" | Out-File -FilePath $watchLog -Encoding utf8
"Python=$python" | Out-File -FilePath $watchLog -Append -Encoding utf8

if (-not (Test-Path -LiteralPath $ckpt)) {
    if (-not (Test-Path -LiteralPath $baseBest)) {
        throw "Base checkpoint not found: $baseBest"
    }
    Copy-Item -LiteralPath $baseBest -Destination $ckpt -Force
    "[$(Get-Date -Format s)] Seeded v8 checkpoint from v6 best" | Out-File -FilePath $watchLog -Append -Encoding utf8
}

$trainArgs = @(
    'scripts\train.py',
    '--train-data', 'data\jarvis_mix_train.txt',
    '--val-data', 'data\jarvis_mix_val.txt',
    '--ckpt-path', $ckpt,
    '--best-path', $best,
    '--metrics-csv', $metrics,
    '--run-steps', '1200',
    '--batch-size', '4',
    '--accum-steps', '4',
    '--lr', '6e-6',
    '--warmup-steps', '100',
    '--eval-every', '100',
    '--eval-batches', '8',
    '--save-every', '200',
    '--sample-every', '200',
    '--log-every', '20',
    '--grad-clip', '1.0',
    '--early-stop-patience', '14',
    '--threads', '6',
    '--interop-threads', '1',
    '--reset-optimizer',
    '--reset-best-val'
)

"[$(Get-Date -Format s)] Training started" | Out-File -FilePath $watchLog -Append -Encoding utf8
& $python @trainArgs 1> $trainLog 2> $trainErr
"[$(Get-Date -Format s)] Training finished" | Out-File -FilePath $watchLog -Append -Encoding utf8

$smokePrompts = @(
    'who made you?',
    'why made you?',
    'give me an example of a city',
    'i am crazy kjhdfkjncfrdfhrujf help',
    'what is 15% of 319',
    'how to make a sandwitch',
    'exit'
)

$smokePrompts | & $python scripts\chat.py --ckpt $best --no-int8 1> $smokeLog 2> $smokeErr
"[$(Get-Date -Format s)] Smoke eval finished" | Out-File -FilePath $watchLog -Append -Encoding utf8

$completionFile = Join-Path $logDir "v8_neurons_$runStamp.done.txt"
$localDone = Get-Date
$utcDone = (Get-Date).ToUniversalTime()
"COMPLETED_LOCAL=$($localDone.ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -FilePath $completionFile -Encoding utf8
"COMPLETED_UTC=$($utcDone.ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -FilePath $completionFile -Append -Encoding utf8
"BEST_CHECKPOINT=$best" | Out-File -FilePath $completionFile -Append -Encoding utf8
"METRICS=$metrics" | Out-File -FilePath $completionFile -Append -Encoding utf8

"[$(Get-Date -Format s)] v8 neuron pipeline complete" | Out-File -FilePath $watchLog -Append -Encoding utf8
