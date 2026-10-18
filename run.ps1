param(
    [ValidateSet('mock','local')][string]$Mode = 'local',
    [string]$BindAddress = $(if ($env:ML_HOST) { $env:ML_HOST } else { '127.0.0.1' }),
    [ValidateRange(1,65535)][int]$Port = $(if ($env:ML_PORT) { [int]$env:ML_PORT } else { 8001 })
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot\.deps;$PSScriptRoot"
$env:ML_MODE = $Mode
if (-not $env:ML_API_KEY -and (Test-Path -LiteralPath "$PSScriptRoot\tmp\ml-api-key.txt")) {
    $env:ML_API_KEY = (Get-Content -LiteralPath "$PSScriptRoot\tmp\ml-api-key.txt" -Raw).Trim()
}
if ($Mode -eq 'local' -and [string]::IsNullOrWhiteSpace($env:ML_API_KEY)) {
    throw 'Set ML_API_KEY or provide tmp/ml-api-key.txt before local startup.'
}
if ($Mode -eq 'mock' -and $BindAddress -notin @('127.0.0.1', 'localhost', '::1')) {
    throw 'Mock is restricted to loopback. Use -Mode local for backend access.'
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& 'C:\Users\Magzhan\cash-ml\.venv\Scripts\python.exe' -m uvicorn app:app --host $BindAddress --port $Port --workers 1
