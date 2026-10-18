param(
    [string]$RuleIds = '',
    [int]$Port = 8001
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$adapterPath = Join-Path $PSScriptRoot 'outputs/cash-lora/best-adapter'
if (-not (Test-Path (Join-Path $adapterPath 'training_contract.json'))) {
    throw 'No completed best-adapter. Run preparation, smoke and training first.'
}
$env:ML_MODEL_DIR = Join-Path $PSScriptRoot 'models/Qwen3-VL-2B-Instruct'
$env:ML_ADAPTER_DIR = $adapterPath
$env:ML_RULES_FILE = Join-Path $PSScriptRoot 'data/rules_v2.json'
$env:ML_LOAD_4BIT = '1'
$env:ML_CONSTRAINED_JSON = '1'
# This launcher is explicitly for the supplied staged training scenes.
$env:ML_STAGED_CONTEXT = '1'
$env:ML_FRAMES_PER_WINDOW = '16'
$env:ML_MAX_SIDE = '224'
$env:ML_WINDOW_SEC = '150'
$env:ML_STRIDE_SEC = '150'
$env:ML_MAX_NEW_TOKENS = '768'
$env:ML_TIMEOUT_SEC = '900'
$env:ML_RULE_IDS = $RuleIds
& "$PSScriptRoot/run.ps1" -Mode local -Port $Port
