param(
    [int]$GpuId = 0,
    [int]$NumWorkers = 1,
    [string]$ModelSaveDir = "pt",
    [string]$ResultRoot = "result",
    [string]$ConfigFile = "config/config.json",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$argsList = @(
    ".\replay_mosi_sentiment_analysis_predictions.py",
    "--dataset", "mosi",
    "--gpu-ids", "$GpuId",
    "--num-workers", "$NumWorkers",
    "--model-save-dir", $ModelSaveDir,
    "--result-root", $ResultRoot,
    "--config-file", $ConfigFile,
    "--acknowledge-posthoc-test-analysis"
)

if ($Overwrite) {
    $argsList += "--overwrite"
}

Write-Host "POSTHOC MOSI Test replay: historically accessed Test; analysis only." -ForegroundColor Yellow
Write-Host "No training / checkpoint selection / calibration / blend-weight search will be performed." -ForegroundColor Yellow

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MOSI post-hoc prediction replay failed with exit code $LASTEXITCODE"
}
