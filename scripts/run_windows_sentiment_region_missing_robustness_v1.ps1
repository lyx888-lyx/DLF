param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('mosi','mosei')]
    [string]$Dataset,

    [Parameter(Mandatory=$true)]
    [ValidateSet('valid','test')]
    [string]$Split,

    [Parameter(Mandatory=$true)]
    [string]$BaselinePredictions,

    [Parameter(Mandatory=$true)]
    [string]$OursPredictions,

    [string]$BaselineName = 'DLF',
    [string]$OursName = 'Ours',
    [string]$OutputDir = '',
    [switch]$AllowTestAnalysis
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$argsList = @(
    '.\analyze_sentiment_region_missing_robustness.py',
    '--dataset', $Dataset,
    '--split', $Split,
    '--baseline-predictions', $BaselinePredictions,
    '--ours-predictions', $OursPredictions,
    '--baseline-name', $BaselineName,
    '--ours-name', $OursName
)

if ($OutputDir -ne '') {
    $argsList += @('--output-dir', $OutputDir)
}
if ($AllowTestAnalysis) {
    $argsList += '--allow-test-analysis'
}

Write-Host 'Running frozen post-hoc sentiment-region / missing-modality analysis'
Write-Host "Dataset: $Dataset"
Write-Host "Split:   $Split"
Write-Host "Baseline: $BaselinePredictions"
Write-Host "Ours:     $OursPredictions"

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Analysis failed with exit code $LASTEXITCODE"
}
