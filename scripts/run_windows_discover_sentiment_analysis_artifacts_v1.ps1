param(
    [string]$ResultRoot = "result",
    [ValidateSet("mosi", "mosei", "all")]
    [string]$Dataset = "all",
    [string]$WriteReport = ""
)

$ErrorActionPreference = "Stop"

$ArgsList = @(
    ".\discover_sentiment_analysis_artifacts.py",
    "--result-root", $ResultRoot,
    "--dataset", $Dataset
)

if ($WriteReport -ne "") {
    $ArgsList += @("--write-report", $WriteReport)
}

python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "Artifact discovery failed with exit code $LASTEXITCODE"
}
