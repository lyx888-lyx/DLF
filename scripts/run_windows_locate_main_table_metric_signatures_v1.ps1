param(
    [string]$ResultRoot = "result",
    [int]$TopK = 20,
    [string]$WriteReport = "result\main_table_metric_signature_matches.json"
)

$ErrorActionPreference = "Stop"

python .\locate_main_table_metric_signatures.py `
    --result-root $ResultRoot `
    --top-k $TopK `
    --write-report $WriteReport

if ($LASTEXITCODE -ne 0) {
    throw "Main-table metric signature locator failed with exit code $LASTEXITCODE"
}
