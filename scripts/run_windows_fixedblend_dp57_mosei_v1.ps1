param(
    [ValidateSet("valid","test")]
    [string]$Split = "valid",
    [Parameter(Mandatory=$true)]
    [string]$V13Predictions,
    [string]$Raw5Root = ".\result\missing_baseline\cfcompat_prediction_ensemble_v1\mosei",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/fixedblend-dp57-mosei-ready-v1"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

foreach ($seed in @(1111,1112,1113,1114,1115)) {
    $path = Join-Path $Raw5Root ("online_seed{0}_{1}_predictions.csv" -f $seed,$Split)
    if (-not (Test-Path $path)) {
        throw "Missing frozen MOSEI Raw5 member: $path"
    }
}
if (-not (Test-Path $V13Predictions)) {
    throw "Missing frozen MOSEI v13 predictions: $V13Predictions"
}

python -m py_compile .\evaluate_fixedblend_dp57_v1.py .\trains\singleTask\anchor_decision_projection.py
if ($LASTEXITCODE -ne 0) {
    throw "FixedBlend-DP57 py_compile failed with exit code $LASTEXITCODE"
}

$output = ".\result\missing_baseline\fixedblend_dp57_v1\mosei\$Split"
$argsList = @(
    ".\evaluate_fixedblend_dp57_v1.py",
    "--dataset", "mosei",
    "--split", $Split,
    "--raw5-root", $Raw5Root,
    "--v13-predictions", $V13Predictions,
    "--output-root", $output
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MOSEI FixedBlend-DP57 $Split evaluation failed with exit code $LASTEXITCODE"
}

$metrics = Join-Path $output "fixedblend_dp57_metrics.csv"
$diag = Join-Path $output "fixedblend_dp57_projection_diagnostics.csv"
$summary = Join-Path $output "fixedblend_dp57_summary.json"
foreach ($path in @($metrics,$diag,$summary)) {
    if (-not (Test-Path $path)) { throw "Missing output: $path" }
}

Write-Host ""
Write-Host "================ MOSEI LAV / MissingMacro ============================"
Import-Csv $metrics |
    Where-Object { $_.Mode -in @("LAV","MissingMacro") } |
    Select-Object Method,Mode,J,MAE,Corr,acc_2,F1_score,acc_7,acc_5 |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ Projection diagnostics =============================="
Import-Csv $diag | Format-Table -AutoSize

$data = Get-Content $summary -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "================ Summary ============================================="
Write-Host ("Split:                          {0}" -f $Split)
Write-Host ("Anchor seed:                    {0}" -f $data.anchor_seed)
Write-Host ("FixedBlend J:                   {0}" -f $data.fixedblend_J)
Write-Host ("DP57 J:                         {0}" -f $data.dp57_J)
Write-Host ("Delta J DP57-FixedBlend:        {0}" -f $data.delta_J_dp57_minus_fixedblend)
Write-Host ("DP57 LAV Acc7:                  {0}" -f $data.dp57_LAV.acc_7)
Write-Host ("DP57 LAV Acc5:                  {0}" -f $data.dp57_LAV.acc_5)
Write-Host ("DP57 LAV Acc2:                  {0}" -f $data.dp57_LAV.acc_2)
Write-Host ("DP57 LAV F1:                    {0}" -f $data.dp57_LAV.F1_score)
Write-Host ("DP57 LAV Corr:                  {0}" -f $data.dp57_LAV.Corr)
Write-Host ("DP57 LAV MAE:                   {0}" -f $data.dp57_LAV.MAE)
Write-Host "SAMPLE-LEVEL PROJECTED OUTPUT WRITTEN: False"
Write-Host "Result root: $output"
