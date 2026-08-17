param(
    [string]$ResultRoot = "result",
    [double]$Corr = 0.802,
    [double]$MAE = 0.693,
    [int]$TopK = 40,
    [switch]$IncludePosthoc
)

$ErrorActionPreference = "Stop"

$argsList = @(
    ".\locate_corr_mae_signature.py",
    "--result-root", $ResultRoot,
    "--corr", $Corr,
    "--mae", $MAE,
    "--top-k", $TopK
)

if ($IncludePosthoc) {
    $argsList += "--include-posthoc"
}

python @argsList
