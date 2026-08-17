param(
    [string[]]$Roots = @("result", "log"),
    [double]$Corr = 0.802,
    [double]$MAE = 0.693,
    [int]$ExpectedN = 686,
    [int]$TopK = 30,
    [switch]$IncludeNonTest
)

$ErrorActionPreference = "Stop"

$argsList = @(
    ".\trace_mosi_main_table_provenance.py",
    "--roots"
) + $Roots + @(
    "--corr", $Corr,
    "--mae", $MAE,
    "--expected-n", $ExpectedN,
    "--top-k", $TopK
)

if ($IncludeNonTest) {
    $argsList += "--include-nontest"
}

python @argsList
