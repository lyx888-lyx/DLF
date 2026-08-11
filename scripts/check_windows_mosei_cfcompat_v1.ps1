$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    Write-Host "================ MOSEI CFCompat-v1 preflight ======================"

    $dataset = ".\dataset\MOSEI\Processed\aligned_50.pkl"
    if (-not (Test-Path $dataset)) {
        throw "MOSEI aligned dataset missing: $dataset"
    }
    Write-Host "Dataset: PASS  $dataset"

    $stage1Csv = ".\result\missing_baseline\moddrop\train\mosei_per_seed.csv"
    if (-not (Test-Path $stage1Csv)) {
        throw "Canonical Stage1 CSV missing: $stage1Csv"
    }
    $stage1Rows = @(Import-Csv $stage1Csv)
    $formalSeeds = @(1111,1112,1113,1114,1115)
    $uniqueSeeds = @($stage1Rows | ForEach-Object { [int]$_.Seed } | Sort-Object -Unique)
    if ($stage1Rows.Count -ne 5 -or $uniqueSeeds.Count -ne 5) {
        throw "Stage1 canonical CSV must contain exactly five unique formal rows; rows=$($stage1Rows.Count) unique=$($uniqueSeeds.Count)"
    }
    if (Compare-Object $formalSeeds $uniqueSeeds) {
        throw "Stage1 canonical seed set is not exactly 1111..1115."
    }

    $allPass = $true
    $auditRows = @()
    foreach ($seed in $formalSeeds) {
        $cleanCkpt = ".\pt\DLF_mosei_seed${seed}_best.pth"
        $cleanManifest = ".\result\missing_baseline\mosei_clean_stage0_v1\seed${seed}\run_manifest.json"
        $stage1Manifest = ".\result\missing_baseline\mosei_moddrop_stage1_v1\seed${seed}\run_manifest.json"
        $stage1Row = @($stage1Rows | Where-Object { [int]$_.Seed -eq $seed })

        $cleanOk = Test-Path $cleanCkpt
        $cleanManifestOk = Test-Path $cleanManifest
        $stage1ManifestOk = Test-Path $stage1Manifest
        $stage1RowOk = ($stage1Row.Count -eq 1)
        $stage1Ckpt = if ($stage1RowOk) { [string]$stage1Row[0].Checkpoint } else { "" }
        $stage1CkptOk = ($stage1RowOk -and (Test-Path $stage1Ckpt))
        $cleanShaOk = $false
        $stage1ShaOk = $false
        $testFree = $false
        $bestEpochOk = $false

        if ($cleanOk -and $cleanManifestOk) {
            $cm = Get-Content $cleanManifest -Raw | ConvertFrom-Json
            $cleanActual = (Get-FileHash $cleanCkpt -Algorithm SHA256).Hash.ToLowerInvariant()
            $cleanRecorded = ([string]$cm.checkpoint_sha256).ToLowerInvariant()
            $cleanShaOk = ($cleanActual -eq $cleanRecorded)
        }

        if ($stage1CkptOk -and $stage1ManifestOk) {
            $sm = Get-Content $stage1Manifest -Raw | ConvertFrom-Json
            $stage1Actual = (Get-FileHash $stage1Ckpt -Algorithm SHA256).Hash.ToLowerInvariant()
            $stage1Recorded = ([string]$sm.stage1_checkpoint.sha256).ToLowerInvariant()
            $stage1ShaOk = ($stage1Actual -eq $stage1Recorded)
            $testFree = (-not [bool]$sm.official_test_constructed) -and (-not [bool]$sm.test_used_for_tuning)
            $bestEpochOk = ([int]$stage1Row[0].BestEpoch -eq [int]$sm.training.best_epoch)
        }

        $seedPass = $cleanOk -and $cleanManifestOk -and $cleanShaOk -and `
                    $stage1RowOk -and $stage1CkptOk -and $stage1ManifestOk -and `
                    $stage1ShaOk -and $testFree -and $bestEpochOk
        if (-not $seedPass) { $allPass = $false }

        $formalCache = ".\result\counterfactual_compatibility\mosei_cf_compat_v1\mosei\seed${seed}\train_counterfactual_compatibility.csv"
        $cfCkpt = ".\pt\missing_baseline\cf_compat_kd_v1\mosei\seed${seed}\DLF_mosei_seed${seed}_best_valid.pth"

        $auditRows += [PSCustomObject]@{
            Seed = $seed
            CleanSHA = $cleanShaOk
            Stage1BestEpoch = if ($stage1RowOk) { [int]$stage1Row[0].BestEpoch } else { $null }
            Stage1J = if ($stage1RowOk) { [double]$stage1Row[0].J_val } else { $null }
            Stage1SHA = $stage1ShaOk
            TestFree = $testFree
            FormalCacheExists = (Test-Path $formalCache)
            CFCompatCheckpointExists = (Test-Path $cfCkpt)
            PASS = $seedPass
        }
    }

    $auditRows | Format-Table -AutoSize

    if (-not $allPass) {
        throw "At least one Stage0/Stage1 SHA/protocol binding failed. Do not start CFCompat."
    }

    Write-Host "==================================================================="
    Write-Host "STATUS: MOSEI_STAGE0_X5_AND_STAGE1_X5_FROZEN_READY_FOR_CFCOMPAT_V1"
    Write-Host "EXPECTED TRAIN CACHE SIZE: 16326 PER SEED"
    Write-Host "CFCOMPAT TRAINING WILL CONSTRUCT TRAIN + VALID ONLY"
    Write-Host "OFFICIAL MOSEI TEST WILL NOT BE CONSTRUCTED"
}
finally {
    Pop-Location
}
