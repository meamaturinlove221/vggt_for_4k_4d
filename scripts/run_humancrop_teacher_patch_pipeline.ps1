param(
    [string]$TeacherPredictionsNpz = "output\\modal_results\\20260422_60views_humancrop_teacher_from_family_r1\\predictions.npz",
    [int[]]$ViewCounts = @(6, 8, 12, 20),
    [ValidateSet("teacher", "refined")]
    [string]$NormalSource = "teacher",
    [string]$RefinerCheckpoint = "",
    [ValidateSet("coarse_valid", "human_mask", "coarse_or_human")]
    [string]$PatchMaskSource = "coarse_or_human",
    [ValidateSet("none", "mean_view_delta", "projected_token_sample")]
    [string]$SummaryUpdate = "none",
    [int]$TargetSize = 256,
    [double]$TeacherConfPercentile = 15.0,
    [string]$OutputTag = "20260422_humancrop_teacherpatch_r1",
    [switch]$SkipSubset,
    [switch]$SkipExport,
    [switch]$SkipPatch,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$teacherPredictionsPath = Join-Path $repoRoot $TeacherPredictionsNpz
$outputRoot = Join-Path $repoRoot ("output\\detail_normal_refiner_20260422\\" + $OutputTag)
$subsetRoot = Join-Path $outputRoot "subset_predictions"
$exportRoot = Join-Path $outputRoot "dataset_exports"
$tempRoot = Join-Path $outputRoot "tmp_cases"

foreach ($dir in @($outputRoot, $subsetRoot, $exportRoot, $tempRoot)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

if ($NormalSource -eq "refined" -and [string]::IsNullOrWhiteSpace($RefinerCheckpoint)) {
    throw "RefinerCheckpoint is required when NormalSource=refined."
}

if (-not $DryRun -and -not $SkipSubset -and -not (Test-Path $teacherPredictionsPath)) {
    throw "Teacher predictions not found: $teacherPredictionsPath"
}

function Invoke-Step([string[]]$CommandArgs) {
    if ($DryRun) {
        Write-Host ("DRY RUN: " + ($CommandArgs -join " "))
        return
    }
    $command = $CommandArgs[0]
    $arguments = @()
    if ($CommandArgs.Length -gt 1) {
        $arguments = $CommandArgs[1..($CommandArgs.Length - 1)]
    }
    & $command @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $($CommandArgs -join ' ')"
    }
}

function Get-SceneName([int]$ViewCount) {
    if ($ViewCount -eq 60) {
        return "0012_11_frame0000_60views_human_crop"
    }
    return "0012_11_frame0000_${ViewCount}views_sparseproto_human_crop"
}

function Get-CaseName([int]$ViewCount) {
    return "0012_11_frame0000_${ViewCount}views_sparseproto_humancrop_resume_r1"
}

$records = @()

foreach ($viewCount in $ViewCounts) {
    $sceneName = Get-SceneName $viewCount
    $sceneDir = Join-Path $repoRoot ("output\\4k4d_preprocessed_scene_variants\\" + $sceneName)
    $caseName = Get-CaseName $viewCount
    $caseDir = Join-Path $repoRoot ("output\\training_cases\\" + $caseName)
    $subsetPredPath = Join-Path $subsetRoot ($sceneName + "_teacher60subset.npz")
    $datasetOutputDir = Join-Path $exportRoot ($sceneName + "_dataset")
    $headDataset = Join-Path $datasetOutputDir "head_roi\\head_samples.npz"
    $faceDataset = Join-Path $datasetOutputDir "face_roi\\face_samples.npz"
    $tempHeadCase = Join-Path $tempRoot ($sceneName + "_headstage")
    $finalCaseDir = Join-Path $repoRoot ("output\\training_cases\\0012_11_frame0000_${viewCount}views_sparseproto_humancrop_teacherpatch_r1")

    if (-not (Test-Path $sceneDir)) {
        throw "Scene directory not found: $sceneDir"
    }
    if (-not (Test-Path $caseDir)) {
        throw "Training case directory not found: $caseDir"
    }

    if (-not $SkipSubset) {
        $subsetArgs = @(
            "python",
            "tools\\subset_predictions_npz.py",
            "--predictions-npz", $teacherPredictionsPath,
            "--subset-scene-dir", $sceneDir,
            "--output-path", $subsetPredPath,
            "--overwrite"
        )
        Invoke-Step $subsetArgs
    }

    if (-not $SkipExport) {
        $exportArgs = @(
            "python",
            "tools\\export_detail_normal_refiner_dataset.py",
            "--scene-dir", $sceneDir,
            "--prior-maps-npz", (Join-Path $sceneDir "prior_maps.npz"),
            "--predictions-npz", $subsetPredPath,
            "--output-dir", $datasetOutputDir,
            "--roi-kind", "all",
            "--target-size", "$TargetSize",
            "--teacher-conf-percentile", "$TeacherConfPercentile"
        )
        Invoke-Step $exportArgs
    }

    if (-not $SkipPatch) {
        $headPatchArgs = @(
            "python",
            "tools\\patch_training_case_with_refined_normals.py",
            "--dataset-npz", $headDataset,
            "--case-dir", $caseDir,
            "--output-case-dir", $tempHeadCase,
            "--normal-source", $NormalSource,
            "--patch-mask-source", $PatchMaskSource,
            "--summary-update", $SummaryUpdate,
            "--overwrite"
        )
        if ($NormalSource -eq "refined") {
            $headPatchArgs += @("--checkpoint", $RefinerCheckpoint)
        }
        Invoke-Step $headPatchArgs

        $facePatchArgs = @(
            "python",
            "tools\\patch_training_case_with_refined_normals.py",
            "--dataset-npz", $faceDataset,
            "--case-dir", $tempHeadCase,
            "--output-case-dir", $finalCaseDir,
            "--normal-source", $NormalSource,
            "--patch-mask-source", $PatchMaskSource,
            "--summary-update", $SummaryUpdate,
            "--overwrite"
        )
        if ($NormalSource -eq "refined") {
            $facePatchArgs += @("--checkpoint", $RefinerCheckpoint)
        }
        Invoke-Step $facePatchArgs

        if ((Test-Path $tempHeadCase) -and -not $DryRun) {
            Remove-Item -LiteralPath $tempHeadCase -Recurse -Force
        }
    }

    $records += [pscustomobject]@{
        view_count = [int]$viewCount
        scene_dir = $sceneDir
        source_case_dir = $caseDir
        subset_predictions = $subsetPredPath
        dataset_output_dir = $datasetOutputDir
        final_case_dir = $finalCaseDir
        normal_source = $NormalSource
        patch_mask_source = $PatchMaskSource
        summary_update = $SummaryUpdate
    }
}

$summary = [pscustomobject]@{
    teacher_predictions_npz = $teacherPredictionsPath
    output_root = $outputRoot
    normal_source = $NormalSource
    refiner_checkpoint = $(if ($NormalSource -eq "refined") { $RefinerCheckpoint } else { $null })
    patch_mask_source = $PatchMaskSource
    summary_update = $SummaryUpdate
    target_size = [int]$TargetSize
    teacher_conf_percentile = [double]$TeacherConfPercentile
    records = $records
}

$summaryPath = Join-Path $outputRoot "pipeline_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host ($summary | ConvertTo-Json -Depth 6)
