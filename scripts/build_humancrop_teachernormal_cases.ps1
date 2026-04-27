param(
    [string]$TeacherPredictionsNpz = "output\\modal_results\\20260422_60views_humancrop_teacher_from_family_r1\\predictions.npz",
    [string]$RemoteOutputSubdir = "",
    [int[]]$ViewCounts = @(6, 8, 12, 20),
    [double]$TeacherConfPercentile = 15.0,
    [string]$OutputTag = "20260422_humancrop_teachernormal_r1",
    [switch]$SkipSubset,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$teacherPredictionsPath = Join-Path $repoRoot $TeacherPredictionsNpz
$outputRoot = Join-Path $repoRoot ("output\\teacher_normal_case_builds\\" + $OutputTag)
$subsetRoot = Join-Path $outputRoot "subset_predictions"
$remoteCacheDir = Join-Path $subsetRoot "_remote_cache"

foreach ($dir in @($outputRoot, $subsetRoot)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

if (-not $DryRun -and [string]::IsNullOrWhiteSpace($RemoteOutputSubdir) -and -not (Test-Path $teacherPredictionsPath)) {
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

function Get-SourceCaseName([int]$ViewCount) {
    return "0012_11_frame0000_${ViewCount}views_sparseproto_humancrop_resume_r1"
}

function Get-TargetCaseName([int]$ViewCount) {
    return "0012_11_frame0000_${ViewCount}views_sparseproto_humancrop_teachernormal_r1"
}

$records = @()

foreach ($viewCount in $ViewCounts) {
    $sceneName = Get-SceneName $viewCount
    $sceneDir = Join-Path $repoRoot ("output\\4k4d_preprocessed_scene_variants\\" + $sceneName)
    $sourceCaseDir = Join-Path $repoRoot ("output\\training_cases\\" + (Get-SourceCaseName $viewCount))
    $targetCaseDir = Join-Path $repoRoot ("output\\training_cases\\" + (Get-TargetCaseName $viewCount))
    $subsetPredPath = Join-Path $subsetRoot ($sceneName + "_teacher60subset.npz")

    if (-not (Test-Path $sceneDir)) {
        throw "Scene directory not found: $sceneDir"
    }
    if (-not (Test-Path $sourceCaseDir)) {
        throw "Source training case directory not found: $sourceCaseDir"
    }

    if (-not $SkipSubset) {
        if ([string]::IsNullOrWhiteSpace($RemoteOutputSubdir)) {
            $subsetArgs = @(
                "python",
                "tools\\subset_predictions_npz.py",
                "--predictions-npz", $teacherPredictionsPath,
                "--subset-scene-dir", $sceneDir,
                "--output-path", $subsetPredPath,
                "--overwrite"
            )
        } else {
            $subsetArgs = @(
                "python",
                "tools\\download_remote_subset_predictions.py",
                "--remote-output-subdir", $RemoteOutputSubdir,
                "--subset-scene-dir", $sceneDir,
                "--output-path", $subsetPredPath,
                "--cache-dir", $remoteCacheDir,
                "--overwrite"
            )
        }
        Invoke-Step $subsetArgs
    }

    $augmentArgs = @(
        "python",
        "tools\\augment_training_case_with_teacher_normals.py",
        "--case-dir", $sourceCaseDir,
        "--predictions-npz", $subsetPredPath,
        "--output-case-dir", $targetCaseDir,
        "--teacher-conf-percentile", "$TeacherConfPercentile",
        "--overwrite"
    )
    Invoke-Step $augmentArgs

    $records += [pscustomobject]@{
        view_count = [int]$viewCount
        scene_dir = $sceneDir
        source_case_dir = $sourceCaseDir
        subset_predictions = $subsetPredPath
        target_case_dir = $targetCaseDir
    }
}

$summary = [pscustomobject]@{
    teacher_predictions_npz = $teacherPredictionsPath
    remote_output_subdir = $(if ([string]::IsNullOrWhiteSpace($RemoteOutputSubdir)) { $null } else { $RemoteOutputSubdir })
    output_root = $outputRoot
    teacher_conf_percentile = [double]$TeacherConfPercentile
    records = $records
}

$summaryPath = Join-Path $outputRoot "build_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host ($summary | ConvertTo-Json -Depth 6)
