param(
    [ValidateSet("family", "focus6")]
    [string]$Mode = "family",
    [string]$ModalExe = "",
    [string]$TrainOutputSubdir = "",
    [string]$TrainDownloadDir = "",
    [string]$ResumeCheckpointRelpath = "vggt_4k4d_train/20260423_sparseproto_humancrop_teachernormal_r1/logs/ckpts/checkpoint_2.pt",
    [string]$ModalGpu = "A100-80GB",
    [switch]$SkipPreflight,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$todayTag = Get-Date -Format "yyyyMMdd"

$familyCaseDirs = @(
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_6views_sparseproto_humancrop_teachergeom_teachernormal_r1"),
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_8views_sparseproto_humancrop_teachergeom_teachernormal_r1"),
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_12views_sparseproto_humancrop_teachergeom_teachernormal_r1"),
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_20views_sparseproto_humancrop_teachergeom_teachernormal_r1")
)

$spec = switch ($Mode) {
    "family" {
        @{
            ConfigName = "4k4d_prior_case_sparseproto_humancrop_teachergeom_teachernormal_r1"
            ExpName = "4k4d_prior_case_sparseproto_humancrop_teachergeom_teachernormal_r1"
            LocalCaseDirs = ($familyCaseDirs -join ",")
            TrainRunName = "${todayTag}_sparseproto_humancrop_teachergeom_teachernormal_r1"
            MaxEpochs = 6
            LimitTrainBatches = 180
            LimitValBatches = 20
            MaxImgPerGpu = 8
            ImgNumsMin = 6
            ImgNumsMax = 12
        }
    }
    "focus6" {
        @{
            ConfigName = "4k4d_prior_case_6view_focus_humancrop_teachergeom_teachernormal_r1"
            ExpName = "4k4d_prior_case_6view_focus_humancrop_teachergeom_teachernormal_r1"
            LocalCaseDirs = (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_6views_sparseproto_humancrop_teachergeom_teachernormal_r1")
            TrainRunName = "${todayTag}_6view_focus_humancrop_teachergeom_teachernormal_r1"
            MaxEpochs = 4
            LimitTrainBatches = 220
            LimitValBatches = 20
            MaxImgPerGpu = 6
            ImgNumsMin = 6
            ImgNumsMax = 6
        }
    }
}

if ([string]::IsNullOrWhiteSpace($TrainOutputSubdir)) {
    $TrainOutputSubdir = "vggt_4k4d_train/$($spec.TrainRunName)"
}

if ([string]::IsNullOrWhiteSpace($TrainDownloadDir)) {
    $TrainDownloadDir = Join-Path $repoRoot ("output\modal_training_results\" + [System.IO.Path]::GetFileName($TrainOutputSubdir))
}

$caseDirList = @($spec.LocalCaseDirs -split "," | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
foreach ($caseDir in $caseDirList) {
    if (-not (Test-Path $caseDir)) {
        throw "Training case directory not found: $caseDir"
    }
}

Write-Host "[teachergeom-teachernormal] mode=$Mode"
Write-Host "[teachergeom-teachernormal] config_name=$($spec.ConfigName)"
Write-Host "[teachergeom-teachernormal] local_case_dirs=$($spec.LocalCaseDirs)"
Write-Host "[teachergeom-teachernormal] train_output_subdir=$TrainOutputSubdir"
Write-Host "[teachergeom-teachernormal] train_download_dir=$TrainDownloadDir"
Write-Host "[teachergeom-teachernormal] resume_checkpoint_relpath=$ResumeCheckpointRelpath"

$trainScript = Join-Path $repoRoot "scripts\run_modal_4k4d_vggt_train.ps1"
$trainArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $trainScript,
    "-LocalCaseDirs", $spec.LocalCaseDirs,
    "-DownloadLocalDir", $TrainDownloadDir,
    "-ConfigName", $spec.ConfigName,
    "-ExpName", $spec.ExpName,
    "-OutputSubdir", $TrainOutputSubdir,
    "-PretrainedVolumeSubpath", $ResumeCheckpointRelpath,
    "-MaxEpochs", "$($spec.MaxEpochs)",
    "-LimitTrainBatches", "$($spec.LimitTrainBatches)",
    "-LimitValBatches", "$($spec.LimitValBatches)",
    "-ValEpochFreq", "1",
    "-MaxImgPerGpu", "$($spec.MaxImgPerGpu)",
    "-ImgNumsMin", "$($spec.ImgNumsMin)",
    "-ImgNumsMax", "$($spec.ImgNumsMax)",
    "-LenTrain", "200",
    "-LenTest", "20",
    "-ModalGpu", $ModalGpu
)
if (-not [string]::IsNullOrWhiteSpace($ModalExe)) {
    $trainArgs += @("-ModalExe", $ModalExe)
}
if ($SkipPreflight) {
    $trainArgs += "-SkipPreflight"
}
if ($DryRun) {
    $trainArgs += "-DryRun"
}

& powershell @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "teachergeom teachernormal train step failed with exit code $LASTEXITCODE."
}
