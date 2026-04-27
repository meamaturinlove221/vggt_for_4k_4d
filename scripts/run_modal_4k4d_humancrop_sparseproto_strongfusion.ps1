param(
    [ValidateSet("family", "focus6")]
    [string]$Mode = "family",
    [string]$ModalExe = "",
    [string]$TrainOutputSubdir = "",
    [string]$TrainDownloadDir = "",
    [string]$ResumeCheckpointRelpath = "vggt_4k4d_train/20260422_6view_focus_resume_strongfusion_r1/inference_model.pt",
    [string]$EvalCheckpointRelpath = "",
    [string]$EvalOutputSubdir = "",
    [string]$ModalGpu = "A100-80GB",
    [switch]$RunEval,
    [switch]$SkipTrain,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$todayTag = Get-Date -Format "yyyyMMdd"

$familyCaseDirs = @(
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_6views_sparseproto_humancrop_resume_r1"),
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_8views_sparseproto_humancrop_resume_r1"),
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_12views_sparseproto_humancrop_resume_r1"),
    (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_20views_sparseproto_humancrop_resume_r1")
)

$spec = switch ($Mode) {
    "family" {
        @{
            ConfigName = "4k4d_prior_case_sparseproto_humancrop_resume_r1"
            ExpName = "4k4d_prior_case_sparseproto_humancrop_resume_r1"
            LocalCaseDirs = ($familyCaseDirs -join ",")
            TrainRunName = "${todayTag}_sparseproto_humancrop_resume_r1"
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
            ConfigName = "4k4d_prior_case_6view_focus_humancrop_resume_r1"
            ExpName = "4k4d_prior_case_6view_focus_humancrop_resume_r1"
            LocalCaseDirs = (Join-Path $repoRoot "output\training_cases\0012_11_frame0000_6views_sparseproto_humancrop_resume_r1")
            TrainRunName = "${todayTag}_6view_focus_humancrop_resume_r1"
            MaxEpochs = 4
            LimitTrainBatches = 220
            LimitValBatches = 20
            MaxImgPerGpu = 6
            ImgNumsMin = 6
            ImgNumsMax = 6
        }
    }
}

$evalSceneDir = Join-Path $repoRoot "output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_human_crop"

if ($SkipTrain -and -not $RunEval) {
    throw "SkipTrain requires RunEval."
}

$caseDirList = @($spec.LocalCaseDirs -split "," | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
foreach ($caseDir in $caseDirList) {
    if (-not (Test-Path $caseDir)) {
        throw "Training case directory not found: $caseDir"
    }
}

if ($RunEval -and -not (Test-Path $evalSceneDir)) {
    throw "Eval scene directory not found: $evalSceneDir"
}

if ([string]::IsNullOrWhiteSpace($TrainOutputSubdir)) {
    $TrainOutputSubdir = "vggt_4k4d_train/$($spec.TrainRunName)"
}

if ([string]::IsNullOrWhiteSpace($TrainDownloadDir)) {
    $TrainDownloadDir = Join-Path $repoRoot ("output\modal_training_results\" + [System.IO.Path]::GetFileName($TrainOutputSubdir))
}

if ([string]::IsNullOrWhiteSpace($EvalCheckpointRelpath)) {
    if ($SkipTrain) {
        throw "EvalCheckpointRelpath is required when SkipTrain is set."
    }
    $EvalCheckpointRelpath = "$TrainOutputSubdir/inference_model.pt"
}

if ($RunEval -and [string]::IsNullOrWhiteSpace($EvalOutputSubdir)) {
    $EvalOutputSubdir = "vggt_4k4d_infer/$($spec.TrainRunName)_eval6"
}

Write-Host "[humancrop-strongfusion] mode=$Mode"
Write-Host "[humancrop-strongfusion] config_name=$($spec.ConfigName)"
Write-Host "[humancrop-strongfusion] local_case_dirs=$($spec.LocalCaseDirs)"
Write-Host "[humancrop-strongfusion] train_output_subdir=$TrainOutputSubdir"
Write-Host "[humancrop-strongfusion] train_download_dir=$TrainDownloadDir"
Write-Host "[humancrop-strongfusion] resume_checkpoint_relpath=$ResumeCheckpointRelpath"
if ($RunEval) {
    Write-Host "[humancrop-strongfusion] eval_scene_dir=$evalSceneDir"
    Write-Host "[humancrop-strongfusion] eval_checkpoint_relpath=$EvalCheckpointRelpath"
    Write-Host "[humancrop-strongfusion] eval_output_subdir=$EvalOutputSubdir"
}

if (-not $SkipTrain) {
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
    if ($DryRun) {
        $trainArgs += "-SkipPreflight"
        $trainArgs += "-DryRun"
    }
    & powershell @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "humancrop strongfusion train step failed with exit code $LASTEXITCODE."
    }
}

if ($RunEval) {
    $inferScript = Join-Path $repoRoot "scripts\run_modal_4k4d_vggt_infer.ps1"
    $inferArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $inferScript,
        "-LocalSceneDir", $evalSceneDir,
        "-OutputSubdir", $EvalOutputSubdir,
        "-CheckpointRelpath", $EvalCheckpointRelpath
    )
    if (-not [string]::IsNullOrWhiteSpace($ModalExe)) {
        $inferArgs += @("-ModalExe", $ModalExe)
    }
    if ($DryRun) {
        $inferArgs += "-DryRun"
    }
    & powershell @inferArgs
    if ($LASTEXITCODE -ne 0) {
        throw "humancrop strongfusion eval step failed with exit code $LASTEXITCODE."
    }
}
