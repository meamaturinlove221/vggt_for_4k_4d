param(
    [ValidateSet("human_crop", "human_crop_softmatte")]
    [string]$Variant = "human_crop",
    [string]$ModalExe = "",
    [string]$TrainOutputSubdir = "",
    [string]$TrainDownloadDir = "",
    [string]$EvalCheckpointRelpath = "",
    [string]$EvalOutputSubdir = "",
    [string]$ModalGpu = "A100-40GB",
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

$variantSpec = switch ($Variant) {
    "human_crop" {
        @{
            ConfigName = "4k4d_prior_case_6view_singlecase_human_crop_overfit_b40"
            ExpName = "4k4d_prior_case_6view_singlecase_human_crop_overfit_b40"
            LocalCaseDir = Join-Path $repoRoot "output\training_cases\0012_11_frame0000_6views_preprocess_crop_b40"
            LocalSceneDir = Join-Path $repoRoot "output\preprocess_ablation_20260421\0012_11_frame0000_6views_sparseproto_human_crop"
            TrainRunName = "${todayTag}_6view_singlecase_human_crop_overfit_b40"
            EvalRunName = "${todayTag}_6views_human_crop_from_singlecase_human_crop_overfit_b40"
        }
    }
    "human_crop_softmatte" {
        @{
            ConfigName = "4k4d_prior_case_6view_singlecase_human_crop_softmatte_overfit_b40"
            ExpName = "4k4d_prior_case_6view_singlecase_human_crop_softmatte_overfit_b40"
            LocalCaseDir = Join-Path $repoRoot "output\training_cases\0012_11_frame0000_6views_preprocess_crop_softmatte_v2_b40"
            LocalSceneDir = Join-Path $repoRoot "output\preprocess_ablation_20260421\0012_11_frame0000_6views_sparseproto_human_crop_softmatte"
            TrainRunName = "${todayTag}_6view_singlecase_human_crop_softmatte_overfit_b40"
            EvalRunName = "${todayTag}_6views_human_crop_softmatte_from_singlecase_human_crop_softmatte_overfit_b40"
        }
    }
}

if ($SkipTrain -and -not $RunEval) {
    throw "SkipTrain requires RunEval."
}

if (-not (Test-Path $variantSpec.LocalCaseDir)) {
    throw "Training case directory not found: $($variantSpec.LocalCaseDir)"
}

if ($RunEval -and -not (Test-Path $variantSpec.LocalSceneDir)) {
    throw "Eval scene directory not found: $($variantSpec.LocalSceneDir)"
}

if ([string]::IsNullOrWhiteSpace($TrainOutputSubdir)) {
    $TrainOutputSubdir = "vggt_4k4d_train/$($variantSpec.TrainRunName)"
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
    $EvalOutputSubdir = "vggt_4k4d_infer/$($variantSpec.EvalRunName)"
}

Write-Host "[preprocess-singlecase] variant=$Variant"
Write-Host "[preprocess-singlecase] config_name=$($variantSpec.ConfigName)"
Write-Host "[preprocess-singlecase] local_case_dir=$($variantSpec.LocalCaseDir)"
Write-Host "[preprocess-singlecase] train_output_subdir=$TrainOutputSubdir"
Write-Host "[preprocess-singlecase] train_download_dir=$TrainDownloadDir"
if ($RunEval) {
    Write-Host "[preprocess-singlecase] local_scene_dir=$($variantSpec.LocalSceneDir)"
    Write-Host "[preprocess-singlecase] eval_checkpoint_relpath=$EvalCheckpointRelpath"
    Write-Host "[preprocess-singlecase] eval_output_subdir=$EvalOutputSubdir"
}

if (-not $SkipTrain) {
    $trainScript = Join-Path $repoRoot "scripts\run_modal_4k4d_vggt_train.ps1"
    $trainArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $trainScript,
        "-LocalCaseDirs", $variantSpec.LocalCaseDir,
        "-DownloadLocalDir", $TrainDownloadDir,
        "-ConfigName", $variantSpec.ConfigName,
        "-ExpName", $variantSpec.ExpName,
        "-OutputSubdir", $TrainOutputSubdir,
        "-MaxEpochs", "1",
        "-LimitTrainBatches", "40",
        "-LimitValBatches", "5",
        "-ValEpochFreq", "1",
        "-MaxImgPerGpu", "6",
        "-ImgNumsMin", "6",
        "-ImgNumsMax", "6",
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
}

if ($RunEval) {
    $inferScript = Join-Path $repoRoot "scripts\run_modal_4k4d_vggt_infer.ps1"
    $inferArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $inferScript,
        "-LocalSceneDir", $variantSpec.LocalSceneDir,
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
}
