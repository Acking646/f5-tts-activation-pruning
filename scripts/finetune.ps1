param(
    [Parameter(Mandatory=$true)][string]$DatasetName,
    [Parameter(Mandatory=$true)][string]$MetadataCsv,
    [double]$LearningRate = 0.00001,
    [int]$Epochs = 10,
    [int]$BatchFrames = 1600
)

$ErrorActionPreference = "Stop"
$F5DataRoot = python -c "from importlib.resources import files; print(files('f5_tts').joinpath('../../data'))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$DatasetOutput = Join-Path $F5DataRoot ($DatasetName + "_pinyin")

python -m f5_tts.train.datasets.prepare_csv_wavs $MetadataCsv $DatasetOutput
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

accelerate launch -m f5_tts.train.finetune_cli `
    --exp_name F5TTS_v1_Base `
    --dataset_name $DatasetName `
    --learning_rate $LearningRate `
    --epochs $Epochs `
    --batch_size_type frame `
    --batch_size_per_gpu $BatchFrames `
    --max_samples 64 `
    --grad_accumulation_steps 1 `
    --num_warmup_updates 100 `
    --save_per_updates 500 `
    --last_per_updates 100 `
    --keep_last_n_checkpoints 2 `
    --finetune `
    --tokenizer pinyin `
    --logger tensorboard `
    --log_samples
