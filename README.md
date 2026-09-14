# Cognitive impairment classification from speech

This repository contains the reproducible Python pipeline used for a dissertation on binary
cognitive-impairment classification from spontaneous speech. It combines transcript embeddings,
eGeMAPS acoustic features, and age/sex metadata.

The old notebooks have been removed. Their useful behavior now lives behind one command-line
interface, and their stored metrics must not be cited: the notebook used its evaluation data for
early stopping and model selection, called repeated 70/30 splits "10-fold cross-validation",
calculated ROC AUC from hard labels, and did not optimize the final classifier layers. No corrected
model results are committed yet.

## What works

- Pitt Cookie Theft parsing, participant-speech acoustic extraction, and transformer embeddings.
- ADReSS train/test preprocessing and official test prediction export.
- Participant-grouped nested evaluation with five neural modality combinations and a logistic
  regression baseline.
- Fold-safe scaling, class weights, early stopping, out-of-fold predictions, checkpoints, plots,
  and machine-readable metrics.
- SHAP explanations calculated only for each fold's held-out recordings.
- Whisper transcription, dataset statistics, prepared-table prediction, and prediction for a new
  participant-only recording.

Full model training and SHAP are intentionally not run as part of repository verification. They
are explicit, potentially long-running commands.

## Setup

The locked environment targets Python 3.11.0 on Linux x86-64 with an NVIDIA GPU. It installs
PyTorch from the CUDA 12.8 index so that Blackwell GPUs, including the RTX 50 series, use a build
containing `sm_120` kernels. FFmpeg must be available on `PATH` for Whisper and compressed audio
support.

Install [uv](https://docs.astral.sh/uv/) and create the exact environment from the committed lock:

```bash
uv sync --locked
```

Verify the PyTorch build before preprocessing or training:

```bash
uv run python - <<'PY'
import torch

print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('Compiled architectures:', torch.cuda.get_arch_list())
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
print('CUDA smoke tensor:', torch.ones(1, device='cuda'))
PY
```

The output must identify a CUDA 12.8 PyTorch build, include `sm_120` in the compiled architectures,
and create the CUDA tensor successfully. Commands using `--device auto` fail immediately when CUDA
cannot execute instead of silently starting a much slower CPU job. Explicit `--device cpu` remains
available for deliberate debugging.

The repository does not download DementiaBank data. Access is restricted by the dataset licence,
and `data/` is ignored by Git.

## Expected data layout

Place the extracted corpora under `data/DementiaBank/`:

```text
data/DementiaBank/
├── Pitt/{Control,Dementia}/cookie/*.cha
├── Pitt_audio/{Control,Dementia}/cookie/*.mp3
├── ADReSS-IS2020-train/ADReSS-IS2020-data/train/
└── ADReSS-IS2020-test/ADReSS-IS2020-data/test/
```

The Pitt source contains 552 Cookie Theft CHAT files in the current local copy. Three dementia
records lack the metadata required by this model, so the existing data produces 549 usable
recordings. The preprocessing manifest records those exclusions rather than dropping them
silently.

## Preprocessing

The checked-in configuration preserves the architecture and numeric settings from the dissertation
notebook. They are legacy defaults, not newly tuned values.

```bash
uv run python -m cognitive_impairments preprocess pitt \
  --data-root data/DementiaBank \
  --config configs/pitt_multimodal.toml \
  --task cookie \
  --output data/processed/pitt_cookie_bge.parquet

uv run python -m cognitive_impairments preprocess adresso \
  --data-root data/DementiaBank \
  --config configs/pitt_multimodal.toml \
  --split train \
  --output data/processed/adresso_train_bge.parquet

uv run python -m cognitive_impairments preprocess adresso \
  --data-root data/DementiaBank \
  --config configs/pitt_multimodal.toml \
  --split test \
  --output data/processed/adresso_test_bge.parquet
```

Each Parquet file has an adjacent `.manifest.json`. The manifest fixes the embedding model,
pooling, acoustic feature order, dimensions, dependency versions, sample counts, and exclusions.
Training and prediction reject incompatible manifests/checkpoints even when tensor dimensions
happen to match.

Pitt labels are defined as `Control = 0` and every non-Control diagnosis (including MCI) as
`cognitive_impairment = 1`. Longitudinal recordings are retained separately. Every recording for
the same participant is assigned to the same fold, including participants whose diagnosis changes
over time.

## Statistics

```bash
uv run python -m cognitive_impairments stats \
  --data data/processed/pitt_cookie_bge.parquet \
  --output outputs/pitt_statistics.json
```

## Dissertation experiment suite

Choose and record the seed explicitly; the repository does not invent one:

```bash
uv run python -m cognitive_impairments train \
  --data data/processed/pitt_cookie_bge.parquet \
  --config configs/pitt_multimodal.toml \
  --seed YOUR_INTEGER_SEED \
  --device auto \
  --output outputs/pitt_bge_seed_YOUR_INTEGER_SEED
```

The primary protocol is ten-fold `StratifiedGroupKFold`, grouped by participant. For each outer
fold, one of nine inner grouped folds selects the epoch using validation cross-entropy. The model is
then reinitialized, fitted on the complete outer-training partition for exactly that number of
epochs, and evaluated once on the outer test fold. The test fold is never used for early stopping.

The suite runs:

1. Demographics only.
2. Acoustic only.
3. Text only.
4. Text and acoustic.
5. Text, acoustic, and demographics.
6. Logistic regression on all modalities.

Every new output directory must not already exist. A completed run contains:

```text
outputs/<run>/
├── config.toml
├── dataset_manifest.json
├── environment.json
├── splits.csv
├── metrics_by_fold.csv
├── metrics_summary.csv
├── confusion_matrices.csv
├── oof_predictions.csv
├── training_history.csv
├── plots/
└── checkpoints/<experiment>/fold_<n>/
    ├── weights.pt
    ├── scalers.npz
    └── metadata.json
```

ROC AUC is calculated from positive-class probabilities. Accuracy, precision, recall, F1, and the
confusion matrix are calculated from complete held-out sample arrays, not averaged batch metrics.
The standard binary probability threshold `0.5` is stored in checkpoint metadata. There is no
"best fold" chosen from test performance.

## Explanations

```bash
uv run python -m cognitive_impairments explain \
  --run outputs/pitt_bge_seed_YOUR_INTEGER_SEED \
  --data data/processed/pitt_cookie_bge.parquet \
  --device auto
```

This writes raw per-fold SHAP arrays, acoustic/demographic plots, and feature/modality importance
tables under the run's `explanations/` directory. Embedding coordinates are retained for numerical
accounting but are not presented as individually interpretable semantic features.

## Prediction

Predict a prepared table by averaging probabilities from all multimodal fold models:

```bash
uv run python -m cognitive_impairments predict table \
  --run outputs/pitt_bge_seed_YOUR_INTEGER_SEED \
  --data data/processed/adresso_test_bge.parquet \
  --adresso-format \
  --output outputs/adresso_test_predictions.txt
```

ADReSS is derived from Pitt and is therefore supported as a separate benchmark/submission format,
not described as independent external validation of a Pitt-trained model. The supplied official
test labels are blank, so this repository exports predictions but cannot calculate test metrics.

For one new sample, the audio must contain participant speech only. It is treated as one participant
segment, which is compatible with the training aggregation:

```bash
uv run python -m cognitive_impairments predict recording \
  --run outputs/pitt_bge_seed_YOUR_INTEGER_SEED \
  --audio recording.wav \
  --transcript transcript.txt \
  --age 70 \
  --sex female \
  --output outputs/recording_prediction.json
```

This is research code, not a clinical diagnostic system.

## Transcription

```bash
uv run python -m cognitive_impairments transcribe \
  --audio recording.wav \
  --model base \
  --output transcript.txt
```

Use `--language <code>` to constrain Whisper when the language is known.

## Tests

Tests cover parsing, data contracts, grouped splitting, fold-only scaling, probability metrics,
optimizer coverage, and artifact collision protection without training a model:

```bash
uv run pytest tests -v
uv run python -m compileall cognitive_impairments tests
uv run python -m cognitive_impairments --help
```

## Notebook migration

| Removed notebook | Replacement |
| --- | --- |
| `audio_parsing.ipynb` | `transcribe` command |
| `demo_stats.ipynb` | `stats` command |
| `separate_models.ipynb` | modality experiments in `train` |
| `better_model_creation.ipynb` | `train`, `explain`, and `predict` commands |

Legacy pickles and checkpoints under ignored `data/` are not deleted, but the new code does not load
them. They lack reliable split, feature, and preprocessing provenance and cannot support corrected
results.
