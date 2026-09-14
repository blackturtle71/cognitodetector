from __future__ import annotations

import json
import pathlib
import typing

import numpy as np
import pandas as pd
from torch import device as TorchDevice, float32, no_grad, softmax, tensor

from cognitive_impairments.artifacts import ScalerState, load_neural_fold
from cognitive_impairments.config import load_experiment_config
from cognitive_impairments.data import read_feature_table
from cognitive_impairments.features import TextEmbedder
from cognitive_impairments.features import create_acoustic_extractor
from cognitive_impairments.features import extract_whole_audio_features
from cognitive_impairments.model import Modalities, MultimodalNet


FULL_MODEL_EXPERIMENT = 'text_acoustic_demographics'


def _fold_directories(run_directory: pathlib.Path) -> list[pathlib.Path]:
    checkpoint_root = run_directory / 'checkpoints' / FULL_MODEL_EXPERIMENT
    directories = sorted(
        checkpoint_root.glob('fold_*'),
        key=lambda path: int(path.name.removeprefix('fold_')),
    )
    if not directories:
        raise FileNotFoundError(f'No multimodal fold checkpoints found under {checkpoint_root}')
    return directories


def _validate_compatibility(
    manifest: dict[str, typing.Any],
    metadata: dict[str, typing.Any],
) -> None:
    fields = ('embedder', 'pooling', 'max_length', 'embedding_dimension')
    metadata_values = {
        'embedder': metadata['embedder'],
        'pooling': metadata['pooling'],
        'max_length': metadata['max_length'],
        'embedding_dimension': metadata['text_dimension'],
    }
    mismatched = [field for field in fields if manifest[field] != metadata_values[field]]
    if manifest['acoustic_features'] != metadata['acoustic_features']:
        mismatched.append('acoustic_features')
    if mismatched:
        raise ValueError(f'Feature/checkpoint mismatch: {", ".join(mismatched)}')


def load_fold_model(
    run_directory: pathlib.Path,
    fold_directory: pathlib.Path,
    device: TorchDevice,
) -> tuple[MultimodalNet, ScalerState, dict[str, typing.Any]]:
    state_dict, scaler, metadata = load_neural_fold(fold_directory)
    config = load_experiment_config(run_directory / 'config.toml')
    model = MultimodalNet(
        text_dimension=metadata['text_dimension'],
        audio_dimension=metadata['audio_dimension'],
        config=config.model,
        modalities=Modalities(**metadata['modalities']),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, metadata


def _fold_probabilities(
    model: MultimodalNet,
    scaler: ScalerState,
    embeddings: np.ndarray,
    audio: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    device: TorchDevice,
) -> np.ndarray:
    scaled_audio = (audio - scaler.audio_mean) / scaler.audio_scale
    scaled_age = (age - scaler.age_mean) / scaler.age_scale
    with no_grad():
        logits = model(
            tensor(embeddings, dtype=float32, device=device),
            tensor(scaled_audio, dtype=float32, device=device),
            tensor(scaled_age, dtype=float32, device=device),
            tensor(sex, dtype=float32, device=device),
        )
    return softmax(logits, dim=1)[:, 1].cpu().numpy()


def _write_predictions(
    output_path: pathlib.Path,
    recording_ids: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    adresso_format: bool,
) -> None:
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite prediction output: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    if adresso_format:
        lines = ['ID   ; Prediction']
        lines.extend(
            f'{recording_id} ; {prediction}'
            for recording_id, prediction in zip(recording_ids, predictions, strict=True)
        )
        output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return
    pd.DataFrame(
        {
            'recording_id': recording_ids,
            'cognitive_impairment_probability': probabilities,
            'prediction': predictions,
        },
    ).to_csv(output_path, index=False)


def predict_table(
    run_directory: pathlib.Path,
    dataset_path: pathlib.Path,
    output_path: pathlib.Path,
    device: TorchDevice,
    adresso_format: bool,
) -> None:
    frame, manifest = read_feature_table(dataset_path)
    embeddings = np.stack(frame['embedding'].map(lambda value: np.asarray(value).reshape(-1)))
    audio = frame[manifest['acoustic_features']].to_numpy(dtype=np.float64)
    age = frame['age'].to_numpy(dtype=np.float64).reshape(-1, 1)
    encoded_sex = frame['sex'].map({'female': 0, 'male': 1})
    if encoded_sex.isna().any():
        raise ValueError('Prediction data contains an unsupported sex value')
    sex = encoded_sex.to_numpy(dtype=np.float64).reshape(-1, 1)

    fold_probabilities = []
    threshold: float | None = None
    for fold_directory in _fold_directories(run_directory):
        model, scaler, metadata = load_fold_model(run_directory, fold_directory, device)
        _validate_compatibility(manifest, metadata)
        fold_probabilities.append(
            _fold_probabilities(model, scaler, embeddings, audio, age, sex, device),
        )
        threshold = metadata['probability_threshold']
    if threshold is None:
        raise RuntimeError('No probability threshold was loaded')
    probabilities = np.mean(fold_probabilities, axis=0)
    _write_predictions(
        output_path,
        frame['recording_id'].astype(str).to_numpy(),
        probabilities,
        threshold,
        adresso_format,
    )


def predict_recording(
    run_directory: pathlib.Path,
    audio_path: pathlib.Path,
    transcript_path: pathlib.Path,
    age_value: int,
    sex_value: str,
    output_path: pathlib.Path,
    device: TorchDevice,
) -> None:
    fold_directories = _fold_directories(run_directory)
    _state, _scaler, first_metadata = load_neural_fold(fold_directories[0])
    embedder = TextEmbedder(
        model_name=first_metadata['embedder'],
        pooling=first_metadata['pooling'],
        max_length=first_metadata['max_length'],
        device=device,
    )
    transcript = transcript_path.read_text(encoding='utf-8').strip()
    embeddings = embedder.encode(transcript).reshape(1, -1)
    acoustic = extract_whole_audio_features(audio_path, create_acoustic_extractor())
    acoustic = acoustic.reindex(first_metadata['acoustic_features'])
    if acoustic.isna().any():
        raise ValueError('Recording does not provide every checkpoint acoustic feature')
    audio = acoustic.to_numpy().reshape(1, -1)
    age = np.array([[age_value]], dtype=np.float64)
    sex = np.array([[{'female': 0, 'male': 1}[sex_value]]], dtype=np.float64)

    fold_probabilities = []
    threshold: float | None = None
    for fold_directory in fold_directories:
        model, scaler, metadata = load_fold_model(run_directory, fold_directory, device)
        fold_probabilities.append(
            _fold_probabilities(model, scaler, embeddings, audio, age, sex, device),
        )
        threshold = metadata['probability_threshold']
    if threshold is None:
        raise RuntimeError('No probability threshold was loaded')
    probability = float(np.mean(fold_probabilities))
    result = {
        'cognitive_impairment_probability': probability,
        'prediction': int(probability >= threshold),
        'threshold': threshold,
    }
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite prediction output: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
