from __future__ import annotations

import dataclasses
import pathlib
import tomllib


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    text_hidden: tuple[int, int, int]
    text_dropout: tuple[float, float]
    audio_hidden: tuple[int, int]
    audio_dropout: tuple[float, float]
    demographics_hidden: tuple[int, int, int]
    demographics_dropout: float
    classifier_hidden: int
    classifier_dropout: float


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    outer_folds: int
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    use_class_weights: bool


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    embedder: str
    pooling: str
    max_length: int


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    features: FeatureConfig


def _as_int_tuple(values: list[int], field_name: str) -> tuple[int, ...]:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f'{field_name} must contain positive integers')
    return tuple(values)


def _as_float_tuple(values: list[float], field_name: str) -> tuple[float, ...]:
    if any(value < 0 or value >= 1 for value in values):
        raise ValueError(f'{field_name} must contain values in [0, 1)')
    return tuple(values)


def load_experiment_config(path: pathlib.Path) -> ExperimentConfig:
    with path.open('rb') as config_file:
        raw = tomllib.load(config_file)

    model_raw = raw['model']
    training_raw = raw['training']
    features_raw = raw['features']

    text_hidden = _as_int_tuple(model_raw['text_hidden'], 'model.text_hidden')
    audio_hidden = _as_int_tuple(model_raw['audio_hidden'], 'model.audio_hidden')
    demographics_hidden = _as_int_tuple(
        model_raw['demographics_hidden'],
        'model.demographics_hidden',
    )
    text_dropout = _as_float_tuple(model_raw['text_dropout'], 'model.text_dropout')
    audio_dropout = _as_float_tuple(model_raw['audio_dropout'], 'model.audio_dropout')
    if len(text_hidden) != 3 or len(text_dropout) != 2:
        raise ValueError('The legacy text branch requires three widths and two dropout values')
    if len(audio_hidden) != 2 or len(audio_dropout) != 2:
        raise ValueError('The legacy audio branch requires two widths and two dropout values')
    if len(demographics_hidden) != 3:
        raise ValueError('The legacy demographics branch requires three widths')

    model = ModelConfig(
        text_hidden=(text_hidden[0], text_hidden[1], text_hidden[2]),
        text_dropout=(text_dropout[0], text_dropout[1]),
        audio_hidden=(audio_hidden[0], audio_hidden[1]),
        audio_dropout=(audio_dropout[0], audio_dropout[1]),
        demographics_hidden=(
            demographics_hidden[0],
            demographics_hidden[1],
            demographics_hidden[2],
        ),
        demographics_dropout=model_raw['demographics_dropout'],
        classifier_hidden=model_raw['classifier_hidden'],
        classifier_dropout=model_raw['classifier_dropout'],
    )
    training = TrainingConfig(**training_raw)
    features = FeatureConfig(**features_raw)

    if training.outer_folds < 3:
        raise ValueError('training.outer_folds must be at least three')
    if training.max_epochs <= 0 or training.patience <= 0 or training.batch_size <= 0:
        raise ValueError('Epoch, patience, and batch-size values must be positive')
    if features.pooling not in {'mean', 'max'}:
        raise ValueError("features.pooling must be either 'mean' or 'max'")

    return ExperimentConfig(model=model, training=training, features=features)
