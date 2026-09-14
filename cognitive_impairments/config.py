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
    revision: str | None
    pooling: str
    normalize: bool
    max_length: int
    acoustic_aggregation: str


@dataclasses.dataclass(frozen=True)
class SelectionConfig:
    capacities: tuple[str, ...]
    optimizers: tuple[str, ...]
    class_weights: tuple[bool, ...]


@dataclasses.dataclass(frozen=True)
class PaperConfig:
    seeds: tuple[int, ...]
    confidence_level: float
    bootstrap_replicates: int
    matching_seed: int
    shap_seed: int


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    features: FeatureConfig
    selection: SelectionConfig | None = None
    paper: PaperConfig | None = None


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
    features = FeatureConfig(
        embedder=features_raw['embedder'],
        revision=features_raw.get('revision'),
        pooling=features_raw['pooling'],
        normalize=features_raw.get('normalize', False),
        max_length=features_raw['max_length'],
        acoustic_aggregation=features_raw.get('acoustic_aggregation', 'utterance_mean'),
    )

    if training.outer_folds < 3:
        raise ValueError('training.outer_folds must be at least three')
    if training.max_epochs <= 0 or training.patience <= 0 or training.batch_size <= 0:
        raise ValueError('Epoch, patience, and batch-size values must be positive')
    if features.pooling not in {'cls', 'mean', 'max'}:
        raise ValueError("features.pooling must be 'cls', 'mean', or 'max'")
    if features.acoustic_aggregation not in {'concatenate', 'utterance_mean'}:
        raise ValueError(
            "features.acoustic_aggregation must be 'concatenate' or 'utterance_mean'",
        )

    selection = None
    if 'selection' in raw:
        selection_raw = raw['selection']
        selection = SelectionConfig(
            capacities=tuple(selection_raw['capacities']),
            optimizers=tuple(selection_raw['optimizers']),
            class_weights=tuple(selection_raw['class_weights']),
        )
        if set(selection.capacities) - {'compact', 'current'}:
            raise ValueError('selection.capacities contains an unsupported value')
        if set(selection.optimizers) - {'sgd', 'adamw'}:
            raise ValueError('selection.optimizers contains an unsupported value')
        if not selection.capacities or len(set(selection.capacities)) != len(
            selection.capacities,
        ):
            raise ValueError('selection.capacities must contain unique values')
        if not selection.optimizers or len(set(selection.optimizers)) != len(
            selection.optimizers,
        ):
            raise ValueError('selection.optimizers must contain unique values')
        if not selection.class_weights or len(set(selection.class_weights)) != len(
            selection.class_weights,
        ):
            raise ValueError('selection.class_weights must contain unique values')

    paper = None
    if 'paper' in raw:
        paper_raw = raw['paper']
        paper = PaperConfig(
            seeds=tuple(paper_raw['seeds']),
            confidence_level=paper_raw['confidence_level'],
            bootstrap_replicates=paper_raw['bootstrap_replicates'],
            matching_seed=paper_raw['matching_seed'],
            shap_seed=paper_raw['shap_seed'],
        )
        if not paper.seeds or len(set(paper.seeds)) != len(paper.seeds):
            raise ValueError('paper.seeds must contain unique values')
        if not 0 < paper.confidence_level < 1:
            raise ValueError('paper.confidence_level must be between zero and one')
        if paper.bootstrap_replicates <= 0:
            raise ValueError('paper.bootstrap_replicates must be positive')
        if paper.shap_seed not in paper.seeds:
            raise ValueError('paper.shap_seed must be included in paper.seeds')

    return ExperimentConfig(
        model=model,
        training=training,
        features=features,
        selection=selection,
        paper=paper,
    )


def require_paper_protocol(config: ExperimentConfig) -> tuple[SelectionConfig, PaperConfig]:
    if config.selection is None or config.paper is None:
        raise ValueError('Paper runs require [selection] and [paper] configuration sections')
    return config.selection, config.paper
