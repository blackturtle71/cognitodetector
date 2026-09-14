from __future__ import annotations

import copy
import dataclasses
import logging
import pathlib
import typing

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch import device as TorchDevice
from xgboost import XGBClassifier

from cognitive_impairments.artifacts import ScalerState
from cognitive_impairments.artifacts import modalities_to_dict, save_neural_fold, write_json
from cognitive_impairments.config import ExperimentConfig, ModelConfig, SelectionConfig
from cognitive_impairments.evaluation import calculate_binary_metrics
from cognitive_impairments.model import EXPERIMENT_MODALITIES, Modalities, MultimodalNet
from cognitive_impairments.training import DEFAULT_PROBABILITY_THRESHOLD
from cognitive_impairments.training import ModelArrays
from cognitive_impairments.training import _build_loader
from cognitive_impairments.training import _evaluate_loss
from cognitive_impairments.training import _loss_function
from cognitive_impairments.training import _predict_probabilities
from cognitive_impairments.training import _train_epoch
from cognitive_impairments.training import build_optimizer
from cognitive_impairments.training import create_grouped_splits
from cognitive_impairments.training import fit_scaler
from cognitive_impairments.training import set_reproducible_seed
from cognitive_impairments.training import transform_arrays


logger = logging.getLogger(__name__)

FULL_MODEL = Modalities(text=True, audio=True, demographics=True)
BASELINE_EXPERIMENTS = ('dummy_prevalence', 'logistic_all', 'xgboost_all')


@dataclasses.dataclass(frozen=True)
class NeuralCandidate:
    candidate_id: str
    model: ModelConfig
    optimizer: str
    use_class_weights: bool


@dataclasses.dataclass(frozen=True)
class EpochSelection:
    selected_epochs: int
    validation_loss: float
    validation_roc_auc: float
    history: list[dict[str, float | int]]


@dataclasses.dataclass(frozen=True)
class PaperTrainingResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    selections: pd.DataFrame
    history: pd.DataFrame
    splits: pd.DataFrame


def compact_model_config(config: ModelConfig) -> ModelConfig:
    widths = (
        *config.text_hidden,
        *config.audio_hidden,
        *config.demographics_hidden,
        config.classifier_hidden,
    )
    if any(width < 2 for width in widths):
        raise ValueError('Compact capacity requires every current width to be at least two')
    return dataclasses.replace(
        config,
        text_hidden=tuple(width // 2 for width in config.text_hidden),
        audio_hidden=tuple(width // 2 for width in config.audio_hidden),
        demographics_hidden=tuple(width // 2 for width in config.demographics_hidden),
        classifier_hidden=config.classifier_hidden // 2,
    )


def build_neural_candidates(
    model_config: ModelConfig,
    selection_config: SelectionConfig,
) -> tuple[NeuralCandidate, ...]:
    candidates = []
    for capacity in selection_config.capacities:
        candidate_model = (
            compact_model_config(model_config) if capacity == 'compact' else model_config
        )
        for optimizer_name in selection_config.optimizers:
            for weighted in selection_config.class_weights:
                weighting = 'weighted' if weighted else 'unweighted'
                candidate_id = f'{capacity}_{optimizer_name}_{weighting}'
                candidates.append(
                    NeuralCandidate(
                        candidate_id=candidate_id,
                        model=candidate_model,
                        optimizer=optimizer_name,
                        use_class_weights=weighted,
                    ),
                )
    return tuple(candidates)


def _new_candidate_model(
    arrays: ModelArrays,
    candidate: NeuralCandidate,
    modalities: Modalities,
    device: TorchDevice,
) -> MultimodalNet:
    return MultimodalNet(
        text_dimension=arrays.embeddings.shape[1],
        audio_dimension=arrays.audio.shape[1],
        config=candidate.model,
        modalities=modalities,
    ).to(device)


def select_epoch(
    arrays: ModelArrays,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: ExperimentConfig,
    candidate: NeuralCandidate,
    modalities: Modalities,
    seed: int,
    device: TorchDevice,
) -> EpochSelection:
    set_reproducible_seed(seed)
    scaler = fit_scaler(arrays, train_indices)
    train_loader = _build_loader(
        transform_arrays(arrays, train_indices, scaler),
        config.training.batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _build_loader(
        transform_arrays(arrays, validation_indices, scaler),
        config.training.batch_size,
        shuffle=False,
        seed=seed,
    )
    model = _new_candidate_model(arrays, candidate, modalities, device)
    optimizer = build_optimizer(model, config.training, candidate.optimizer)
    training_loss = _loss_function(
        arrays.labels[train_indices],
        candidate.use_class_weights,
        device,
    )
    validation_loss_function = _loss_function(
        arrays.labels[train_indices],
        weighted=False,
        device=device,
    )

    best_loss = float('inf')
    selected_epochs = 1
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, training_loss, device)
        validation_loss = _evaluate_loss(
            model,
            validation_loader,
            validation_loss_function,
            device,
        )
        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
            },
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            selected_epochs = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.training.patience:
            break

    model.load_state_dict(best_state)
    probabilities = _predict_probabilities(model, validation_loader, device)
    validation_roc_auc = float(
        roc_auc_score(arrays.labels[validation_indices], probabilities),
    )
    return EpochSelection(
        selected_epochs=selected_epochs,
        validation_loss=best_loss,
        validation_roc_auc=validation_roc_auc,
        history=history,
    )


def _candidate_sort_key(
    candidate: NeuralCandidate,
    selection: EpochSelection,
) -> tuple[float, float, str]:
    return (-selection.validation_roc_auc, selection.validation_loss, candidate.candidate_id)


def _inner_partition(
    arrays: ModelArrays,
    outer_train_indices: np.ndarray,
    outer_fold: int,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    relative_splits = create_grouped_splits(
        arrays.labels[outer_train_indices],
        arrays.groups[outer_train_indices],
        folds - 1,
        seed,
    )
    train_relative, validation_relative = relative_splits[outer_fold % len(relative_splits)]
    return outer_train_indices[train_relative], outer_train_indices[validation_relative]


def select_candidate(
    arrays: ModelArrays,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: ExperimentConfig,
    candidates: tuple[NeuralCandidate, ...],
    seed: int,
    device: TorchDevice,
) -> tuple[NeuralCandidate, EpochSelection, list[dict[str, typing.Any]]]:
    evaluated = []
    rows = []
    for candidate in candidates:
        selection = select_epoch(
            arrays,
            train_indices,
            validation_indices,
            config,
            candidate,
            FULL_MODEL,
            seed,
            device,
        )
        evaluated.append((candidate, selection))
        rows.append(
            {
                'candidate_id': candidate.candidate_id,
                'validation_roc_auc': selection.validation_roc_auc,
                'validation_loss': selection.validation_loss,
                'selected_epochs': selection.selected_epochs,
            },
        )
    candidate, selection = min(
        evaluated,
        key=lambda value: _candidate_sort_key(value[0], value[1]),
    )
    for row in rows:
        row['selection_stage'] = 'neural_candidate'
        row['selected'] = row['candidate_id'] == candidate.candidate_id
    return candidate, selection, rows


def _refit_neural_model(
    arrays: ModelArrays,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: ExperimentConfig,
    candidate: NeuralCandidate,
    modalities: Modalities,
    selected_epochs: int,
    seed: int,
    device: TorchDevice,
) -> tuple[MultimodalNet, ScalerState, np.ndarray, list[dict[str, float | int]]]:
    set_reproducible_seed(seed)
    scaler = fit_scaler(arrays, train_indices)
    train_loader = _build_loader(
        transform_arrays(arrays, train_indices, scaler),
        config.training.batch_size,
        shuffle=True,
        seed=seed,
    )
    test_loader = _build_loader(
        transform_arrays(arrays, test_indices, scaler),
        config.training.batch_size,
        shuffle=False,
        seed=seed,
    )
    model = _new_candidate_model(arrays, candidate, modalities, device)
    optimizer = build_optimizer(model, config.training, candidate.optimizer)
    loss_function = _loss_function(
        arrays.labels[train_indices],
        candidate.use_class_weights,
        device,
    )
    history = []
    for epoch in range(1, selected_epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_function, device)
        history.append({'epoch': epoch, 'train_loss': train_loss})
    probabilities = _predict_probabilities(model, test_loader, device)
    return model, scaler, probabilities, history


def _feature_matrix(arrays: ModelArrays) -> np.ndarray:
    return np.concatenate((arrays.embeddings, arrays.audio, arrays.age, arrays.sex), axis=1)


def _balanced_sample_weights(labels: np.ndarray, weighted: bool) -> np.ndarray | None:
    if not weighted:
        return None
    return compute_sample_weight(class_weight='balanced', y=labels)


def _select_baseline_weighting(
    model_name: str,
    arrays: ModelArrays,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
) -> tuple[bool, list[dict[str, typing.Any]]]:
    features = _feature_matrix(arrays)
    rows = []
    for weighted in (False, True):
        if model_name == 'logistic_all':
            scaler = StandardScaler().fit(features[train_indices])
            model = LogisticRegression(
                class_weight='balanced' if weighted else None,
                random_state=seed,
            )
            model.fit(scaler.transform(features[train_indices]), arrays.labels[train_indices])
            probabilities = model.predict_proba(
                scaler.transform(features[validation_indices]),
            )[:, 1]
        elif model_name == 'xgboost_all':
            model = XGBClassifier(random_state=seed)
            model.fit(
                features[train_indices],
                arrays.labels[train_indices],
                sample_weight=_balanced_sample_weights(arrays.labels[train_indices], weighted),
            )
            probabilities = model.predict_proba(features[validation_indices])[:, 1]
        else:
            raise ValueError(f'Unsupported baseline: {model_name}')
        rows.append(
            {
                'candidate_id': 'weighted' if weighted else 'unweighted',
                'validation_roc_auc': float(
                    roc_auc_score(arrays.labels[validation_indices], probabilities),
                ),
                'validation_loss': float(
                    log_loss(arrays.labels[validation_indices], probabilities, labels=[0, 1]),
                ),
                'weighted': weighted,
            },
        )
    chosen = min(
        rows,
        key=lambda row: (
            -row['validation_roc_auc'],
            row['validation_loss'],
            row['candidate_id'],
        ),
    )
    for row in rows:
        row['selection_stage'] = 'baseline_weighting'
        row['selected'] = row['weighted'] == chosen['weighted']
    return bool(chosen['weighted']), rows


def _fit_baseline(
    experiment: str,
    arrays: ModelArrays,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    weighted: bool,
    seed: int,
    checkpoint_directory: pathlib.Path,
) -> np.ndarray:
    features = _feature_matrix(arrays)
    checkpoint_directory.mkdir(parents=True)
    if experiment == 'dummy_prevalence':
        model = DummyClassifier(strategy='prior')
        model.fit(features[train_indices], arrays.labels[train_indices])
        write_json(
            checkpoint_directory / 'model.json',
            {'class_prior': model.class_prior_.tolist(), 'classes': model.classes_.tolist()},
        )
        return model.predict_proba(features[test_indices])[:, 1]
    if experiment == 'logistic_all':
        scaler = StandardScaler().fit(features[train_indices])
        model = LogisticRegression(
            class_weight='balanced' if weighted else None,
            random_state=seed,
        )
        model.fit(scaler.transform(features[train_indices]), arrays.labels[train_indices])
        np.savez(
            checkpoint_directory / 'model.npz',
            mean=scaler.mean_,
            scale=scaler.scale_,
            coefficients=model.coef_,
            intercept=model.intercept_,
            classes=model.classes_,
        )
        return model.predict_proba(scaler.transform(features[test_indices]))[:, 1]
    if experiment == 'xgboost_all':
        model = XGBClassifier(random_state=seed)
        model.fit(
            features[train_indices],
            arrays.labels[train_indices],
            sample_weight=_balanced_sample_weights(arrays.labels[train_indices], weighted),
        )
        model.save_model(checkpoint_directory / 'model.ubj')
        return model.predict_proba(features[test_indices])[:, 1]
    raise ValueError(f'Unsupported baseline: {experiment}')


def _prediction_rows(
    experiment: str,
    fold: int,
    arrays: ModelArrays,
    test_indices: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, typing.Any]]:
    predictions = (probabilities >= DEFAULT_PROBABILITY_THRESHOLD).astype(int)
    return [
        {
            'experiment': experiment,
            'fold': fold,
            'recording_id': str(arrays.recording_ids[index]),
            'participant_id': str(arrays.groups[index]),
            'label': int(arrays.labels[index]),
            'probability': float(probability),
            'prediction': int(prediction),
        }
        for index, probability, prediction in zip(
            test_indices,
            probabilities,
            predictions,
            strict=True,
        )
    ]


def run_paper_seed(
    arrays: ModelArrays,
    config: ExperimentConfig,
    manifest: dict[str, typing.Any],
    seed: int,
    run_directory: pathlib.Path,
    device: TorchDevice,
) -> PaperTrainingResult:
    if config.selection is None:
        raise ValueError('Paper training requires selection configuration')
    candidates = build_neural_candidates(config.model, config.selection)
    outer_splits = create_grouped_splits(
        arrays.labels,
        arrays.groups,
        config.training.outer_folds,
        seed,
    )
    prediction_rows: list[dict[str, typing.Any]] = []
    selection_rows: list[dict[str, typing.Any]] = []
    history_rows: list[dict[str, typing.Any]] = []
    split_rows: list[dict[str, typing.Any]] = []

    for fold, (outer_train, test_indices) in enumerate(outer_splits):
        logger.info('Training paper seed=%s fold=%s/%s', seed, fold + 1, len(outer_splits))
        inner_train, validation = _inner_partition(
            arrays,
            outer_train,
            fold,
            config.training.outer_folds,
            seed,
        )
        split_rows.extend(
            {
                'recording_id': str(arrays.recording_ids[index]),
                'participant_id': str(arrays.groups[index]),
                'outer_fold': fold,
            }
            for index in test_indices
        )
        chosen, full_selection, candidate_rows = select_candidate(
            arrays,
            inner_train,
            validation,
            config,
            candidates,
            seed,
            device,
        )
        selection_rows.extend(
            {'fold': fold, 'experiment': FULL_MODEL.name, **row}
            for row in candidate_rows
        )

        for modalities in EXPERIMENT_MODALITIES:
            selection = full_selection
            if modalities != FULL_MODEL:
                selection = select_epoch(
                    arrays,
                    inner_train,
                    validation,
                    config,
                    chosen,
                    modalities,
                    seed,
                    device,
                )
            model, scaler, probabilities, refit_history = _refit_neural_model(
                arrays,
                outer_train,
                test_indices,
                config,
                chosen,
                modalities,
                selection.selected_epochs,
                seed,
                device,
            )
            prediction_rows.extend(
                _prediction_rows(modalities.name, fold, arrays, test_indices, probabilities),
            )
            selection_rows.append(
                {
                    'fold': fold,
                    'experiment': modalities.name,
                    'selection_stage': 'modality_epoch',
                    'candidate_id': chosen.candidate_id,
                    'selected': True,
                    'validation_roc_auc': selection.validation_roc_auc,
                    'validation_loss': selection.validation_loss,
                    'selected_epochs': selection.selected_epochs,
                },
            )
            history_rows.extend(
                {
                    'fold': fold,
                    'experiment': modalities.name,
                    'stage': stage,
                    **row,
                }
                for stage, history in (
                    ('selection', selection.history),
                    ('refit', refit_history),
                )
                for row in history
            )
            save_neural_fold(
                run_directory,
                modalities.name,
                fold,
                model.state_dict(),
                scaler,
                {
                    'experiment': modalities.name,
                    'fold': fold,
                    'modalities': modalities_to_dict(modalities),
                    'candidate_id': chosen.candidate_id,
                    'model_config': dataclasses.asdict(chosen.model),
                    'optimizer': chosen.optimizer,
                    'use_class_weights': chosen.use_class_weights,
                    'selected_epochs': selection.selected_epochs,
                    'text_dimension': arrays.embeddings.shape[1],
                    'audio_dimension': arrays.audio.shape[1],
                    'acoustic_features': manifest['acoustic_features'],
                    'embedder': manifest['embedder'],
                    'embedder_revision': manifest.get('embedder_revision'),
                    'pooling': manifest['pooling'],
                    'embedding_normalized': manifest.get('embedding_normalized', False),
                    'max_length': manifest['max_length'],
                    'acoustic_aggregation': manifest.get('acoustic_aggregation'),
                    'probability_threshold': DEFAULT_PROBABILITY_THRESHOLD,
                },
            )

        for experiment in BASELINE_EXPERIMENTS:
            weighted = False
            baseline_rows: list[dict[str, typing.Any]] = []
            if experiment != 'dummy_prevalence':
                weighted, baseline_rows = _select_baseline_weighting(
                    experiment,
                    arrays,
                    inner_train,
                    validation,
                    seed,
                )
                selection_rows.extend(
                    {'fold': fold, 'experiment': experiment, **row}
                    for row in baseline_rows
                )
            probabilities = _fit_baseline(
                experiment,
                arrays,
                outer_train,
                test_indices,
                weighted,
                seed,
                run_directory / 'checkpoints' / experiment / f'fold_{fold}',
            )
            write_json(
                run_directory / 'checkpoints' / experiment / f'fold_{fold}' / 'metadata.json',
                {
                    'experiment': experiment,
                    'fold': fold,
                    'weighted': weighted,
                    'probability_threshold': DEFAULT_PROBABILITY_THRESHOLD,
                },
            )
            prediction_rows.extend(
                _prediction_rows(experiment, fold, arrays, test_indices, probabilities),
            )

    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for experiment, experiment_predictions in predictions.groupby('experiment', sort=True):
        metrics = calculate_binary_metrics(
            experiment_predictions['label'].to_numpy(),
            experiment_predictions['probability'].to_numpy(),
            DEFAULT_PROBABILITY_THRESHOLD,
        )
        metric_rows.append({'experiment': experiment, **dataclasses.asdict(metrics)})
    return PaperTrainingResult(
        predictions=predictions,
        metrics=pd.DataFrame(metric_rows),
        selections=pd.DataFrame(selection_rows),
        history=pd.DataFrame(history_rows),
        splits=pd.DataFrame(split_rows),
    )
