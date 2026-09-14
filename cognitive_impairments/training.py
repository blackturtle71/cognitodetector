from __future__ import annotations

import dataclasses
import logging
import pathlib
import random
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import Generator, Tensor, device as TorchDevice, float32, long
from torch import manual_seed, nn, no_grad, optim, softmax, tensor
from torch.utils.data import DataLoader, TensorDataset

from cognitive_impairments.artifacts import ScalerState
from cognitive_impairments.artifacts import copy_run_inputs
from cognitive_impairments.artifacts import create_run_directory
from cognitive_impairments.artifacts import modalities_to_dict
from cognitive_impairments.artifacts import save_neural_fold
from cognitive_impairments.artifacts import write_json
from cognitive_impairments.config import ExperimentConfig, TrainingConfig
from cognitive_impairments.config import load_experiment_config
from cognitive_impairments.data import read_feature_table
from cognitive_impairments.evaluation import calculate_binary_metrics
from cognitive_impairments.model import EXPERIMENT_MODALITIES, Modalities, MultimodalNet


DEFAULT_PROBABILITY_THRESHOLD = 0.5
LOGISTIC_EXPERIMENT_NAME = 'logistic_all'

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ModelArrays:
    embeddings: np.ndarray
    audio: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    recording_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    model: MultimodalNet
    selected_epochs: int
    selection_history: list[dict[str, float | int]]
    refit_history: list[dict[str, float | int]]
    scaler: ScalerState


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)


def arrays_from_frame(frame: pd.DataFrame, acoustic_features: list[str]) -> ModelArrays:
    if frame['label'].isna().any():
        raise ValueError('Training data contains unlabeled records')
    sex = frame['sex'].map({'female': 0, 'male': 1})
    if sex.isna().any():
        raise ValueError('Training data contains an unsupported sex value')
    return ModelArrays(
        embeddings=np.stack(frame['embedding'].map(lambda value: np.asarray(value).reshape(-1))),
        audio=frame[acoustic_features].to_numpy(dtype=np.float64),
        age=frame['age'].to_numpy(dtype=np.float64).reshape(-1, 1),
        sex=sex.to_numpy(dtype=np.float64).reshape(-1, 1),
        labels=frame['label'].to_numpy(dtype=np.int64),
        groups=frame['participant_id'].astype(str).to_numpy(),
        recording_ids=frame['recording_id'].astype(str).to_numpy(),
    )


def fit_scaler(arrays: ModelArrays, indices: np.ndarray) -> ScalerState:
    audio_scaler = StandardScaler().fit(arrays.audio[indices])
    age_scaler = StandardScaler().fit(arrays.age[indices])
    return ScalerState(
        audio_mean=audio_scaler.mean_,
        audio_scale=audio_scaler.scale_,
        age_mean=age_scaler.mean_,
        age_scale=age_scaler.scale_,
    )


def transform_arrays(
    arrays: ModelArrays,
    indices: np.ndarray,
    scaler: ScalerState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    audio = (arrays.audio[indices] - scaler.audio_mean) / scaler.audio_scale
    age = (arrays.age[indices] - scaler.age_mean) / scaler.age_scale
    return (
        arrays.embeddings[indices],
        audio,
        age,
        arrays.sex[indices],
        arrays.labels[indices],
    )


def build_optimizer(model: MultimodalNet, config: TrainingConfig) -> optim.Optimizer:
    return optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    placeholders = np.zeros(shape=(len(labels), 1))
    return list(splitter.split(placeholders, labels, groups))


def _build_loader(
    transformed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, ...]]:
    dataset = TensorDataset(
        *(tensor(values, dtype=dtype) for values, dtype in zip(
            transformed,
            (
                float32,
                float32,
                float32,
                float32,
                long,
            ),
            strict=True,
        )),
    )
    generator = Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _loss_function(labels: np.ndarray, weighted: bool, device: TorchDevice) -> nn.Module:
    if not weighted:
        return nn.CrossEntropyLoss()
    classes, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError('Both classes must be present in every training partition')
    weights = len(labels) / (len(classes) * counts)
    return nn.CrossEntropyLoss(weight=tensor(weights, dtype=float32, device=device))


def _train_epoch(
    model: MultimodalNet,
    loader: DataLoader[tuple[Tensor, ...]],
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    device: TorchDevice,
) -> float:
    model.train()
    total_loss = 0.0
    samples = 0
    for embeddings, audio, age, sex, labels in loader:
        embeddings = embeddings.to(device)
        audio = audio.to(device)
        age = age.to(device)
        sex = sex.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings, audio, age, sex)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        samples += batch_size
    return total_loss / samples


def _evaluate_loss(
    model: MultimodalNet,
    loader: DataLoader[tuple[Tensor, ...]],
    loss_function: nn.Module,
    device: TorchDevice,
) -> float:
    model.eval()
    total_loss = 0.0
    samples = 0
    with no_grad():
        for embeddings, audio, age, sex, labels in loader:
            labels = labels.to(device)
            logits = model(
                embeddings.to(device),
                audio.to(device),
                age.to(device),
                sex.to(device),
            )
            loss = loss_function(logits, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            samples += batch_size
    return total_loss / samples


def _predict_probabilities(
    model: MultimodalNet,
    loader: DataLoader[tuple[Tensor, ...]],
    device: TorchDevice,
) -> np.ndarray:
    model.eval()
    probabilities = []
    with no_grad():
        for embeddings, audio, age, sex, _labels in loader:
            logits = model(
                embeddings.to(device),
                audio.to(device),
                age.to(device),
                sex.to(device),
            )
            probabilities.append(softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def _new_model(
    arrays: ModelArrays,
    config: ExperimentConfig,
    modalities: Modalities,
    device: TorchDevice,
) -> MultimodalNet:
    model = MultimodalNet(
        text_dimension=arrays.embeddings.shape[1],
        audio_dimension=arrays.audio.shape[1],
        config=config.model,
        modalities=modalities,
    )
    return model.to(device)


def train_neural_fold(
    arrays: ModelArrays,
    outer_train_indices: np.ndarray,
    test_indices: np.ndarray,
    outer_fold: int,
    config: ExperimentConfig,
    modalities: Modalities,
    seed: int,
    device: TorchDevice,
) -> tuple[TrainingResult, np.ndarray]:
    inner_splitter = StratifiedGroupKFold(
        n_splits=config.training.outer_folds - 1,
        shuffle=True,
        random_state=seed,
    )
    inner_splits = list(
        inner_splitter.split(
            arrays.embeddings[outer_train_indices],
            arrays.labels[outer_train_indices],
            arrays.groups[outer_train_indices],
        ),
    )
    inner_train_relative, validation_relative = inner_splits[outer_fold % len(inner_splits)]
    inner_train_indices = outer_train_indices[inner_train_relative]
    validation_indices = outer_train_indices[validation_relative]

    set_reproducible_seed(seed)
    selection_scaler = fit_scaler(arrays, inner_train_indices)
    selection_train_loader = _build_loader(
        transform_arrays(arrays, inner_train_indices, selection_scaler),
        config.training.batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _build_loader(
        transform_arrays(arrays, validation_indices, selection_scaler),
        config.training.batch_size,
        shuffle=False,
        seed=seed,
    )
    selection_model = _new_model(arrays, config, modalities, device)
    selection_optimizer = build_optimizer(selection_model, config.training)
    selection_loss = _loss_function(
        arrays.labels[inner_train_indices],
        config.training.use_class_weights,
        device,
    )

    best_validation_loss = float('inf')
    selected_epochs = 1
    epochs_without_improvement = 0
    selection_history: list[dict[str, float | int]] = []
    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = _train_epoch(
            selection_model,
            selection_train_loader,
            selection_optimizer,
            selection_loss,
            device,
        )
        validation_loss = _evaluate_loss(
            selection_model,
            validation_loader,
            selection_loss,
            device,
        )
        selection_history.append(
            {'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss},
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            selected_epochs = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.training.patience:
            break

    set_reproducible_seed(seed)
    refit_scaler = fit_scaler(arrays, outer_train_indices)
    refit_loader = _build_loader(
        transform_arrays(arrays, outer_train_indices, refit_scaler),
        config.training.batch_size,
        shuffle=True,
        seed=seed,
    )
    test_loader = _build_loader(
        transform_arrays(arrays, test_indices, refit_scaler),
        config.training.batch_size,
        shuffle=False,
        seed=seed,
    )
    model = _new_model(arrays, config, modalities, device)
    optimizer = build_optimizer(model, config.training)
    refit_loss = _loss_function(
        arrays.labels[outer_train_indices],
        config.training.use_class_weights,
        device,
    )
    refit_history = []
    for epoch in range(1, selected_epochs + 1):
        train_loss = _train_epoch(model, refit_loader, optimizer, refit_loss, device)
        refit_history.append({'epoch': epoch, 'train_loss': train_loss})

    probabilities = _predict_probabilities(model, test_loader, device)
    result = TrainingResult(
        model=model,
        selected_epochs=selected_epochs,
        selection_history=selection_history,
        refit_history=refit_history,
        scaler=refit_scaler,
    )
    return result, probabilities


def _all_feature_matrix(arrays: ModelArrays) -> np.ndarray:
    return np.concatenate(
        (arrays.embeddings, arrays.audio, arrays.age, arrays.sex),
        axis=1,
    )


def _run_logistic_fold(
    arrays: ModelArrays,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    run_directory: pathlib.Path,
    fold: int,
) -> np.ndarray:
    features = _all_feature_matrix(arrays)
    scaler = StandardScaler().fit(features[train_indices])
    train_features = scaler.transform(features[train_indices])
    test_features = scaler.transform(features[test_indices])
    model = LogisticRegression(class_weight='balanced', random_state=seed)
    model.fit(train_features, arrays.labels[train_indices])
    directory = run_directory / 'checkpoints' / LOGISTIC_EXPERIMENT_NAME / f'fold_{fold}'
    directory.mkdir(parents=True)
    np.savez(
        directory / 'model.npz',
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficients=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
    )
    write_json(
        directory / 'metadata.json',
        {
            'experiment': LOGISTIC_EXPERIMENT_NAME,
            'fold': fold,
            'probability_threshold': DEFAULT_PROBABILITY_THRESHOLD,
        },
    )
    return model.predict_proba(test_features)[:, 1]


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
            'recording_id': arrays.recording_ids[index],
            'participant_id': arrays.groups[index],
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


def _save_summary(
    run_directory: pathlib.Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    metric_columns = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
    summary = metrics.groupby('experiment')[list(metric_columns)].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{statistic}' for metric, statistic in summary.columns]
    summary.reset_index().to_csv(run_directory / 'metrics_summary.csv', index=False)

    confusion_columns = ('true_negative', 'false_positive', 'false_negative', 'true_positive')
    confusion = metrics.groupby('experiment')[list(confusion_columns)].sum().reset_index()
    confusion.to_csv(run_directory / 'confusion_matrices.csv', index=False)

    plot_directory = run_directory / 'plots'
    plot_directory.mkdir()

    figure, axis = plt.subplots()
    plotted = summary.reset_index()
    axis.barh(plotted['experiment'], plotted['roc_auc_mean'])
    axis.set_xlabel('Mean ROC AUC')
    figure.tight_layout()
    figure.savefig(plot_directory / 'ablation_roc_auc.png')
    plt.close(figure)

    figure, axis = plt.subplots()
    for experiment, experiment_predictions in predictions.groupby('experiment'):
        RocCurveDisplay.from_predictions(
            experiment_predictions['label'],
            experiment_predictions['probability'],
            name=experiment,
            ax=axis,
        )
    figure.tight_layout()
    figure.savefig(plot_directory / 'oof_roc_curves.png')
    plt.close(figure)

    for experiment, experiment_predictions in predictions.groupby('experiment'):
        figure, axis = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            experiment_predictions['label'],
            experiment_predictions['prediction'],
            display_labels=('Control', 'Cognitive impairment'),
            ax=axis,
        )
        figure.tight_layout()
        figure.savefig(plot_directory / f'{experiment}_confusion_matrix.png')
        plt.close(figure)


def run_experiment_suite(
    dataset_path: pathlib.Path,
    config_path: pathlib.Path,
    run_directory: pathlib.Path,
    seed: int,
    device: TorchDevice,
) -> None:
    frame, manifest = read_feature_table(dataset_path)
    if manifest.get('corpus') != 'pitt':
        raise ValueError('The dissertation experiment suite requires a Pitt feature table')
    config = load_experiment_config(config_path)
    arrays = arrays_from_frame(frame, manifest['acoustic_features'])

    create_run_directory(run_directory)
    copy_run_inputs(run_directory, config_path, manifest, seed)
    outer_splits = create_grouped_splits(
        arrays.labels,
        arrays.groups,
        config.training.outer_folds,
        seed,
    )
    split_rows = []
    for fold, (_train_indices, test_indices) in enumerate(outer_splits):
        split_rows.extend(
            {
                'recording_id': arrays.recording_ids[index],
                'participant_id': arrays.groups[index],
                'outer_fold': fold,
            }
            for index in test_indices
        )
    pd.DataFrame(split_rows).to_csv(run_directory / 'splits.csv', index=False)

    metric_rows: list[dict[str, typing.Any]] = []
    prediction_rows: list[dict[str, typing.Any]] = []
    history_rows: list[dict[str, typing.Any]] = []
    for modalities in EXPERIMENT_MODALITIES:
        for fold, (train_indices, test_indices) in enumerate(outer_splits):
            logger.info(
                'Training experiment=%s fold=%s/%s',
                modalities.name,
                fold + 1,
                len(outer_splits),
            )
            result, probabilities = train_neural_fold(
                arrays=arrays,
                outer_train_indices=train_indices,
                test_indices=test_indices,
                outer_fold=fold,
                config=config,
                modalities=modalities,
                seed=seed,
                device=device,
            )
            metrics = calculate_binary_metrics(
                arrays.labels[test_indices],
                probabilities,
                DEFAULT_PROBABILITY_THRESHOLD,
            )
            metric_rows.append(
                {
                    'experiment': modalities.name,
                    'fold': fold,
                    'selected_epochs': result.selected_epochs,
                    **dataclasses.asdict(metrics),
                },
            )
            prediction_rows.extend(
                _prediction_rows(modalities.name, fold, arrays, test_indices, probabilities),
            )
            history_rows.extend(
                {
                    'experiment': modalities.name,
                    'fold': fold,
                    'stage': stage,
                    **row,
                }
                for stage, history in (
                    ('selection', result.selection_history),
                    ('refit', result.refit_history),
                )
                for row in history
            )
            save_neural_fold(
                run_directory=run_directory,
                experiment_name=modalities.name,
                fold=fold,
                state_dict=result.model.state_dict(),
                scaler=result.scaler,
                metadata={
                    'experiment': modalities.name,
                    'fold': fold,
                    'modalities': modalities_to_dict(modalities),
                    'selected_epochs': result.selected_epochs,
                    'text_dimension': arrays.embeddings.shape[1],
                    'audio_dimension': arrays.audio.shape[1],
                    'acoustic_features': manifest['acoustic_features'],
                    'embedder': manifest['embedder'],
                    'pooling': manifest['pooling'],
                    'max_length': manifest['max_length'],
                    'probability_threshold': DEFAULT_PROBABILITY_THRESHOLD,
                },
            )

    for fold, (train_indices, test_indices) in enumerate(outer_splits):
        logger.info(
            'Training experiment=%s fold=%s/%s',
            LOGISTIC_EXPERIMENT_NAME,
            fold + 1,
            len(outer_splits),
        )
        probabilities = _run_logistic_fold(
            arrays,
            train_indices,
            test_indices,
            seed,
            run_directory,
            fold,
        )
        metrics = calculate_binary_metrics(
            arrays.labels[test_indices],
            probabilities,
            DEFAULT_PROBABILITY_THRESHOLD,
        )
        metric_rows.append(
            {'experiment': LOGISTIC_EXPERIMENT_NAME, 'fold': fold, **dataclasses.asdict(metrics)},
        )
        prediction_rows.extend(
            _prediction_rows(LOGISTIC_EXPERIMENT_NAME, fold, arrays, test_indices, probabilities),
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_directory / 'metrics_by_fold.csv', index=False)
    predictions_frame = pd.DataFrame(prediction_rows)
    predictions_frame.to_csv(run_directory / 'oof_predictions.csv', index=False)
    pd.DataFrame(history_rows).to_csv(run_directory / 'training_history.csv', index=False)
    _save_summary(run_directory, metrics_frame, predictions_frame)
