from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from torch import Tensor, cat, device as TorchDevice, float32, nn, softmax, tensor

from cognitive_impairments.data import read_feature_table
from cognitive_impairments.prediction import FULL_MODEL_EXPERIMENT, load_fold_model


class ConcatenatedInputModel(nn.Module):
    def __init__(self, model: nn.Module, text_dimension: int, audio_dimension: int) -> None:
        super().__init__()
        self.model = model
        self.text_dimension = text_dimension
        self.audio_dimension = audio_dimension

    def forward(self, values: Tensor) -> Tensor:
        text_end = self.text_dimension
        audio_end = text_end + self.audio_dimension
        embeddings = values[:, :text_end]
        audio = values[:, text_end:audio_end]
        age = values[:, audio_end:audio_end + 1]
        sex = values[:, audio_end + 1:audio_end + 2]
        logits = self.model(embeddings, audio, age, sex)
        return softmax(logits, dim=1)[:, 1:2]


def _select_positive_class(values: object) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1])
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    if array.ndim == 3:
        return array[:, :, 1]
    if array.ndim == 2:
        return array
    raise ValueError(f'Unsupported SHAP output shape: {array.shape}')


def _full_input(
    frame: pd.DataFrame,
    indices: np.ndarray,
    acoustic_features: list[str],
    audio_mean: np.ndarray,
    audio_scale: np.ndarray,
    age_mean: np.ndarray,
    age_scale: np.ndarray,
    device: TorchDevice,
) -> Tensor:
    embeddings = np.stack(frame.iloc[indices]['embedding'].map(lambda value: np.asarray(value)))
    audio = frame.iloc[indices][acoustic_features].to_numpy(dtype=np.float64)
    age = frame.iloc[indices]['age'].to_numpy(dtype=np.float64).reshape(-1, 1)
    sex = (
        frame.iloc[indices]['sex']
        .map({'female': 0, 'male': 1})
        .to_numpy(dtype=np.float64)
        .reshape(-1, 1)
    )
    return cat(
        (
            tensor(embeddings, dtype=float32, device=device),
            tensor((audio - audio_mean) / audio_scale, dtype=float32, device=device),
            tensor((age - age_mean) / age_scale, dtype=float32, device=device),
            tensor(sex, dtype=float32, device=device),
        ),
        dim=1,
    )


def explain_cross_validation(
    run_directory: pathlib.Path,
    dataset_path: pathlib.Path,
    device: TorchDevice,
) -> None:
    output_directory = run_directory / 'explanations'
    if output_directory.exists():
        raise FileExistsError(f'Refusing to overwrite explanations: {output_directory}')
    output_directory.mkdir(parents=True)

    frame, manifest = read_feature_table(dataset_path)
    split_frame = pd.read_csv(run_directory / 'splits.csv', dtype={'recording_id': str})
    fold_by_recording = dict(zip(
        split_frame['recording_id'],
        split_frame['outer_fold'],
        strict=True,
    ))
    row_folds = frame['recording_id'].astype(str).map(fold_by_recording)
    if row_folds.isna().any():
        raise ValueError('Run split assignments do not cover the supplied feature table')

    feature_values = []
    shap_values = []
    checkpoint_root = run_directory / 'checkpoints' / FULL_MODEL_EXPERIMENT
    fold_directories = sorted(
        checkpoint_root.glob('fold_*'),
        key=lambda path: int(path.name.removeprefix('fold_')),
    )
    if not fold_directories:
        raise FileNotFoundError(f'No multimodal fold checkpoints found under {checkpoint_root}')
    for fold_directory in fold_directories:
        fold = int(fold_directory.name.removeprefix('fold_'))
        train_indices = np.flatnonzero(row_folds.to_numpy() != fold)
        test_indices = np.flatnonzero(row_folds.to_numpy() == fold)
        model, scaler, metadata = load_fold_model(run_directory, fold_directory, device)
        wrapped_model = ConcatenatedInputModel(
            model,
            text_dimension=metadata['text_dimension'],
            audio_dimension=metadata['audio_dimension'],
        ).to(device)
        background = _full_input(
            frame,
            train_indices,
            manifest['acoustic_features'],
            scaler.audio_mean,
            scaler.audio_scale,
            scaler.age_mean,
            scaler.age_scale,
            device,
        )
        test_values = _full_input(
            frame,
            test_indices,
            manifest['acoustic_features'],
            scaler.audio_mean,
            scaler.audio_scale,
            scaler.age_mean,
            scaler.age_scale,
            device,
        )
        explainer = shap.GradientExplainer(wrapped_model, background)
        fold_shap = _select_positive_class(explainer.shap_values(test_values))
        fold_values = test_values.detach().cpu().numpy()
        np.savez(
            output_directory / f'fold_{fold}.npz',
            shap_values=fold_shap,
            feature_values=fold_values,
            recording_ids=frame.iloc[test_indices]['recording_id'].astype(str).to_numpy(),
        )
        shap_values.append(fold_shap)
        feature_values.append(fold_values)

    all_shap = np.concatenate(shap_values)
    all_values = np.concatenate(feature_values)
    text_dimension = manifest['embedding_dimension']
    audio_dimension = len(manifest['acoustic_features'])
    feature_names = [
        *(f'embedding_{index}' for index in range(text_dimension)),
        *manifest['acoustic_features'],
        'age',
        'sex',
    ]
    importance = np.mean(np.abs(all_shap), axis=0)
    pd.DataFrame(
        {'feature': feature_names, 'mean_absolute_shap': importance},
    ).sort_values('mean_absolute_shap', ascending=False).to_csv(
        output_directory / 'feature_importance.csv',
        index=False,
    )

    text_end = text_dimension
    audio_end = text_end + audio_dimension
    modality_importance = {
        'text': float(np.mean(np.abs(all_shap[:, :text_end]).sum(axis=1))),
        'acoustic': float(np.mean(np.abs(all_shap[:, text_end:audio_end]).sum(axis=1))),
        'demographics': float(np.mean(np.abs(all_shap[:, audio_end:]).sum(axis=1))),
    }
    pd.DataFrame(
        modality_importance.items(),
        columns=('modality', 'mean_total_absolute_shap'),
    ).to_csv(output_directory / 'modality_importance.csv', index=False)

    shap.summary_plot(
        all_shap[:, text_end:audio_end],
        all_values[:, text_end:audio_end],
        feature_names=manifest['acoustic_features'],
        max_display=audio_dimension,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_directory / 'acoustic_summary.png', dpi=300)
    plt.savefig(output_directory / 'acoustic_summary.eps')
    plt.close()

    shap.summary_plot(
        all_shap[:, audio_end:],
        all_values[:, audio_end:],
        feature_names=['age', 'sex'],
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_directory / 'demographics_summary.png')
    plt.close()

    figure, axis = plt.subplots()
    axis.bar(modality_importance.keys(), modality_importance.values())
    axis.set_ylabel('Mean total absolute SHAP value')
    figure.tight_layout()
    figure.savefig(output_directory / 'modality_importance.png')
    plt.close(figure)
