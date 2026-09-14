from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import pathlib
import platform
import shutil
import subprocess
import typing

import numpy as np
from torch import Tensor, load, save

from cognitive_impairments.model import Modalities


def create_run_directory(path: pathlib.Path) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing run directory: {path}')
    path.mkdir(parents=True)


def write_json(path: pathlib.Path, value: typing.Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def copy_run_inputs(
    run_directory: pathlib.Path,
    config_path: pathlib.Path,
    dataset_manifest: dict[str, typing.Any],
    seed: int,
) -> None:
    shutil.copyfile(config_path, run_directory / 'config.toml')
    write_json(run_directory / 'dataset_manifest.json', dataset_manifest)
    write_json(
        run_directory / 'environment.json',
        {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'seed': seed,
            'git': _git_state(),
            'packages': _package_versions(),
        },
    )


def _package_versions() -> dict[str, str]:
    packages = (
        'matplotlib',
        'numpy',
        'pandas',
        'pyarrow',
        'scikit-learn',
        'shap',
        'torch',
    )
    return {package: importlib.metadata.version(package) for package in packages}


def _git_state() -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ('git', 'rev-parse', 'HEAD'),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ('git', 'status', '--porcelain'),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {'revision': None, 'dirty': None}
    return {'revision': revision, 'dirty': dirty}


@dataclasses.dataclass(frozen=True)
class ScalerState:
    audio_mean: np.ndarray
    audio_scale: np.ndarray
    age_mean: np.ndarray
    age_scale: np.ndarray


def save_neural_fold(
    run_directory: pathlib.Path,
    experiment_name: str,
    fold: int,
    state_dict: dict[str, Tensor],
    scaler: ScalerState,
    metadata: dict[str, typing.Any],
) -> None:
    directory = run_directory / 'checkpoints' / experiment_name / f'fold_{fold}'
    directory.mkdir(parents=True)
    save(state_dict, directory / 'weights.pt')
    np.savez(
        directory / 'scalers.npz',
        audio_mean=scaler.audio_mean,
        audio_scale=scaler.audio_scale,
        age_mean=scaler.age_mean,
        age_scale=scaler.age_scale,
    )
    write_json(directory / 'metadata.json', metadata)


def load_neural_fold(
    directory: pathlib.Path,
) -> tuple[dict[str, Tensor], ScalerState, dict[str, typing.Any]]:
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    with np.load(directory / 'scalers.npz', allow_pickle=False) as scalers:
        scaler = ScalerState(
            audio_mean=scalers['audio_mean'],
            audio_scale=scalers['audio_scale'],
            age_mean=scalers['age_mean'],
            age_scale=scalers['age_scale'],
        )
    state_dict = load(directory / 'weights.pt', map_location='cpu', weights_only=True)
    return state_dict, scaler, metadata


def modalities_to_dict(modalities: Modalities) -> dict[str, bool]:
    return dataclasses.asdict(modalities)
