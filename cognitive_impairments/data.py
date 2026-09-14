from __future__ import annotations

import json
import pathlib
import typing

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    'recording_id',
    'participant_id',
    'corpus',
    'split',
    'age',
    'sex',
    'diagnosis',
    'label',
    'text',
    'embedding',
)


def manifest_path_for(dataset_path: pathlib.Path) -> pathlib.Path:
    return dataset_path.with_suffix('.manifest.json')


def write_feature_table(
    frame: pd.DataFrame,
    output_path: pathlib.Path,
    manifest: dict[str, typing.Any],
) -> None:
    manifest_path = manifest_path_for(output_path)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite dataset artifacts for {output_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def load_manifest(dataset_path: pathlib.Path) -> dict[str, typing.Any]:
    path = manifest_path_for(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset manifest not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_feature_table(frame: pd.DataFrame, manifest: dict[str, typing.Any]) -> None:
    missing_columns = set(IDENTITY_COLUMNS) - set(frame.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(f'Feature table is missing columns: {missing}')

    acoustic_features = manifest.get('acoustic_features')
    if not isinstance(acoustic_features, list) or not acoustic_features:
        raise ValueError('Manifest must contain a non-empty acoustic_features list')
    missing_acoustic = set(acoustic_features) - set(frame.columns)
    if missing_acoustic:
        missing = ', '.join(sorted(missing_acoustic))
        raise ValueError(f'Feature table is missing acoustic features: {missing}')

    embedding_dimension = manifest.get('embedding_dimension')
    if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
        raise ValueError('Manifest embedding_dimension must be a positive integer')
    invalid_dimensions = [
        len(np.asarray(embedding).reshape(-1)) != embedding_dimension
        for embedding in frame['embedding']
    ]
    if any(invalid_dimensions):
        raise ValueError('An embedding does not match manifest embedding_dimension')


def read_feature_table(
    dataset_path: pathlib.Path,
) -> tuple[pd.DataFrame, dict[str, typing.Any]]:
    frame = pd.read_parquet(dataset_path)
    manifest = load_manifest(dataset_path)
    validate_feature_table(frame, manifest)
    return frame, manifest
