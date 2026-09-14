from __future__ import annotations

import json
import pathlib
import typing

import pandas as pd

from cognitive_impairments.data import read_feature_table


def _normalized_counts(values: pd.Series) -> dict[str, int]:
    return {
        str(value) if pd.notna(value) else 'missing': int(count)
        for value, count in values.value_counts(dropna=False).items()
    }


def summarize_dataset(frame: pd.DataFrame) -> dict[str, typing.Any]:
    summary: dict[str, typing.Any] = {
        'records': len(frame),
        'participants': int(frame['participant_id'].nunique()),
        'diagnoses': _normalized_counts(frame['diagnosis']),
        'sex': _normalized_counts(frame['sex']),
    }
    if not frame['label'].isna().all():
        summary['labels'] = {
            str(int(label)): int(count)
            for label, count in frame['label'].dropna().value_counts().items()
        }
    age_by_label = {}
    for label, group in frame.dropna(subset=['label']).groupby('label'):
        age_by_label[str(int(label))] = {
            'minimum': float(group['age'].min()),
            'mean': float(group['age'].mean()),
            'maximum': float(group['age'].max()),
        }
    summary['age_by_label'] = age_by_label
    return summary


def save_dataset_statistics(dataset_path: pathlib.Path, output_path: pathlib.Path) -> None:
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite statistics output: {output_path}')
    frame, _manifest = read_feature_table(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summarize_dataset(frame), indent=2, sort_keys=True),
        encoding='utf-8',
    )
