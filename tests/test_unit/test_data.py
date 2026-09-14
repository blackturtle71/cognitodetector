from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from cognitive_impairments.data import read_feature_table, write_feature_table


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'recording_id': ['001-0'],
            'participant_id': ['001'],
            'corpus': ['pitt'],
            'split': ['all'],
            'age': [70],
            'sex': ['female'],
            'diagnosis': ['Control'],
            'label': [0],
            'text': ['sample'],
            'embedding': [np.array([1.0, 2.0])],
            'acoustic': [3.0],
        },
    )


def _manifest() -> dict[str, object]:
    return {
        'corpus': 'pitt',
        'embedding_dimension': 2,
        'acoustic_features': ['acoustic'],
    }


def test__write_feature_table__round_trip(tmp_path: pathlib.Path) -> None:
    """
    Arrange: подготовлены каноническая таблица и согласованный манифест.
    Act: артефакты записываются и читаются обратно.
    Assert: схема, эмбеддинг и манифест сохраняются без pickle.
    """
    output_path = tmp_path / 'features.parquet'

    write_feature_table(_feature_frame(), output_path, _manifest())
    frame, manifest = read_feature_table(output_path)

    assert frame['recording_id'].tolist() == ['001-0']
    assert np.array_equal(frame['embedding'].iloc[0], np.array([1.0, 2.0]))
    assert manifest['acoustic_features'] == ['acoustic']


def test__write_feature_table__existing_output() -> None:
    """
    Arrange: целевой файл набора данных уже существует.
    Act: выполняется попытка записать артефакты повторно.
    Assert: перезапись отклоняется явным исключением.
    """
    output_path = pathlib.Path(__file__)

    with pytest.raises(FileExistsError):
        write_feature_table(_feature_frame(), output_path, _manifest())
