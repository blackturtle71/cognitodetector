from __future__ import annotations

import pathlib

import numpy as np
import pytest
from torch import tensor

from cognitive_impairments.artifacts import ScalerState
from cognitive_impairments.artifacts import create_run_directory, load_neural_fold, save_neural_fold


def test__create_run_directory__existing_directory(tmp_path: pathlib.Path) -> None:
    """
    Arrange: каталог запуска уже существует.
    Act: выполняется попытка создать запуск с тем же именем.
    Assert: существующие результаты не перезаписываются.
    """
    run_directory = tmp_path / 'existing'
    run_directory.mkdir()

    with pytest.raises(FileExistsError):
        create_run_directory(run_directory)


def test__create_run_directory__new_directory(tmp_path: pathlib.Path) -> None:
    """
    Arrange: выбран новый путь для результатов.
    Act: создаётся каталог запуска.
    Assert: каталог создан на диске.
    """
    run_directory = tmp_path / 'new'

    create_run_directory(run_directory)

    assert run_directory.is_dir()


def test__save_neural_fold__safe_round_trip(tmp_path: pathlib.Path) -> None:
    """
    Arrange: подготовлены веса, числовые параметры масштабирования и JSON-метаданные.
    Act: контрольная точка сохраняется и загружается безопасным способом.
    Assert: все части артефакта восстановлены без pickled sklearn-объектов.
    """
    scaler = ScalerState(
        audio_mean=np.array([1.0]),
        audio_scale=np.array([2.0]),
        age_mean=np.array([3.0]),
        age_scale=np.array([4.0]),
    )

    save_neural_fold(
        run_directory=tmp_path,
        experiment_name='model',
        fold=0,
        state_dict={'weight': tensor([5.0])},
        scaler=scaler,
        metadata={'name': 'model'},
    )
    state, loaded_scaler, metadata = load_neural_fold(
        tmp_path / 'checkpoints' / 'model' / 'fold_0',
    )

    assert state['weight'].tolist() == [5.0]
    assert loaded_scaler.audio_mean.tolist() == [1.0]
    assert loaded_scaler.audio_scale.tolist() == [2.0]
    assert loaded_scaler.age_mean.tolist() == [3.0]
    assert loaded_scaler.age_scale.tolist() == [4.0]
    assert metadata == {'name': 'model'}
