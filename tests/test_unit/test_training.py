from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from torch import device, rand

from cognitive_impairments.config import load_experiment_config
from cognitive_impairments.model import EXPERIMENT_MODALITIES, Modalities, MultimodalNet
from cognitive_impairments.training import arrays_from_frame
from cognitive_impairments.training import build_optimizer
from cognitive_impairments.training import create_grouped_splits
from cognitive_impairments.training import fit_scaler


CONFIG_PATH = pathlib.Path('configs/pitt_multimodal.toml')


def test__create_grouped_splits__participant_isolation_and_coverage() -> None:
    """
    Arrange: несколько записей принадлежат одним участникам двух классов.
    Act: строятся стратифицированные групповые разбиения.
    Assert: участники не пересекают границы и каждая запись тестируется один раз.
    """
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'])

    splits = create_grouped_splits(labels, groups, folds=2, seed=7)
    test_indices = np.concatenate([test for _train, test in splits])

    assert sorted(test_indices.tolist()) == list(range(len(labels)))
    for train, test in splits:
        assert set(groups[train]).isdisjoint(groups[test])


def test__fit_scaler__uses_only_selected_rows() -> None:
    """
    Arrange: тестовая строка содержит экстремальные значения вне обучающего индекса.
    Act: масштабирование обучается по выбранным строкам.
    Assert: параметры масштабирования не зависят от тестового значения.
    """
    frame = pd.DataFrame(
        {
            'recording_id': ['a', 'b', 'c'],
            'participant_id': ['a', 'b', 'c'],
            'sex': ['female', 'male', 'female'],
            'age': [10.0, 20.0, 1000.0],
            'label': [0, 1, 1],
            'embedding': [np.array([1.0]), np.array([2.0]), np.array([3.0])],
            'audio': [2.0, 4.0, 1000.0],
        },
    )
    arrays = arrays_from_frame(frame, ['audio'])

    scaler = fit_scaler(arrays, np.array([0, 1]))

    assert scaler.audio_mean.tolist() == [3.0]
    assert scaler.age_mean.tolist() == [15.0]


def test__build_optimizer__contains_classifier_parameters() -> None:
    """
    Arrange: создана полная мультимодальная модель из конфигурации диссертации.
    Act: для модели создаётся оптимизатор.
    Assert: каждый параметр классификатора включён в оптимизацию.
    """
    config = load_experiment_config(CONFIG_PATH)
    model = MultimodalNet(
        text_dimension=4,
        audio_dimension=3,
        config=config.model,
        modalities=Modalities(text=True, audio=True, demographics=True),
    )

    optimizer = build_optimizer(model, config.training)
    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }

    assert all(id(parameter) in optimized_parameters for parameter in model.classifier.parameters())
    assert optimizer.defaults['lr'] == config.training.learning_rate
    assert next(model.parameters()).device == device('cpu')


@pytest.mark.parametrize('modalities', EXPERIMENT_MODALITIES, ids=lambda value: value.name)
def test__multimodal_net__forward_for_every_modality_combination(
    modalities: Modalities,
) -> None:
    """
    Arrange: создана модель с одной из поддерживаемых комбинаций модальностей.
    Act: выполняется прямой проход для полного набора входных тензоров.
    Assert: модель возвращает по две логические оценки на запись.
    """
    config = load_experiment_config(CONFIG_PATH)
    model = MultimodalNet(
        text_dimension=4,
        audio_dimension=3,
        config=config.model,
        modalities=modalities,
    )

    result = model(rand(2, 4), rand(2, 3), rand(2, 1), rand(2, 1))

    assert result.shape == (2, 2)
