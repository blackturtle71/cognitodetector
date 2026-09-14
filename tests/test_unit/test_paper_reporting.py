from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cognitive_impairments.paper_reporting import _bootstrap_group_metrics
from cognitive_impairments.paper_reporting import _weighted_metrics


def test__bootstrap_group_metrics__matches_repeated_participant_rows() -> None:
    """
    Arrange: участник с двумя записями получает вес два в кластерной выборке.
    Act: метрики считаются векторизованно по весам участников.
    Assert: результат совпадает с обычными взвешенными бинарными метриками.
    """
    frame = pd.DataFrame(
        {
            'participant_id': ['a', 'a', 'b', 'c'],
            'label': [0, 0, 1, 1],
            'probability': [0.1, 0.4, 0.3, 0.9],
        },
    )
    participant_positions = {'a': 0, 'b': 1, 'c': 2}
    participant_counts = np.array([[2, 1, 1]])

    result = _bootstrap_group_metrics(
        frame,
        participant_positions,
        participant_counts,
    )
    expected = _weighted_metrics(frame, weights=np.array([2, 2, 1, 1]))

    for metric, values in result.items():
        assert values[0] == pytest.approx(expected[metric])
