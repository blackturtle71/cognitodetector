from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from cognitive_impairments.evaluation import calculate_binary_metrics


def test__calculate_binary_metrics__uses_probabilities_for_auc() -> None:
    """
    Arrange: вероятности содержат больше информации, чем пороговые классы.
    Act: рассчитываются бинарные метрики.
    Assert: ROC AUC совпадает с расчётом по вероятностям и матрица ориентирована верно.
    """
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.4, 0.3, 0.2, 0.9])

    result = calculate_binary_metrics(labels, probabilities, threshold=0.5)

    assert result.roc_auc == roc_auc_score(labels, probabilities)
    assert result.true_negative == 2
    assert result.false_positive == 0
    assert result.false_negative == 1
    assert result.true_positive == 1
    assert result.sensitivity == 0.5
    assert result.specificity == 1.0
