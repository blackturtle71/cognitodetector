from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


@dataclasses.dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def calculate_binary_metrics(
    labels: np.ndarray,
    positive_probabilities: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    predictions = (positive_probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return BinaryMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, positive_probabilities)),
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
    )
