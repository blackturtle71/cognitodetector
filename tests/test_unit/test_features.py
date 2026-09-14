from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from cognitive_impairments.chat import Utterance
from cognitive_impairments.features import extract_concatenated_acoustic_features


class RecordingExtractor:
    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self.sampling_rate: int | None = None

    def process_signal(self, signal: np.ndarray, sampling_rate: int) -> pd.DataFrame:
        self.signal = signal
        self.sampling_rate = sampling_rate
        return pd.DataFrame({'feature': [signal.sum()], 'samples': [signal.shape[1]]})


def test__extract_concatenated_acoustic_features__keeps_chronological_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Arrange: две реплики участника переданы не в хронологическом порядке.
    Act: аудиофрагменты объединяются и один раз передаются openSMILE.
    Assert: извлекатель получает единый сигнал в хронологическом порядке.
    """
    signal = np.arange(10, dtype=np.float32).reshape(1, -1)
    monkeypatch.setattr(
        'cognitive_impairments.features.audiofile.read',
        lambda *_args, **_kwargs: (signal, 10),
    )
    utterances = (
        Utterance(text='second', start_ms=500, end_ms=700),
        Utterance(text='first', start_ms=100, end_ms=300),
    )
    extractor = RecordingExtractor()

    result = extract_concatenated_acoustic_features(
        pathlib.Path('recording.wav'),
        utterances,
        extractor,
    )

    assert extractor.signal is not None
    assert extractor.signal.tolist() == [[1.0, 2.0, 5.0, 6.0]]
    assert extractor.sampling_rate == 10
    assert result['feature'] == 14.0
