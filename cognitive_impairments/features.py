from __future__ import annotations

import dataclasses
import importlib.metadata
import pathlib

import audiofile
import numpy as np
import pandas as pd
from opensmile import FeatureLevel, FeatureSet, Smile
from torch import device as TorchDevice, finfo, inference_mode
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

from cognitive_impairments.chat import Utterance


MILLISECONDS_PER_SECOND = 1000


@dataclasses.dataclass(frozen=True)
class EmbeddingResult:
    values: np.ndarray
    token_count: int
    truncated: bool


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        revision: str | None,
        pooling: str,
        normalize_embedding: bool,
        max_length: int,
        device: TorchDevice,
    ) -> None:
        if pooling not in {'cls', 'mean', 'max'}:
            raise ValueError("pooling must be 'cls', 'mean', or 'max'")
        self.model_name = model_name
        self.revision = revision
        self.pooling = pooling
        self.normalize_embedding = normalize_embedding
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        return self.encode_with_metadata(text).values

    def encode_with_metadata(self, text: str) -> EmbeddingResult:
        untruncated = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )['input_ids']
        token_count = len(untruncated)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with inference_mode():
            hidden_state = self.model(**inputs).last_hidden_state

        attention_mask = inputs['attention_mask'].unsqueeze(-1).bool()
        if self.pooling == 'cls':
            pooled = hidden_state[:, 0]
        elif self.pooling == 'max':
            masked_state = hidden_state.masked_fill(~attention_mask, finfo(hidden_state.dtype).min)
            pooled = masked_state.max(dim=1).values
        else:
            weights = attention_mask.to(hidden_state.dtype)
            pooled = (hidden_state * weights).sum(dim=1) / weights.sum(dim=1)
        if self.normalize_embedding:
            pooled = normalize(pooled, p=2, dim=1)
        return EmbeddingResult(
            values=pooled.squeeze(0).cpu().numpy(),
            token_count=token_count,
            truncated=token_count > self.max_length,
        )


def create_acoustic_extractor() -> Smile:
    return Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.Functionals,
    )


def extract_acoustic_features(
    audio_path: pathlib.Path,
    utterances: tuple[Utterance, ...],
    extractor: Smile,
) -> pd.Series:
    if not utterances:
        raise ValueError(f'No timed participant utterances found for {audio_path}')

    feature_rows = []
    for utterance in utterances:
        start_seconds = utterance.start_ms / MILLISECONDS_PER_SECOND
        end_seconds = utterance.end_ms / MILLISECONDS_PER_SECOND
        features = extractor.process_file(
            str(audio_path),
            start=start_seconds,
            end=end_seconds,
        )
        feature_rows.append(features.squeeze())
    return pd.DataFrame(feature_rows).mean(axis=0)


def extract_concatenated_acoustic_features(
    audio_path: pathlib.Path,
    utterances: tuple[Utterance, ...],
    extractor: Smile,
) -> pd.Series:
    if not utterances:
        raise ValueError(f'No timed participant utterances found for {audio_path}')

    signal, sampling_rate = audiofile.read(str(audio_path), always_2d=True)
    segments = []
    for utterance in sorted(utterances, key=lambda value: (value.start_ms, value.end_ms)):
        start_sample = round(utterance.start_ms * sampling_rate / MILLISECONDS_PER_SECOND)
        end_sample = round(utterance.end_ms * sampling_rate / MILLISECONDS_PER_SECOND)
        start_sample = max(start_sample, 0)
        end_sample = min(end_sample, signal.shape[1])
        if end_sample > start_sample:
            segments.append(signal[:, start_sample:end_sample])
    if not segments:
        raise ValueError(f'No non-empty participant audio segments found for {audio_path}')
    participant_signal = np.concatenate(segments, axis=1)
    return extractor.process_signal(participant_signal, sampling_rate).squeeze()


def extract_whole_audio_features(audio_path: pathlib.Path, extractor: Smile) -> pd.Series:
    return extractor.process_file(str(audio_path)).squeeze()


def package_versions() -> dict[str, str]:
    packages = (
        'audiofile',
        'numpy',
        'opensmile',
        'pandas',
        'pyarrow',
        'scikit-learn',
        'torch',
        'transformers',
    )
    return {package: importlib.metadata.version(package) for package in packages}
