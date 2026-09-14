from __future__ import annotations

import importlib.metadata
import pathlib

import numpy as np
import pandas as pd
from opensmile import FeatureLevel, FeatureSet, Smile
from torch import device as TorchDevice, finfo, inference_mode
from transformers import AutoModel, AutoTokenizer

from cognitive_impairments.chat import Utterance


MILLISECONDS_PER_SECOND = 1000


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        pooling: str,
        max_length: int,
        device: TorchDevice,
    ) -> None:
        if pooling not in {'mean', 'max'}:
            raise ValueError("pooling must be either 'mean' or 'max'")
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
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
        if self.pooling == 'max':
            masked_state = hidden_state.masked_fill(~attention_mask, finfo(hidden_state.dtype).min)
            pooled = masked_state.max(dim=1).values
        else:
            weights = attention_mask.to(hidden_state.dtype)
            pooled = (hidden_state * weights).sum(dim=1) / weights.sum(dim=1)
        return pooled.squeeze(0).cpu().numpy()


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


def extract_whole_audio_features(audio_path: pathlib.Path, extractor: Smile) -> pd.Series:
    return extractor.process_file(str(audio_path)).squeeze()


def package_versions() -> dict[str, str]:
    packages = (
        'numpy',
        'opensmile',
        'pandas',
        'pyarrow',
        'scikit-learn',
        'torch',
        'transformers',
    )
    return {package: importlib.metadata.version(package) for package in packages}
