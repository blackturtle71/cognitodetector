from __future__ import annotations

import dataclasses

from torch import Tensor, cat, nn

from cognitive_impairments.config import ModelConfig


@dataclasses.dataclass(frozen=True)
class Modalities:
    text: bool
    audio: bool
    demographics: bool

    @property
    def name(self) -> str:
        names = [
            name
            for name, enabled in (
                ('text', self.text),
                ('acoustic', self.audio),
                ('demographics', self.demographics),
            )
            if enabled
        ]
        return '_'.join(names)


EXPERIMENT_MODALITIES = (
    Modalities(text=False, audio=False, demographics=True),
    Modalities(text=False, audio=True, demographics=False),
    Modalities(text=True, audio=False, demographics=False),
    Modalities(text=True, audio=True, demographics=False),
    Modalities(text=True, audio=True, demographics=True),
)


class MultimodalNet(nn.Module):
    def __init__(
        self,
        text_dimension: int,
        audio_dimension: int,
        config: ModelConfig,
        modalities: Modalities,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.text_dimension = text_dimension
        self.audio_dimension = audio_dimension

        text_first, text_second, text_output = config.text_hidden
        audio_first, audio_output = config.audio_hidden
        demographics_first, demographics_second, demographics_output = (
            config.demographics_hidden
        )

        self.text_branch = nn.Sequential(
            nn.Linear(text_dimension, text_first),
            nn.LayerNorm(text_first),
            nn.GELU(),
            nn.Dropout(config.text_dropout[0]),
            nn.Linear(text_first, text_second),
            nn.LayerNorm(text_second),
            nn.GELU(),
            nn.Dropout(config.text_dropout[1]),
            nn.Linear(text_second, text_output),
        )
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dimension, audio_first),
            nn.LayerNorm(audio_first),
            nn.GELU(),
            nn.Dropout(config.audio_dropout[0]),
            nn.Linear(audio_first, audio_output),
            nn.LayerNorm(audio_output),
            nn.GELU(),
            nn.Dropout(config.audio_dropout[1]),
        )
        self.demographics_branch = nn.Sequential(
            nn.Linear(2, demographics_first),
            nn.GELU(),
            nn.Dropout(config.demographics_dropout),
            nn.Linear(demographics_first, demographics_second),
            nn.GELU(),
            nn.Linear(demographics_second, demographics_output),
        )

        classifier_input = 0
        if modalities.text:
            classifier_input += text_output
        if modalities.audio:
            classifier_input += audio_output
        if modalities.demographics:
            classifier_input += demographics_output
        if classifier_input == 0:
            raise ValueError('At least one modality must be enabled')

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, config.classifier_hidden),
            nn.LayerNorm(config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden, 2),
        )

    def forward(
        self,
        embeddings: Tensor,
        audio_features: Tensor,
        age: Tensor,
        sex: Tensor,
    ) -> Tensor:
        branch_outputs: list[Tensor] = []
        if self.modalities.text:
            branch_outputs.append(self.text_branch(embeddings))
        if self.modalities.audio:
            branch_outputs.append(self.audio_branch(audio_features))
        if self.modalities.demographics:
            demographics = cat((age, sex), dim=1)
            branch_outputs.append(self.demographics_branch(demographics))
        combined = cat(branch_outputs, dim=1)
        return self.classifier(combined)
