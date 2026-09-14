from __future__ import annotations

import pathlib

from cognitive_impairments.config import load_experiment_config, require_paper_protocol
from cognitive_impairments.paper_training import build_neural_candidates, compact_model_config


CONFIG_PATH = pathlib.Path('configs/pitt_paper.toml')


def test__paper_config__prespecified_protocol_is_loaded() -> None:
    """
    Arrange: задана конфигурация протокола статьи.
    Act: конфигурация читается и проверяется.
    Assert: закреплены пять повторов и восемь нейросетевых кандидатов.
    """
    config = load_experiment_config(CONFIG_PATH)
    selection, paper = require_paper_protocol(config)

    candidates = build_neural_candidates(config.model, selection)

    assert paper.seeds == (7, 8, 9, 10, 11)
    assert paper.bootstrap_replicates == 10000
    assert len(candidates) == 8
    assert len({candidate.candidate_id for candidate in candidates}) == 8


def test__compact_model_config__halves_widths_and_preserves_dropout() -> None:
    """
    Arrange: задана текущая архитектура нейросети.
    Act: строится компактный вариант.
    Assert: ширины делятся пополам, а прежние dropout остаются неизменными.
    """
    model = load_experiment_config(CONFIG_PATH).model

    compact = compact_model_config(model)

    assert compact.text_hidden == tuple(width // 2 for width in model.text_hidden)
    assert compact.audio_hidden == tuple(width // 2 for width in model.audio_hidden)
    assert compact.demographics_hidden == tuple(
        width // 2 for width in model.demographics_hidden
    )
    assert compact.classifier_hidden == model.classifier_hidden // 2
    assert compact.text_dropout == model.text_dropout
    assert compact.audio_dropout == model.audio_dropout
