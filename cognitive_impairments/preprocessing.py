from __future__ import annotations

import dataclasses
import hashlib
import logging
import pathlib
import typing

import numpy as np
import pandas as pd
from opensmile import Smile
from torch import device as TorchDevice

from cognitive_impairments.chat import ChatRecord, parse_chat_file
from cognitive_impairments.data import IDENTITY_COLUMNS, PREPROCESSING_COLUMNS, write_feature_table
from cognitive_impairments.features import TextEmbedder
from cognitive_impairments.features import create_acoustic_extractor
from cognitive_impairments.features import extract_acoustic_features
from cognitive_impairments.features import extract_concatenated_acoustic_features
from cognitive_impairments.features import package_versions


logger = logging.getLogger(__name__)

PITT_GROUPS = ('Dementia', 'Control')
PITT_POSITIVE_DIAGNOSES = {
    'MCI',
    'Memory',
    'Other',
    'PossibleAD',
    'Probable',
    'ProbableAD',
    'Vascular',
}


@dataclasses.dataclass(frozen=True)
class Exclusion:
    path: str
    reason: str


def cognitive_impairment_label(diagnosis: str) -> int:
    if diagnosis == 'Control':
        return 0
    if diagnosis in PITT_POSITIVE_DIAGNOSES:
        return 1
    raise ValueError(f'Unsupported Pitt diagnosis: {diagnosis}')


def _validate_record(record: ChatRecord, require_diagnosis: bool) -> str | None:
    metadata = record.metadata
    if metadata.media_id is None:
        return 'missing media ID'
    if metadata.age is None:
        return 'missing age'
    if metadata.sex not in {'female', 'male'}:
        return 'missing or unsupported sex'
    if require_diagnosis and metadata.diagnosis is None:
        return 'missing diagnosis'
    if not record.utterances:
        return 'no timed participant utterances'
    return None


def _build_row(
    record: ChatRecord,
    audio_path: pathlib.Path,
    corpus: str,
    split: str,
    diagnosis: str | None,
    label: int | None,
    embedder: TextEmbedder,
    extractor: Smile,
    acoustic_aggregation: str,
) -> dict[str, typing.Any]:
    metadata = record.metadata
    if metadata.media_id is None or metadata.age is None or metadata.sex is None:
        raise ValueError('Validated CHAT record unexpectedly contains incomplete metadata')

    text = ' '.join(utterance.text for utterance in record.utterances)
    if acoustic_aggregation == 'concatenate':
        acoustic_features = extract_concatenated_acoustic_features(
            audio_path,
            record.utterances,
            extractor,
        )
    elif acoustic_aggregation == 'utterance_mean':
        acoustic_features = extract_acoustic_features(audio_path, record.utterances, extractor)
    else:
        raise ValueError(f'Unsupported acoustic aggregation: {acoustic_aggregation}')
    if not np.isfinite(acoustic_features.to_numpy(dtype=np.float64)).all():
        raise ValueError('acoustic extraction produced non-finite values')
    embedding = embedder.encode_with_metadata(text)
    if not np.isfinite(embedding.values).all():
        raise ValueError('text embedding produced non-finite values')
    participant_id = metadata.media_id.split('-', maxsplit=1)[0]
    return {
        'recording_id': metadata.media_id,
        'participant_id': participant_id,
        'corpus': corpus,
        'split': split,
        'age': metadata.age,
        'sex': metadata.sex,
        'diagnosis': diagnosis,
        'label': label,
        'text': text,
        'embedding': embedding.values,
        'embedding_token_count': embedding.token_count,
        'embedding_truncated': embedding.truncated,
        **acoustic_features.to_dict(),
    }


def _prepare_records(
    sources: list[tuple[pathlib.Path, pathlib.Path, str | None, int | None]],
    corpus: str,
    split: str,
    require_diagnosis: bool,
    use_record_diagnosis: bool,
    embedder: TextEmbedder,
    acoustic_aggregation: str,
) -> tuple[pd.DataFrame, list[Exclusion]]:
    extractor = create_acoustic_extractor()
    rows: list[dict[str, typing.Any]] = []
    exclusions: list[Exclusion] = []

    for chat_path, audio_path, forced_diagnosis, forced_label in sources:
        record = parse_chat_file(chat_path)
        reason = _validate_record(record, require_diagnosis=require_diagnosis)
        if reason is not None:
            exclusions.append(Exclusion(path=str(chat_path), reason=reason))
            continue
        if not audio_path.exists():
            exclusions.append(Exclusion(path=str(audio_path), reason='missing audio'))
            continue

        diagnosis = forced_diagnosis
        if diagnosis is None and use_record_diagnosis:
            diagnosis = record.metadata.diagnosis
        label = forced_label
        if label is None and diagnosis is not None:
            label = cognitive_impairment_label(diagnosis)
        try:
            row = _build_row(
                record=record,
                audio_path=audio_path,
                corpus=corpus,
                split=split,
                diagnosis=diagnosis,
                label=label,
                embedder=embedder,
                extractor=extractor,
                acoustic_aggregation=acoustic_aggregation,
            )
        except (RuntimeError, ValueError) as error:
            exclusions.append(Exclusion(path=str(chat_path), reason=str(error)))
            continue
        rows.append(row)

    if not rows:
        raise ValueError('No usable records were produced')
    frame = pd.DataFrame(rows)
    acoustic_columns = sorted(
        set(frame.columns) - set(IDENTITY_COLUMNS) - set(PREPROCESSING_COLUMNS),
    )
    return frame[[*IDENTITY_COLUMNS, *PREPROCESSING_COLUMNS, *acoustic_columns]], exclusions


def _pitt_sources(
    data_root: pathlib.Path,
    task: str,
) -> list[tuple[pathlib.Path, pathlib.Path, None, None]]:
    sources = []
    for group in PITT_GROUPS:
        transcript_dir = data_root / 'Pitt' / group / task
        audio_dir = data_root / 'Pitt_audio' / group / task
        for chat_path in sorted(transcript_dir.glob('*.cha')):
            media_id = chat_path.stem
            sources.append((chat_path, audio_dir / f'{media_id}.mp3', None, None))
    return sources


def prepare_pitt(
    data_root: pathlib.Path,
    output_path: pathlib.Path,
    embedder: TextEmbedder,
    task: str,
    acoustic_aggregation: str = 'utterance_mean',
) -> None:
    sources = _pitt_sources(data_root, task)
    frame, exclusions = _prepare_records(
        sources=sources,
        corpus='pitt',
        split='all',
        require_diagnosis=True,
        use_record_diagnosis=True,
        embedder=embedder,
        acoustic_aggregation=acoustic_aggregation,
    )
    manifest = _build_manifest(
        frame,
        exclusions,
        embedder,
        corpus='pitt',
        split='all',
        acoustic_aggregation=acoustic_aggregation,
        source_digest=_source_digest(sources, data_root),
    )
    manifest['label_definition'] = 'Control=0; every supported non-Control diagnosis=1'
    write_feature_table(frame, output_path, manifest)
    logger.info('Prepared %s Pitt records at %s', len(frame), output_path)


def _adresso_train_sources(
    data_root: pathlib.Path,
) -> list[tuple[pathlib.Path, pathlib.Path, str, int]]:
    base = data_root / 'ADReSS-IS2020-train' / 'ADReSS-IS2020-data' / 'train'
    sources = []
    for directory_name, diagnosis, label in (
        ('cc', 'Control', 0),
        ('cd', 'CognitiveImpairment', 1),
    ):
        for chat_path in sorted((base / 'transcription' / directory_name).glob('*.cha')):
            audio_path = (
                base
                / 'Full_wave_enhanced_audio'
                / directory_name
                / f'{chat_path.stem}.wav'
            )
            sources.append((chat_path, audio_path, diagnosis, label))
    return sources


def _adresso_test_sources(
    data_root: pathlib.Path,
) -> list[tuple[pathlib.Path, pathlib.Path, None, None]]:
    base = data_root / 'ADReSS-IS2020-test' / 'ADReSS-IS2020-data' / 'test'
    return [
        (
            chat_path,
            base / 'Full_wave_enhanced_audio' / f'{chat_path.stem}.wav',
            None,
            None,
        )
        for chat_path in sorted((base / 'transcription').glob('*.cha'))
    ]


def prepare_adresso(
    data_root: pathlib.Path,
    output_path: pathlib.Path,
    embedder: TextEmbedder,
    split: str,
    acoustic_aggregation: str = 'utterance_mean',
) -> None:
    if split == 'train':
        sources = _adresso_train_sources(data_root)
        require_diagnosis = False
    elif split == 'test':
        sources = _adresso_test_sources(data_root)
        require_diagnosis = False
    else:
        raise ValueError("ADReSS split must be either 'train' or 'test'")

    frame, exclusions = _prepare_records(
        sources=sources,
        corpus='adresso',
        split=split,
        require_diagnosis=require_diagnosis,
        use_record_diagnosis=split == 'train',
        embedder=embedder,
        acoustic_aggregation=acoustic_aggregation,
    )
    manifest = _build_manifest(
        frame,
        exclusions,
        embedder,
        corpus='adresso',
        split=split,
        acoustic_aggregation=acoustic_aggregation,
        source_digest=_source_digest(sources, data_root),
    )
    manifest['independent_pitt_validation'] = False
    write_feature_table(frame, output_path, manifest)
    logger.info('Prepared %s ADReSS %s records at %s', len(frame), split, output_path)


def _build_manifest(
    frame: pd.DataFrame,
    exclusions: list[Exclusion],
    embedder: TextEmbedder,
    corpus: str,
    split: str,
    acoustic_aggregation: str,
    source_digest: str,
) -> dict[str, typing.Any]:
    acoustic_features = sorted(
        set(frame.columns) - set(IDENTITY_COLUMNS) - set(PREPROCESSING_COLUMNS),
    )
    embedding_dimension = len(np.asarray(frame['embedding'].iloc[0]).reshape(-1))
    return {
        'corpus': corpus,
        'split': split,
        'records': len(frame),
        'participants': int(frame['participant_id'].nunique()),
        'embedder': embedder.model_name,
        'embedder_revision': embedder.revision,
        'pooling': embedder.pooling,
        'embedding_normalized': embedder.normalize_embedding,
        'max_length': embedder.max_length,
        'truncated_records': int(frame['embedding_truncated'].sum()),
        'truncated_recording_ids': sorted(
            frame.loc[frame['embedding_truncated'], 'recording_id'].astype(str),
        ),
        'embedding_dimension': embedding_dimension,
        'acoustic_aggregation': acoustic_aggregation,
        'acoustic_features': acoustic_features,
        'source_sha256': source_digest,
        'exclusions': [dataclasses.asdict(exclusion) for exclusion in exclusions],
        'package_versions': package_versions(),
    }


def _source_digest(
    sources: list[tuple[pathlib.Path, pathlib.Path, str | None, int | None]],
    data_root: pathlib.Path,
) -> str:
    digest = hashlib.sha256()
    source_paths = sorted({path for source in sources for path in source[:2]})
    for source_path in source_paths:
        try:
            relative_path = source_path.relative_to(data_root)
        except ValueError:
            relative_path = source_path
        digest.update(str(relative_path).encode('utf-8'))
        if source_path.exists():
            digest.update(source_path.read_bytes())
        else:
            digest.update(b'MISSING')
    return digest.hexdigest()


def resolve_device(name: str) -> TorchDevice:
    from torch import cuda, device, ones

    if name == 'cpu':
        return device('cpu')
    if not cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu only for deliberate CPU runs')

    cuda_device = device('cuda')
    try:
        ones(1, device=cuda_device).add_(1)
    except RuntimeError as error:
        raise RuntimeError(
            'CUDA failed its startup check; verify the NVIDIA driver and PyTorch build',
        ) from error
    return cuda_device
