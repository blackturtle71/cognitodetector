from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import typing

import pandas as pd
from torch import device as TorchDevice

from cognitive_impairments.artifacts import copy_run_inputs, write_json
from cognitive_impairments.config import ExperimentConfig
from cognitive_impairments.config import load_experiment_config, require_paper_protocol
from cognitive_impairments.data import manifest_path_for, read_feature_table
from cognitive_impairments.explanation import explain_cross_validation
from cognitive_impairments.matching import build_age_sex_matched_cohort
from cognitive_impairments.paper_reporting import build_report_artifacts
from cognitive_impairments.paper_training import run_paper_seed
from cognitive_impairments.training import arrays_from_frame


PROTOCOL_FILE = 'protocol.json'
COMPLETE_FILE = 'complete.json'


def _file_digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_digest(package_directory: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_directory.glob('*.py')):
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _protocol_identity(
    dataset_path: pathlib.Path,
    config_path: pathlib.Path,
) -> dict[str, str]:
    manifest_path = manifest_path_for(dataset_path)
    return {
        'dataset_sha256': _file_digest(dataset_path),
        'dataset_manifest_sha256': _file_digest(manifest_path),
        'config_sha256': _file_digest(config_path),
        'implementation_sha256': _implementation_digest(
            pathlib.Path(__file__).resolve().parent,
        ),
        'uv_lock_sha256': _file_digest(
            pathlib.Path(__file__).resolve().parent.parent / 'uv.lock',
        ),
    }


def _prepare_output_directory(
    output_directory: pathlib.Path,
    identity: dict[str, str],
    resume: bool,
) -> None:
    protocol_path = output_directory / PROTOCOL_FILE
    if not output_directory.exists():
        if resume:
            raise FileNotFoundError(f'Cannot resume missing output directory: {output_directory}')
        output_directory.mkdir(parents=True)
        write_json(protocol_path, identity)
        return
    if not resume:
        raise FileExistsError(f'Refusing to overwrite paper run: {output_directory}')
    if not protocol_path.exists():
        raise ValueError(f'Existing paper run has no {PROTOCOL_FILE}: {output_directory}')
    existing = json.loads(protocol_path.read_text(encoding='utf-8'))
    if existing != identity:
        raise ValueError('Cannot resume: the dataset or configuration hash has changed')


def _validate_feature_contract(
    manifest: dict[str, typing.Any],
    config: ExperimentConfig,
) -> None:
    expected = {
        'embedder': config.features.embedder,
        'embedder_revision': config.features.revision,
        'pooling': config.features.pooling,
        'embedding_normalized': config.features.normalize,
        'max_length': config.features.max_length,
        'acoustic_aggregation': config.features.acoustic_aggregation,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise ValueError(
            'Paper feature table is incompatible with its config: ' + ', '.join(mismatched),
        )


def _run_unit(
    cohort: str,
    frame: pd.DataFrame,
    manifest: dict[str, typing.Any],
    config: ExperimentConfig,
    config_path: pathlib.Path,
    output_directory: pathlib.Path,
    identity: dict[str, str],
    seed: int,
    device: TorchDevice,
) -> pathlib.Path:
    unit_directory = output_directory / cohort / f'seed_{seed}'
    complete_path = unit_directory / COMPLETE_FILE
    if unit_directory.exists():
        if not complete_path.exists():
            raise ValueError(f'Refusing ambiguous partial unit: {unit_directory}')
        complete = json.loads(complete_path.read_text(encoding='utf-8'))
        if complete != identity:
            raise ValueError(f'Completed unit has incompatible hashes: {unit_directory}')
        return unit_directory

    unit_directory.mkdir(parents=True)
    copy_run_inputs(unit_directory, config_path, manifest, seed)
    arrays = arrays_from_frame(frame, manifest['acoustic_features'])
    result = run_paper_seed(arrays, config, manifest, seed, unit_directory, device)
    result.predictions.assign(seed=seed, cohort=cohort).to_csv(
        unit_directory / 'oof_predictions.csv',
        index=False,
    )
    result.metrics.assign(seed=seed, cohort=cohort).to_csv(
        unit_directory / 'pooled_metrics.csv',
        index=False,
    )
    result.selections.to_csv(unit_directory / 'model_selection.csv', index=False)
    result.history.to_csv(unit_directory / 'training_history.csv', index=False)
    result.splits.to_csv(unit_directory / 'splits.csv', index=False)
    write_json(complete_path, identity)
    return unit_directory


def _read_predictions(
    output_directory: pathlib.Path,
    cohort: str,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    return pd.concat(
        (
            pd.read_csv(
                output_directory / cohort / f'seed_{seed}' / 'oof_predictions.csv',
                dtype={'recording_id': str, 'participant_id': str},
            )
            for seed in seeds
        ),
        ignore_index=True,
    )


def run_paper_protocol(
    dataset_path: pathlib.Path,
    config_path: pathlib.Path,
    output_directory: pathlib.Path,
    report_directory: pathlib.Path,
    device: TorchDevice,
    resume: bool,
) -> None:
    frame, manifest = read_feature_table(dataset_path)
    if manifest.get('corpus') != 'pitt':
        raise ValueError('The paper protocol requires a Pitt feature table')
    config = load_experiment_config(config_path)
    _selection, paper = require_paper_protocol(config)
    _validate_feature_contract(manifest, config)
    identity = _protocol_identity(dataset_path, config_path)
    _prepare_output_directory(output_directory, identity, resume)

    matched_frame, matching_summary = build_age_sex_matched_cohort(
        frame,
        paper.matching_seed,
    )
    matching_summary_path = output_directory / 'matching_summary.json'
    if matching_summary_path.exists():
        existing_summary = json.loads(matching_summary_path.read_text(encoding='utf-8'))
        if existing_summary != matching_summary:
            raise ValueError('Existing matching summary is incompatible with this dataset')
    else:
        write_json(matching_summary_path, matching_summary)

    for cohort, cohort_frame in (('full', frame), ('matched', matched_frame)):
        for seed in paper.seeds:
            _run_unit(
                cohort,
                cohort_frame,
                manifest,
                config,
                config_path,
                output_directory,
                identity,
                seed,
                device,
            )

    shap_run = output_directory / 'full' / f'seed_{paper.shap_seed}'
    explanation_directory = shap_run / 'explanations'
    if not explanation_directory.exists():
        explain_cross_validation(shap_run, dataset_path, device)

    full_predictions = _read_predictions(output_directory, 'full', paper.seeds)
    matched_predictions = _read_predictions(output_directory, 'matched', paper.seeds)
    report_protocol_path = report_directory / PROTOCOL_FILE
    if report_directory.exists():
        if not report_protocol_path.exists():
            raise ValueError(f'Refusing ambiguous partial report: {report_directory}')
        report_identity = json.loads(report_protocol_path.read_text(encoding='utf-8'))
        if report_identity != identity:
            raise ValueError('Existing report was built from different inputs')
        return
    build_report_artifacts(
        full_frame=frame,
        matched_frame=matched_frame,
        full_predictions=full_predictions,
        matched_predictions=matched_predictions,
        report_directory=report_directory,
        replicates=paper.bootstrap_replicates,
        confidence_level=paper.confidence_level,
        bootstrap_seed=paper.matching_seed,
    )
    for extension in ('png', 'eps'):
        shutil.copyfile(
            explanation_directory / f'acoustic_summary.{extension}',
            report_directory / f'figure_4_acoustic_shap.{extension}',
        )
    write_json(report_protocol_path, identity)
