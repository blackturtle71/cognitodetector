from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from cognitive_impairments.training import DEFAULT_PROBABILITY_THRESHOLD


REPORTED_METRICS = ('roc_auc', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1')
CONFUSION_METRICS = ('true_negative', 'false_positive', 'false_negative', 'true_positive')
ALL_METRICS = (*REPORTED_METRICS, *CONFUSION_METRICS)
FULL_EXPERIMENT = 'text_acoustic_demographics'


def _weighted_metrics(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    labels = frame['label'].to_numpy()
    probabilities = frame['probability'].to_numpy()
    predictions = (probabilities >= DEFAULT_PROBABILITY_THRESHOLD).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1], sample_weight=weights)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / (true_negative + false_positive)
    return {
        'roc_auc': float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        'accuracy': float(accuracy_score(labels, predictions, sample_weight=weights)),
        'precision': float(
            precision_score(labels, predictions, sample_weight=weights, zero_division=0),
        ),
        'sensitivity': float(
            recall_score(labels, predictions, sample_weight=weights, zero_division=0),
        ),
        'specificity': float(specificity),
        'f1': float(f1_score(labels, predictions, sample_weight=weights, zero_division=0)),
        'true_negative': float(true_negative),
        'false_positive': float(false_positive),
        'false_negative': float(false_negative),
        'true_positive': float(true_positive),
    }


def seed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, seed), group in predictions.groupby(['experiment', 'seed'], sort=True):
        rows.append({'experiment': experiment, 'seed': seed, **_weighted_metrics(group)})
    return pd.DataFrame(rows)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _auc_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def _bootstrap_group_metrics(
    frame: pd.DataFrame,
    participant_positions: dict[str, int],
    participant_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    row_positions = frame['participant_id'].astype(str).map(participant_positions).to_numpy()
    weights = participant_counts[:, row_positions]
    labels = frame['label'].to_numpy(dtype=bool)
    predictions = (
        frame['probability'].to_numpy() >= DEFAULT_PROBABILITY_THRESHOLD
    )

    true_positive = weights[:, labels & predictions].sum(axis=1)
    true_negative = weights[:, ~labels & ~predictions].sum(axis=1)
    false_positive = weights[:, ~labels & predictions].sum(axis=1)
    false_negative = weights[:, labels & ~predictions].sum(axis=1)
    total = true_positive + true_negative + false_positive + false_negative

    order = np.argsort(frame['probability'].to_numpy(), kind='stable')
    sorted_probabilities = frame['probability'].to_numpy()[order]
    sorted_labels = labels[order]
    sorted_weights = weights[:, order]
    _unique_probabilities, group_starts = np.unique(
        sorted_probabilities,
        return_index=True,
    )
    positive_by_probability = np.add.reduceat(
        sorted_weights * sorted_labels,
        group_starts,
        axis=1,
    )
    negative_by_probability = np.add.reduceat(
        sorted_weights * ~sorted_labels,
        group_starts,
        axis=1,
    )
    negative_before = np.cumsum(negative_by_probability, axis=1) - negative_by_probability
    auc_numerator = (
        positive_by_probability * (negative_before + negative_by_probability / 2)
    ).sum(axis=1)
    auc_denominator = positive_by_probability.sum(axis=1) * negative_by_probability.sum(axis=1)

    return {
        'roc_auc': _auc_divide(auc_numerator, auc_denominator),
        'accuracy': _safe_divide(true_positive + true_negative, total),
        'precision': _safe_divide(true_positive, true_positive + false_positive),
        'sensitivity': _safe_divide(true_positive, true_positive + false_negative),
        'specificity': _safe_divide(true_negative, true_negative + false_positive),
        'f1': _safe_divide(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_positive': true_positive,
    }


def clustered_bootstrap(
    predictions: pd.DataFrame,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = np.sort(predictions['participant_id'].astype(str).unique())
    generator = np.random.default_rng(seed)
    experiments = sorted(predictions['experiment'].unique())
    seeds = sorted(predictions['seed'].unique())
    participant_positions = {
        participant: position for position, participant in enumerate(participants)
    }
    sampling_probabilities = np.full(len(participants), 1 / len(participants))
    participant_counts = generator.multinomial(
        len(participants),
        sampling_probabilities,
        size=replicates,
    )
    distributions: dict[tuple[str, str], np.ndarray] = {}

    indexed = {
        (experiment, run_seed): group.reset_index(drop=True)
        for (experiment, run_seed), group in predictions.groupby(['experiment', 'seed'])
    }
    for experiment in experiments:
        run_metrics = [
            _bootstrap_group_metrics(
                indexed[(experiment, run_seed)],
                participant_positions,
                participant_counts,
            )
            for run_seed in seeds
        ]
        for metric in ALL_METRICS:
            distributions[(experiment, metric)] = np.mean(
                np.stack([values[metric] for values in run_metrics]),
                axis=0,
            )

    alpha = (1 - confidence_level) / 2
    seed_level = seed_metrics(predictions)
    summary_rows = []
    for experiment in experiments:
        experiment_metrics = seed_level[seed_level['experiment'] == experiment]
        for metric in ALL_METRICS:
            distribution = distributions[(experiment, metric)]
            summary_rows.append(
                {
                    'experiment': experiment,
                    'metric': metric,
                    'mean': float(experiment_metrics[metric].mean()),
                    'seed_standard_deviation': float(experiment_metrics[metric].std(ddof=1)),
                    'ci_lower': float(np.nanquantile(distribution, alpha)),
                    'ci_upper': float(np.nanquantile(distribution, 1 - alpha)),
                },
            )

    delta_rows = []
    for experiment in experiments:
        if experiment == FULL_EXPERIMENT:
            continue
        for metric in ALL_METRICS:
            delta = distributions[(experiment, metric)] - distributions[
                (FULL_EXPERIMENT, metric)
            ]
            point = (
                seed_level[seed_level['experiment'] == experiment][metric].mean()
                - seed_level[seed_level['experiment'] == FULL_EXPERIMENT][metric].mean()
            )
            delta_rows.append(
                {
                    'experiment': experiment,
                    'reference': FULL_EXPERIMENT,
                    'metric': metric,
                    'mean_delta': float(point),
                    'ci_lower': float(np.nanquantile(delta, alpha)),
                    'ci_upper': float(np.nanquantile(delta, 1 - alpha)),
                },
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def cohort_table(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    baseline = (
        frame.sort_values(['participant_id', 'age', 'recording_id'])
        .groupby('participant_id', as_index=False, sort=True)
        .first()
    )
    rows = []
    for label_name, label in (('Control', 0), ('Cognitive impairment', 1), ('Overall', None)):
        recording_group = frame if label is None else frame[frame['label'] == label]
        participant_group = baseline if label is None else baseline[baseline['label'] == label]
        rows.append(
            {
                'cohort': cohort,
                'group': label_name,
                'recordings': len(recording_group),
                'participants': len(participant_group),
                'baseline_age_mean': float(participant_group['age'].mean()),
                'baseline_age_standard_deviation': float(
                    participant_group['age'].std(ddof=1),
                ),
                'female_participants': int((participant_group['sex'] == 'female').sum()),
                'male_participants': int((participant_group['sex'] == 'male').sum()),
            },
        )
    return pd.DataFrame(rows)


def _save_figure(figure: plt.Figure, directory: pathlib.Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(directory / f'{stem}.png', dpi=300)
    figure.savefig(directory / f'{stem}.eps')
    plt.close(figure)


def _pipeline_figure(directory: pathlib.Path) -> None:
    steps = (
        'Pitt recordings\nand CHAT transcripts',
        'Participant speech\n+ BGE embeddings',
        'Repeated participant-\ngrouped nested CV',
        'OOF evaluation,\nbootstrap, sensitivity',
    )
    figure, axis = plt.subplots(figsize=(10, 2.5))
    axis.set_axis_off()
    positions = np.arange(len(steps))
    for position, step in zip(positions, steps, strict=True):
        axis.text(
            position,
            0,
            step,
            ha='center',
            va='center',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'black'},
        )
        if position:
            axis.annotate(
                '',
                xy=(position - 0.35, 0),
                xytext=(position - 0.65, 0),
                arrowprops={'arrowstyle': '->'},
            )
    axis.set_xlim(-0.6, len(steps) - 0.4)
    axis.set_ylim(-1, 1)
    _save_figure(figure, directory, 'figure_1_pipeline')


def _roc_figure(predictions: pd.DataFrame, directory: pathlib.Path) -> None:
    figure, axis = plt.subplots()
    for experiment, group in predictions.groupby('experiment', sort=True):
        false_positive_rate, true_positive_rate, _thresholds = roc_curve(
            group['label'],
            group['probability'],
        )
        axis.plot(false_positive_rate, true_positive_rate, label=experiment)
    axis.plot([0, 1], [0, 1], linestyle='--', color='grey')
    axis.set_xlabel('False positive rate')
    axis.set_ylabel('True positive rate')
    axis.legend(fontsize='small')
    _save_figure(figure, directory, 'figure_2_oof_roc')


def _comparison_figure(summary: pd.DataFrame, directory: pathlib.Path) -> None:
    plotted = summary[summary['metric'] == 'roc_auc'].sort_values('mean').reset_index(drop=True)
    figure, axis = plt.subplots()
    positions = np.arange(len(plotted))
    axis.hlines(positions, plotted['ci_lower'], plotted['ci_upper'])
    axis.scatter(plotted['mean'], positions)
    axis.set_yticks(positions, plotted['experiment'])
    axis.set_xlabel('ROC AUC (95% CI)')
    _save_figure(figure, directory, 'figure_3_model_comparison')


def _sensitivity_figure(table: pd.DataFrame, directory: pathlib.Path) -> None:
    plotted = table[
        (table['experiment'] == FULL_EXPERIMENT) & (table['metric'] == 'roc_auc')
    ].reset_index(drop=True)
    figure, axis = plt.subplots()
    positions = np.arange(len(plotted))
    axis.hlines(positions, plotted['ci_lower'], plotted['ci_upper'])
    axis.scatter(plotted['mean'], positions)
    axis.set_yticks(positions, plotted['cohort'])
    axis.set_xlabel('ROC AUC (95% CI)')
    _save_figure(figure, directory, 'figure_5_full_vs_matched')


def build_report_artifacts(
    full_frame: pd.DataFrame,
    matched_frame: pd.DataFrame,
    full_predictions: pd.DataFrame,
    matched_predictions: pd.DataFrame,
    report_directory: pathlib.Path,
    replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> None:
    if report_directory.exists():
        raise FileExistsError(f'Refusing to overwrite report directory: {report_directory}')
    report_directory.mkdir(parents=True)

    table_1 = pd.concat(
        (cohort_table(full_frame, 'full'), cohort_table(matched_frame, 'matched')),
        ignore_index=True,
    )
    table_1.to_csv(report_directory / 'table_1_cohort.csv', index=False)

    full_summary, full_deltas = clustered_bootstrap(
        full_predictions,
        replicates,
        confidence_level,
        bootstrap_seed,
    )
    full_summary.to_csv(report_directory / 'table_2_performance.csv', index=False)
    full_deltas.to_csv(report_directory / 'table_3_paired_deltas.csv', index=False)

    matched_summary, _matched_deltas = clustered_bootstrap(
        matched_predictions,
        replicates,
        confidence_level,
        bootstrap_seed,
    )
    full_sensitivity = full_predictions[
        full_predictions['recording_id'].astype(str).isin(
            matched_frame['recording_id'].astype(str),
        )
    ]
    restricted_summary, _restricted_deltas = clustered_bootstrap(
        full_sensitivity,
        replicates,
        confidence_level,
        bootstrap_seed,
    )
    table_4 = pd.concat(
        (
            full_summary.assign(cohort='full_protocol'),
            restricted_summary.assign(cohort='full_protocol_matched_records'),
            matched_summary.assign(cohort='matched_protocol_rerun'),
        ),
        ignore_index=True,
    )
    table_4.to_csv(report_directory / 'table_4_age_sensitivity.csv', index=False)

    _pipeline_figure(report_directory)
    _roc_figure(full_predictions, report_directory)
    _comparison_figure(full_summary, report_directory)
    _sensitivity_figure(table_4, report_directory)
    (report_directory / 'README.md').write_text(
        '# Paper artifacts\n\n'
        'All tables contain aggregate results only. PNG files are previews; EPS files are the '
        'publication versions. Figure 1 shows the evaluation pipeline. Figure 2 shows pooled '
        'out-of-fold ROC curves. Figure 3 compares models and ablations. Figure 4 is the acoustic '
        'SHAP summary copied from the prespecified seed-7 run. Figure 5 compares the complete and '
        'age/sex-matched analyses.\n',
        encoding='utf-8',
    )
