from __future__ import annotations

import numpy as np
import pandas as pd


def build_age_sex_matched_cohort(
    frame: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    label_counts = frame.groupby('participant_id')['label'].nunique()
    stable_participants = label_counts[label_counts == 1].index
    stable = frame[frame['participant_id'].isin(stable_participants)].copy()
    baseline = (
        stable.sort_values(['participant_id', 'age', 'recording_id'])
        .groupby('participant_id', sort=True, as_index=False)
        .first()
    )

    generator = np.random.default_rng(seed)
    matched_rows = []
    strata = baseline.groupby(['age', 'sex'], sort=True, dropna=False)
    for _stratum, stratum in strata:
        controls = stratum[stratum['label'] == 0].index.to_numpy(copy=True)
        impaired = stratum[stratum['label'] == 1].index.to_numpy(copy=True)
        generator.shuffle(controls)
        generator.shuffle(impaired)
        pair_count = min(len(controls), len(impaired))
        matched_rows.extend(controls[:pair_count])
        matched_rows.extend(impaired[:pair_count])

    matched = baseline.loc[matched_rows].sort_values('recording_id').reset_index(drop=True)
    summary = {
        'source_participants': int(frame['participant_id'].nunique()),
        'mixed_label_participants_excluded': int((label_counts > 1).sum()),
        'stable_baseline_participants': len(baseline),
        'matched_pairs': len(matched) // 2,
        'matched_participants': len(matched),
    }
    return matched, summary
