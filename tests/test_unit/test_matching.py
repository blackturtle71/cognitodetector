from __future__ import annotations

import pandas as pd

from cognitive_impairments.matching import build_age_sex_matched_cohort


def test__build_age_sex_matched_cohort__stable_baseline_exact_pairs() -> None:
    """
    Arrange: есть повторные визиты, смешанная метка и непарная возрастная страта.
    Act: строится сопоставленная когорта по исходным визитам.
    Assert: смешанный участник исключён, а пары совпадают по возрасту и полу.
    """
    frame = pd.DataFrame(
        {
            'participant_id': ['a', 'a', 'b', 'c', 'd', 'd', 'e'],
            'recording_id': ['a-1', 'a-2', 'b-1', 'c-1', 'd-1', 'd-2', 'e-1'],
            'label': [0, 0, 1, 1, 0, 1, 0],
            'age': [60, 61, 60, 70, 80, 81, 70],
            'sex': ['female', 'female', 'female', 'male', 'female', 'female', 'male'],
        },
    )

    matched, summary = build_age_sex_matched_cohort(frame, seed=7)

    assert set(matched['participant_id']) == {'a', 'b', 'c', 'e'}
    assert matched.groupby(['age', 'sex', 'label']).size().to_dict() == {
        (60, 'female', 0): 1,
        (60, 'female', 1): 1,
        (70, 'male', 0): 1,
        (70, 'male', 1): 1,
    }
    assert summary['mixed_label_participants_excluded'] == 1
    assert summary['matched_pairs'] == 2
