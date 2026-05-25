import argparse
import collections

import pytest

from hf_ehr.eval.mimiciv_tokenization_smoke import (
    SubjectTokenizationStats,
    summarize_subject_stats,
    validate_summary,
)


def make_args(**overrides):
    values = {
        "min_subjects_with_tokens": 1,
        "min_subject_token_fraction": 0.25,
        "min_total_clinical_tokens": 1,
        "min_unique_clinical_tokens": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_tokenization_smoke_summary_passes_when_clinical_tokens_exist():
    summary = summarize_subject_stats(
        [
            SubjectTokenizationStats(patient_id=1, raw_events=10, clinical_tokens=4, tokenized_nonpad_tokens=7),
            SubjectTokenizationStats(patient_id=2, raw_events=8, clinical_tokens=0, tokenized_nonpad_tokens=3),
        ],
        collections.Counter({"LOINC//1": 2}),
        collections.Counter({"LOINC/1": 4}),
        [7],
    )

    assert summary["subjects_with_clinical_tokens"] == 1
    assert summary["subject_token_fraction"] == 0.5
    assert summary["total_clinical_tokens"] == 4
    validate_summary(summary, make_args())


def test_tokenization_smoke_summary_fails_all_pad_case():
    summary = summarize_subject_stats(
        [SubjectTokenizationStats(patient_id=1, raw_events=10, clinical_tokens=0, tokenized_nonpad_tokens=1)],
        collections.Counter({"LOINC//1": 2}),
        collections.Counter(),
        [],
    )

    with pytest.raises(RuntimeError, match="tokenization smoke failed"):
        validate_summary(summary, make_args())
