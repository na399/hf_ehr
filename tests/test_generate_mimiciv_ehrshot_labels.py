import datetime
from types import SimpleNamespace

from hf_ehr.eval.generate_mimiciv_ehrshot_labels import iter_labels


def event(code: str, time: datetime.datetime):
    return SimpleNamespace(code=code, time=time)


def test_iter_labels_requires_prior_observation_window():
    admission_start = datetime.datetime(2020, 1, 1, 8, 0)
    admission_end = datetime.datetime(2020, 1, 10, 8, 0)
    prediction_offset = datetime.timedelta(hours=48)
    long_los = datetime.timedelta(days=7)
    min_prior_history = datetime.timedelta(days=730)
    subject_with_history = SimpleNamespace(
        subject_id=123,
        events=[
            event("SNOMED/old", admission_start - datetime.timedelta(days=800)),
            event("Visit//9201//start", admission_start),
            event("Visit//9201//end", admission_end),
        ],
    )
    subject_without_history = SimpleNamespace(
        subject_id=123,
        events=[
            event("SNOMED/recent", admission_start - datetime.timedelta(days=30)),
            event("Visit//9201//start", admission_start),
            event("Visit//9201//end", admission_end),
        ],
    )

    labels = list(iter_labels(subject_with_history, prediction_offset, long_los, min_prior_history))
    no_history_labels = list(iter_labels(subject_without_history, prediction_offset, long_los, min_prior_history))

    assert [row["task"] for row in labels] == ["death", "long_los"]
    assert labels[1]["value"] == "true"
    assert no_history_labels == []
