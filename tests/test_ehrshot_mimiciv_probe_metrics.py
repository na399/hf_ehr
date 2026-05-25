import pickle

import numpy as np
import pytest

from hf_ehr.eval.ehrshot_mimiciv_probe_metrics import (
    assert_patient_disjoint,
    assign_splits,
    find_feature_path,
    load_feature_dict,
    metric_rows,
)


def test_assign_splits_keeps_repeated_patients_together():
    patient_ids = np.array([1, 1, 2, 2, 3, 4, 4])
    splits = assign_splits(patient_ids)

    assert_patient_disjoint(patient_ids, splits)
    assert splits[0] == splits[1]
    assert splits[2] == splits[3]
    assert splits[5] == splits[6]


def test_assign_splits_uses_meds_subject_split_file(tmp_path):
    pl = pytest.importorskip("polars")
    path = tmp_path / "subject_splits.parquet"
    pl.DataFrame(
        {
            "subject_id": [10, 11, 12],
            "split": ["train", "tuning", "held_out"],
        }
    ).write_parquet(path)

    splits = assign_splits(np.array([12, 10, 11]), str(path))

    assert splits.tolist() == ["test", "train", "val"]


def test_load_feature_dict_reads_ehrshot_feature_payload(tmp_path):
    path = tmp_path / "features.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {
                "data_matrix": np.ones((3, 2), dtype=np.float32),
                "label_values": np.array([True, False, True]),
                "patient_ids": np.array([10, 11, 10]),
            },
            f,
        )

    x, y, patient_ids = load_feature_dict(str(path))

    assert x.shape == (3, 2)
    assert y.dtype == bool
    assert patient_ids.tolist() == [10, 11, 10]


def test_find_feature_path_uses_embedding_strategy(tmp_path):
    path = tmp_path / "model_death_last_chunk:last_embed:last_nonpad_features_1.pkl"
    path.write_bytes(b"x")

    assert find_feature_path(str(tmp_path), "model", "death", "last_nonpad", "last") == str(path)


def test_metric_rows_include_test_bootstrap_intervals():
    rows = metric_rows(
        task="death",
        model_name="model",
        estimator_name="logistic_l2_balanced",
        split="test",
        y_true=np.array([False, False, True, True]),
        y_prob=np.array([0.1, 0.2, 0.8, 0.9]),
        patient_ids=np.array([1, 2, 3, 4]),
        bootstrap_samples=5,
        seed=1,
    )

    auroc = next(row for row in rows if row["metric"] == "auroc")
    assert auroc["value"] == 1.0
    assert "auroc_ci_lower" in auroc
    assert auroc["n_patients"] == 4
