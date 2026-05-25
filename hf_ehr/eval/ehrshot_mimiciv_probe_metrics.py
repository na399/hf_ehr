import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


SPLIT_BUCKETS = {
    "train": range(0, 70),
    "val": range(70, 85),
    "test": range(85, 100),
}
LR_C_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 1e2, 1e4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MIMIC-IV EHRSHOT-style probe metrics from frozen feature pickles.")
    parser.add_argument("--path_to_features_dir", required=True)
    parser.add_argument("--path_to_results_dir", required=True)
    parser.add_argument("--model_name", default="mimiciv_gpt_small_10ep_medsnorm_clmbr_512")
    parser.add_argument("--tasks", nargs="+", default=["long_los", "death"])
    parser.add_argument("--path_to_subject_splits", default=None)
    parser.add_argument("--embed_strat", default="last_nonpad")
    parser.add_argument("--chunk_strat", default="last")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser.parse_args()


def patient_bucket(patient_id: Any) -> int:
    digest = hashlib.blake2b(str(int(patient_id)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def assign_hash_splits(patient_ids: np.ndarray) -> np.ndarray:
    splits = np.empty(patient_ids.shape[0], dtype=object)
    for idx, patient_id in enumerate(patient_ids):
        bucket = patient_bucket(patient_id)
        for split, buckets in SPLIT_BUCKETS.items():
            if bucket in buckets:
                splits[idx] = split
                break
    return splits


def normalize_subject_split(value: Any) -> str:
    try:
        from meds import held_out_split, train_split, tuning_split

        if value == train_split:
            return "train"
        if value == tuning_split:
            return "val"
        if value == held_out_split:
            return "test"
    except Exception:
        pass

    text = str(value).strip().lower().replace("-", "_")
    if text in {"train", "training", "train_split"}:
        return "train"
    if text in {"val", "valid", "validation", "tune", "tuning", "tuning_split"}:
        return "val"
    if text in {"test", "heldout", "held_out", "held_out_split"}:
        return "test"
    raise ValueError(f"Unrecognized subject split value: {value!r}")


def load_subject_split_mapping(path: str) -> Dict[int, str]:
    import polars as pl

    df = pl.read_parquet(path)
    subject_col = "subject_id" if "subject_id" in df.columns else "patient_id"
    if subject_col not in df.columns or "split" not in df.columns:
        raise ValueError(f"Expected subject_id/patient_id and split columns in {path}; found {df.columns}")
    mapping: Dict[int, str] = {}
    for row in df.select([subject_col, "split"]).iter_rows(named=True):
        mapping[int(row[subject_col])] = normalize_subject_split(row["split"])
    return mapping


def assign_splits(patient_ids: np.ndarray, path_to_subject_splits: str | None = None) -> np.ndarray:
    if path_to_subject_splits is None:
        return assign_hash_splits(patient_ids)

    mapping = load_subject_split_mapping(path_to_subject_splits)
    splits = np.empty(patient_ids.shape[0], dtype=object)
    missing = []
    for idx, patient_id in enumerate(patient_ids):
        patient_id = int(patient_id)
        split = mapping.get(patient_id)
        if split is None:
            missing.append(patient_id)
            split = "missing"
        splits[idx] = split
    if missing:
        sample = sorted(set(missing))[:10]
        raise ValueError(f"{len(set(missing))} patient ids were absent from {path_to_subject_splits}; sample={sample}")
    return splits


def assert_patient_disjoint(patient_ids: np.ndarray, splits: np.ndarray) -> None:
    seen: Dict[int, str] = {}
    for patient_id, split in zip(patient_ids, splits):
        patient_id = int(patient_id)
        if patient_id in seen and seen[patient_id] != split:
            raise AssertionError(f"Patient {patient_id} assigned to both {seen[patient_id]} and {split}")
        seen[patient_id] = str(split)


def load_feature_dict(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return (
        np.asarray(payload["data_matrix"], dtype=np.float32),
        np.asarray(payload["label_values"], dtype=bool),
        np.asarray(payload["patient_ids"]),
    )


def safe_metric(fn: Any, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return math.nan


def summarize_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = y_prob >= 0.5
    return {
        "auroc": safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": safe_metric(average_precision_score, y_true, y_prob),
        "brier": safe_metric(brier_score_loss, y_true, y_prob),
        "log_loss": safe_metric(lambda y, p: log_loss(y, p, labels=[False, True]), y_true, y_prob),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray,
    metric_name: str,
    n_samples: int,
    seed: int,
) -> Dict[str, float]:
    if n_samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    per_patient_indices = {patient_id: np.where(patient_ids == patient_id)[0] for patient_id in unique_patients}
    scores: List[float] = []
    for _ in range(n_samples):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sample_indices = np.concatenate([per_patient_indices[patient_id] for patient_id in sampled_patients])
        score = summarize_predictions(y_true[sample_indices], y_prob[sample_indices])[metric_name]
        if not math.isnan(score):
            scores.append(score)
    if not scores:
        return {
            f"{metric_name}_ci_lower": math.nan,
            f"{metric_name}_ci_median": math.nan,
            f"{metric_name}_ci_upper": math.nan,
        }
    return {
        f"{metric_name}_ci_lower": float(np.percentile(scores, 2.5)),
        f"{metric_name}_ci_median": float(np.percentile(scores, 50.0)),
        f"{metric_name}_ci_upper": float(np.percentile(scores, 97.5)),
    }


def metric_rows(
    task: str,
    model_name: str,
    estimator_name: str,
    split: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    scores = summarize_predictions(y_true, y_prob)
    rows = []
    for metric, value in scores.items():
        row: Dict[str, Any] = {
            "task": task,
            "model": model_name,
            "estimator": estimator_name,
            "split": split,
            "metric": metric,
            "value": value,
            "n_labels": int(y_true.shape[0]),
            "n_patients": int(np.unique(patient_ids).shape[0]),
            "n_positive": int(y_true.sum()),
            "prevalence": float(np.mean(y_true)),
        }
        if split == "test" and metric in {"auroc", "auprc", "brier"}:
            row.update(bootstrap_ci(y_true, y_prob, patient_ids, metric, bootstrap_samples, seed))
        rows.append(row)
    return rows


def find_feature_path(path_to_features_dir: str, model_name: str, task: str, embed_strat: str, chunk_strat: str) -> str:
    filename = f"{model_name}_{task}_{chunk_strat}_chunk:last_embed:{embed_strat}_features_1.pkl"
    path = os.path.join(path_to_features_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def fit_ehrshot_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int,
    seed: int,
    n_jobs: int,
    class_weight: str | None = None,
) -> Pipeline:
    scaler = MaxAbsScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    params: Dict[str, Any] = {"C": 1.0, "penalty": "l2"}

    if len(np.unique(y_train)) > 1 and len(np.unique(y_val)) > 1 and x_val.shape[0] > 0:
        x_tune = np.concatenate([x_train_scaled, x_val_scaled], axis=0)
        y_tune = np.concatenate([y_train, y_val], axis=0)
        test_fold = -np.ones(x_tune.shape[0])
        test_fold[x_train_scaled.shape[0]:] = 0
        base = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )
        grid = GridSearchCV(
            base,
            {"C": LR_C_GRID, "penalty": ["l2"]},
            cv=PredefinedSplit(test_fold),
            n_jobs=n_jobs,
            refit=False,
            scoring="roc_auc",
        )
        grid.fit(x_tune, y_tune)
        params = grid.best_params_

    model = LogisticRegression(
        C=float(params["C"]),
        class_weight=class_weight,
        max_iter=max_iter,
        penalty=str(params["penalty"]),
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x_train_scaled, y_train)
    return Pipeline([("scaler", scaler), ("logistic", model)])


def fit_estimators(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int,
    seed: int,
    n_jobs: int,
) -> Dict[str, Any]:
    return {
        "logistic_l2_ehrshot": fit_ehrshot_logistic(x_train, y_train, x_val, y_val, max_iter, seed, n_jobs),
        "logistic_l2_balanced_ehrshot": fit_ehrshot_logistic(
            x_train,
            y_train,
            x_val,
            y_val,
            max_iter,
            seed,
            n_jobs,
            class_weight="balanced",
        ),
        "dummy_prior": DummyClassifier(strategy="prior").fit(x_train, y_train),
    }


def predict_positive(estimator: Any, x: np.ndarray) -> np.ndarray:
    classes = list(estimator.classes_ if hasattr(estimator, "classes_") else estimator[-1].classes_)
    positive_idx = classes.index(True)
    return estimator.predict_proba(x)[:, positive_idx]


def write_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.path_to_results_dir, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for task in args.tasks:
        feature_path = find_feature_path(args.path_to_features_dir, args.model_name, task, args.embed_strat, args.chunk_strat)
        x, y, patient_ids = load_feature_dict(feature_path)
        splits = assign_splits(patient_ids, args.path_to_subject_splits)
        assert_patient_disjoint(patient_ids, splits)

        train_mask = splits == "train"
        val_mask = splits == "val"
        estimators = fit_estimators(
            x[train_mask],
            y[train_mask],
            x[val_mask],
            y[val_mask],
            args.max_iter,
            args.seed,
            args.n_jobs,
        )
        task_rows: List[Dict[str, Any]] = []
        for estimator_name, estimator in estimators.items():
            for split in ["train", "val", "test"]:
                mask = splits == split
                y_prob = predict_positive(estimator, x[mask])
                task_rows.extend(
                    metric_rows(
                        task=task,
                        model_name=args.model_name,
                        estimator_name=estimator_name,
                        split=split,
                        y_true=y[mask],
                        y_prob=y_prob,
                        patient_ids=patient_ids[mask],
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + len(task_rows),
                    )
                )

        task_dir = os.path.join(args.path_to_results_dir, task)
        write_rows(os.path.join(task_dir, "all_results.csv"), task_rows)
        all_rows.extend(task_rows)
        summary[task] = {
            "feature_path": feature_path,
            "n_labels": int(y.shape[0]),
            "n_patients": int(np.unique(patient_ids).shape[0]),
            "n_positive": int(y.sum()),
            "prevalence": float(np.mean(y)),
            "split_source": args.path_to_subject_splits or "deterministic_patient_hash",
            "split_counts": {
                split: {
                    "n_labels": int(np.sum(splits == split)),
                    "n_patients": int(np.unique(patient_ids[splits == split]).shape[0]),
                    "n_positive": int(y[splits == split].sum()),
                    "prevalence": float(np.mean(y[splits == split])),
                }
                for split in ["train", "val", "test"]
            },
        }

    write_rows(os.path.join(args.path_to_results_dir, "all_results.csv"), all_rows)
    with open(os.path.join(args.path_to_results_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    for row in all_rows:
        if row["estimator"].startswith("logistic_l2") and row["split"] == "test":
            print(
                f"{row['task']} {row['estimator']} {row['metric']}={row['value']:.6f} "
                f"(n={row['n_labels']}, positives={row['n_positive']})"
            )


if __name__ == "__main__":
    main()
