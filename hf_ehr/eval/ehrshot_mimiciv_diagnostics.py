import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from hf_ehr.config import Event
from hf_ehr.eval.ehrshot_mimiciv_probe_metrics import assign_splits
from hf_ehr.utils import load_tokenizer_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MIMIC-IV EHRSHOT-style labels, tokens, and frozen features.")
    parser.add_argument("--path_to_database", required=True)
    parser.add_argument("--path_to_labels_dir", required=True)
    parser.add_argument("--path_to_features_dir", required=True)
    parser.add_argument("--path_to_tokenized_timelines_dir", required=True)
    parser.add_argument("--path_to_model", required=True)
    parser.add_argument("--path_to_subject_splits", default=None)
    parser.add_argument("--path_to_output_json", required=True)
    parser.add_argument("--model_name", default="mimiciv_gpt_small_10ep_medsnorm_clmbr_512")
    parser.add_argument("--tasks", nargs="+", default=["long_los", "death"])
    parser.add_argument("--embed_strat", default="last_nonpad")
    parser.add_argument("--chunk_strat", default="last")
    parser.add_argument("--sample_labels_per_task", type=int, default=1000)
    return parser.parse_args()


def parse_time(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value)


def load_label_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            rows.append(
                {
                    "patient_id": int(row["patient_id"]),
                    "prediction_time": parse_time(row["prediction_time"]),
                    "value": row["value"].lower() == "true",
                }
            )
    return rows


def convert_event(raw_event: Any) -> Event:
    if hasattr(raw_event, "start"):
        return Event(
            code=raw_event.code,
            value=raw_event.value,
            unit=raw_event.unit,
            start=raw_event.start,
            end=raw_event.end,
            omop_table=raw_event.omop_table,
        )
    return Event(
        code=raw_event.code,
        value=getattr(raw_event, "numeric_value", None) or getattr(raw_event, "text_value", None),
        unit=getattr(raw_event, "unit", None),
        start=getattr(raw_event, "time", None),
        end=getattr(raw_event, "end", None),
        omop_table=getattr(raw_event, "omop_table", None),
    )


def event_time(event: Event) -> Optional[datetime.datetime]:
    return event.start


def summarize_array(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def hash_rows(arr: np.ndarray, decimals: Optional[int] = None) -> int:
    if decimals is not None:
        arr = np.round(arr, decimals=decimals)
    hashes = set()
    for row in arr:
        hashes.add(hashlib.blake2b(np.ascontiguousarray(row).view(np.uint8), digest_size=8).digest())
    return len(hashes)


def top_tokens(ids: np.ndarray, tokenizer: Any, n: int = 10) -> List[Dict[str, Any]]:
    counter = collections.Counter(int(x) for x in ids.tolist())
    rows = []
    for token_id, count in counter.most_common(n):
        rows.append(
            {
                "token_id": token_id,
                "token": tokenizer.idx_2_token.get(token_id, "<missing>"),
                "count": int(count),
            }
        )
    return rows


def summarize_tokens(path: str, tokenizer: Any) -> Dict[str, Any]:
    timelines = np.load(path)["tokenized_timelines"]
    pad_id = tokenizer.token_2_idx["[PAD]"]
    nonpad = timelines != pad_id
    lengths = nonpad.sum(axis=1)
    last_token_ids = np.array([row[np.where(mask)[0][-1]] if mask.any() else pad_id for row, mask in zip(timelines, nonpad)])
    nonpad_token_ids = timelines[nonpad]
    return {
        "path": path,
        "shape": list(timelines.shape),
        "lengths": summarize_array(lengths),
        "unique_row_hashes_exact": hash_rows(timelines),
        "unique_token_ids": int(np.unique(nonpad_token_ids).shape[0]) if nonpad_token_ids.size else 0,
        "top_last_tokens": top_tokens(last_token_ids, tokenizer),
        "top_any_tokens": top_tokens(nonpad_token_ids, tokenizer),
    }


def summarize_features(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    x = np.asarray(payload["data_matrix"], dtype=np.float32)
    y = np.asarray(payload["label_values"], dtype=bool)
    patient_ids = np.asarray(payload["patient_ids"])
    per_dim_std = x.std(axis=0)
    row_norm = np.linalg.norm(x, axis=1)
    sample = x[: min(1000, x.shape[0])]
    sample_norm = sample / np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12)
    cosine_to_first = sample_norm @ sample_norm[0]
    return {
        "path": path,
        "shape": list(x.shape),
        "n_patients": int(np.unique(patient_ids).shape[0]),
        "n_positive": int(y.sum()),
        "prevalence": float(np.mean(y)),
        "global_std": float(x.std()),
        "mean_per_dim_std": float(per_dim_std.mean()),
        "max_per_dim_std": float(per_dim_std.max()),
        "near_zero_std_dims": int(np.sum(per_dim_std < 1e-6)),
        "row_norms": summarize_array(row_norm),
        "unique_row_hashes_rounded_6": hash_rows(x, decimals=6),
        "sample_cosine_to_first": summarize_array(cosine_to_first),
    }


def summarize_raw_coverage(rows: List[Dict[str, Any]], database: Any, tokenizer: Any) -> Dict[str, Any]:
    raw_counts = []
    token_counts = []
    last_raw_codes = collections.Counter()
    last_tokens = collections.Counter()
    empty_tokenized = 0
    for row in rows:
        events = [convert_event(event) for event in database[row["patient_id"]].events]
        valid_events = [event for event in events if event_time(event) is None or event_time(event) <= row["prediction_time"]]
        tokens = tokenizer.convert_events_to_tokens(valid_events)
        raw_counts.append(len(valid_events))
        token_counts.append(len(tokens))
        if valid_events:
            last_raw_codes[valid_events[-1].code] += 1
        if tokens:
            last_tokens[tokens[-1]] += 1
        else:
            empty_tokenized += 1
    return {
        "sample_size": len(rows),
        "raw_event_counts": summarize_array(raw_counts),
        "token_counts": summarize_array(token_counts),
        "token_to_raw_event_ratio": summarize_array(
            token_count / raw_count
            for raw_count, token_count in zip(raw_counts, token_counts)
            if raw_count > 0
        ),
        "empty_tokenized_count": int(empty_tokenized),
        "top_last_raw_codes": [{"code": code, "count": int(count)} for code, count in last_raw_codes.most_common(10)],
        "top_last_tokens": [{"token": token, "count": int(count)} for token, count in last_tokens.most_common(10)],
    }


def summarize_subject_splits(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"available": False}
    import polars as pl

    df = pl.read_parquet(path)
    rows = {}
    for split in sorted(df["split"].unique().to_list()):
        rows[str(split)] = int(df.filter(pl.col("split") == split).height)
    return {"available": True, "path": path, "counts": rows}


def summarize_label_splits(rows: List[Dict[str, Any]], path_to_subject_splits: Optional[str]) -> Dict[str, Any]:
    if not path_to_subject_splits or not os.path.exists(path_to_subject_splits):
        return {"available": False}
    patient_ids = np.asarray([row["patient_id"] for row in rows])
    values = np.asarray([row["value"] for row in rows], dtype=bool)
    splits = assign_splits(patient_ids, path_to_subject_splits)
    return {
        split: {
            "n_labels": int(np.sum(splits == split)),
            "n_patients": int(np.unique(patient_ids[splits == split]).shape[0]),
            "n_positive": int(values[splits == split].sum()),
            "prevalence": float(np.mean(values[splits == split])) if np.any(splits == split) else float("nan"),
        }
        for split in ["train", "val", "test"]
    }


def feature_path(base: str, model_name: str, task: str, embed_strat: str, chunk_strat: str) -> str:
    return os.path.join(base, f"{model_name}_{task}_{chunk_strat}_chunk:last_embed:{embed_strat}_features_1.pkl")


def timeline_path(base: str, model_name: str, task: str, embed_strat: str, chunk_strat: str) -> str:
    return os.path.join(base, f"{model_name}_{task}_{chunk_strat}_chunk:last_embed:{embed_strat}_tokenized_timelines.npz")


def main() -> None:
    args = parse_args()
    import meds_reader

    tokenizer = load_tokenizer_from_path(args.path_to_model)
    report: Dict[str, Any] = {
        "tokenizer": {
            "vocab_size": len(tokenizer.token_2_idx),
            "config_entries": len(tokenizer.tokenizer_config),
        },
        "subject_splits": summarize_subject_splits(args.path_to_subject_splits),
        "tasks": {},
    }

    with meds_reader.SubjectDatabase(args.path_to_database, num_threads=1) as database:
        for task in args.tasks:
            labels_path = os.path.join(args.path_to_labels_dir, task, "all_labels.csv")
            rows = load_label_rows(labels_path, limit=args.sample_labels_per_task)
            full_rows = load_label_rows(labels_path)
            values = np.asarray([row["value"] for row in full_rows], dtype=bool)
            report["tasks"][task] = {
                "labels": {
                    "path": labels_path,
                    "n_labels": len(full_rows),
                    "n_patients": len({row["patient_id"] for row in full_rows}),
                    "n_positive": int(values.sum()),
                    "prevalence": float(values.mean()),
                    "split_counts": summarize_label_splits(full_rows, args.path_to_subject_splits),
                },
                "raw_coverage_sample": summarize_raw_coverage(rows, database, tokenizer),
                "tokens": summarize_tokens(
                    timeline_path(args.path_to_tokenized_timelines_dir, args.model_name, task, args.embed_strat, args.chunk_strat),
                    tokenizer,
                ),
                "features": summarize_features(
                    feature_path(args.path_to_features_dir, args.model_name, task, args.embed_strat, args.chunk_strat)
                ),
            }

    os.makedirs(os.path.dirname(args.path_to_output_json), exist_ok=True)
    with open(args.path_to_output_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
