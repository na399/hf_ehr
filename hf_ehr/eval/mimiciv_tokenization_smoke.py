import argparse
import collections
import json
import os
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class SubjectTokenizationStats:
    patient_id: int
    raw_events: int
    clinical_tokens: int
    tokenized_nonpad_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MIMIC-IV MEDS reader events map to CLMBR clinical tokens.")
    parser.add_argument("--path_to_meds_reader_extract", required=True)
    parser.add_argument("--path_to_tokenizer_config", required=True)
    parser.add_argument("--tokenizer_cache_dir", default=None)
    parser.add_argument("--max_subjects", type=int, default=1024)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_subjects_with_tokens", type=int, default=25)
    parser.add_argument("--min_subject_token_fraction", type=float, default=0.05)
    parser.add_argument("--min_total_clinical_tokens", type=int, default=100)
    parser.add_argument("--min_unique_clinical_tokens", type=int, default=10)
    return parser.parse_args()


def convert_meds_event(raw_event: Any) -> Any:
    from hf_ehr.config import Event

    return Event(
        code=raw_event.code,
        value=getattr(raw_event, "numeric_value", None) or getattr(raw_event, "text_value", None),
        unit=getattr(raw_event, "unit", None),
        start=getattr(raw_event, "time", None),
        end=getattr(raw_event, "end", None),
        omop_table=getattr(raw_event, "omop_table", None),
    )


def get_patient_id(patient_id: Any) -> int:
    if hasattr(patient_id, "__len__") and not isinstance(patient_id, (str, bytes)):
        return int(patient_id[0])
    return int(patient_id)


def summarize_subject_stats(
    subject_stats: List[SubjectTokenizationStats],
    top_raw_codes: collections.Counter,
    top_clinical_tokens: collections.Counter,
    unique_clinical_token_ids: Iterable[int],
) -> Dict[str, Any]:
    raw_counts = [item.raw_events for item in subject_stats]
    clinical_counts = [item.clinical_tokens for item in subject_stats]
    nonpad_counts = [item.tokenized_nonpad_tokens for item in subject_stats]
    subjects_with_tokens = sum(count > 0 for count in clinical_counts)
    subjects_seen = len(subject_stats)

    return {
        "subjects_seen": subjects_seen,
        "subjects_with_clinical_tokens": subjects_with_tokens,
        "subject_token_fraction": subjects_with_tokens / subjects_seen if subjects_seen else 0.0,
        "total_raw_events": int(sum(raw_counts)),
        "total_clinical_tokens": int(sum(clinical_counts)),
        "event_token_coverage": (sum(clinical_counts) / sum(raw_counts)) if sum(raw_counts) else 0.0,
        "unique_clinical_token_ids": len(set(unique_clinical_token_ids)),
        "raw_events_per_subject": describe_counts(raw_counts),
        "clinical_tokens_per_subject": describe_counts(clinical_counts),
        "tokenized_nonpad_tokens_per_subject": describe_counts(nonpad_counts),
        "top_raw_codes": top_raw_codes.most_common(20),
        "top_clinical_tokens": top_clinical_tokens.most_common(20),
        "sample_subjects": [asdict(item) for item in subject_stats[:20]],
    }


def describe_counts(values: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
    }


def validate_summary(summary: Dict[str, Any], args: argparse.Namespace) -> None:
    failures = []
    if summary["subjects_with_clinical_tokens"] < args.min_subjects_with_tokens:
        failures.append(
            f"subjects_with_clinical_tokens={summary['subjects_with_clinical_tokens']} "
            f"< {args.min_subjects_with_tokens}"
        )
    if summary["subject_token_fraction"] < args.min_subject_token_fraction:
        failures.append(
            f"subject_token_fraction={summary['subject_token_fraction']:.4f} "
            f"< {args.min_subject_token_fraction}"
        )
    if summary["total_clinical_tokens"] < args.min_total_clinical_tokens:
        failures.append(
            f"total_clinical_tokens={summary['total_clinical_tokens']} "
            f"< {args.min_total_clinical_tokens}"
        )
    if summary["unique_clinical_token_ids"] < args.min_unique_clinical_tokens:
        failures.append(
            f"unique_clinical_token_ids={summary['unique_clinical_token_ids']} "
            f"< {args.min_unique_clinical_tokens}"
        )
    if failures:
        raise RuntimeError("MIMIC-IV CLMBR tokenization smoke failed: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    import meds_reader
    from hf_ehr.data.tokenization import CLMBRTokenizer

    tokenizer = CLMBRTokenizer(args.path_to_tokenizer_config, path_to_tokenizer_cache_dir=args.tokenizer_cache_dir)
    pad_token_id = tokenizer.token_2_idx["[PAD]"]

    subject_stats: List[SubjectTokenizationStats] = []
    raw_code_counter: collections.Counter = collections.Counter()
    clinical_token_counter: collections.Counter = collections.Counter()
    unique_clinical_token_ids = set()

    with meds_reader.SubjectDatabase(args.path_to_meds_reader_extract, num_threads=1) as database:
        for idx, patient_id in enumerate(database):
            if idx >= args.max_subjects:
                break
            pid = get_patient_id(patient_id)
            subject = database[patient_id]
            events = [convert_meds_event(event) for event in subject.events]
            raw_code_counter.update(event.code for event in events)

            clinical_tokens = tokenizer.convert_events_to_tokens(events)
            clinical_token_counter.update(clinical_tokens)
            unique_clinical_token_ids.update(tokenizer.token_2_idx[token] for token in clinical_tokens)
            tokenized_ids = tokenizer(events, add_special_tokens=True)["input_ids"][0]
            nonpad_count = sum(int(token_id != pad_token_id) for token_id in tokenized_ids)

            subject_stats.append(
                SubjectTokenizationStats(
                    patient_id=pid,
                    raw_events=len(events),
                    clinical_tokens=len(clinical_tokens),
                    tokenized_nonpad_tokens=nonpad_count,
                )
            )

    summary = summarize_subject_stats(
        subject_stats,
        raw_code_counter,
        clinical_token_counter,
        unique_clinical_token_ids,
    )
    summary["thresholds"] = {
        "min_subjects_with_tokens": args.min_subjects_with_tokens,
        "min_subject_token_fraction": args.min_subject_token_fraction,
        "min_total_clinical_tokens": args.min_total_clinical_tokens,
        "min_unique_clinical_tokens": args.min_unique_clinical_tokens,
    }
    summary["path_to_tokenizer_config"] = args.path_to_tokenizer_config
    summary["path_to_meds_reader_extract"] = args.path_to_meds_reader_extract

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    validate_summary(summary, args)


if __name__ == "__main__":
    main()
