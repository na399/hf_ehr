import argparse
import csv
import datetime
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


ADMISSION_CODES = ("Visit//9201", "Visit//262")
DEATH_CODE = "MEDS_DEATH"
TASKS = ("death", "long_los")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIMIC-IV EHRSHOT-style boolean labels from a MEDS reader extract.")
    parser.add_argument("--path_to_meds_reader_extract", required=True)
    parser.add_argument("--path_to_labels_dir", required=True)
    parser.add_argument("--prediction_hours_after_admission", type=float, default=48.0)
    parser.add_argument("--long_los_days", type=float, default=7.0)
    parser.add_argument("--min_prior_history_days", type=float, default=730.0)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser.parse_args()


def get_time(event: Any) -> Optional[datetime.datetime]:
    return getattr(event, "time", None)


def pair_intervals(events: Iterable[Any], base_code: str) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    starts: List[datetime.datetime] = []
    ends: List[datetime.datetime] = []
    for event in events:
        event_time = get_time(event)
        if event_time is None:
            continue
        if event.code == f"{base_code}//start":
            starts.append(event_time)
        elif event.code == f"{base_code}//end":
            ends.append(event_time)
    intervals = []
    for start, end in zip(starts, ends):
        if end > start:
            intervals.append((start, end))
    return intervals


def has_prior_history(
    events: Iterable[Any],
    prediction_time: datetime.datetime,
    min_prior_history: datetime.timedelta,
) -> bool:
    if min_prior_history <= datetime.timedelta(0):
        return True
    cutoff = prediction_time - min_prior_history
    return any((event_time := get_time(event)) is not None and event_time <= cutoff for event in events)


def iter_labels(
    subject: Any,
    prediction_offset: datetime.timedelta,
    long_los: datetime.timedelta,
    min_prior_history: datetime.timedelta,
) -> Iterable[Dict[str, str]]:
    events = list(subject.events)
    deaths = [get_time(event) for event in events if event.code == DEATH_CODE and get_time(event) is not None]
    death_time = min(deaths) if deaths else None

    seen = set()
    for admission_code in ADMISSION_CODES:
        for admission_start, admission_end in pair_intervals(events, admission_code):
            key = (admission_start, admission_end)
            if key in seen:
                continue
            seen.add(key)

            prediction_time = admission_start + prediction_offset
            if prediction_time >= admission_end:
                continue
            if death_time is not None and prediction_time >= death_time:
                continue
            if not has_prior_history(events, prediction_time, min_prior_history):
                continue

            yield {
                "task": "death",
                "patient_id": str(int(subject.subject_id)),
                "prediction_time": prediction_time.isoformat(timespec="minutes"),
                "value": str(death_time is not None and admission_start <= death_time <= admission_end).lower(),
                "label_type": "boolean",
            }
            yield {
                "task": "long_los",
                "patient_id": str(int(subject.subject_id)),
                "prediction_time": prediction_time.isoformat(timespec="minutes"),
                "value": str(admission_end - admission_start > long_los).lower(),
                "label_type": "boolean",
            }


def main() -> None:
    args = parse_args()
    import meds_reader

    os.makedirs(args.path_to_labels_dir, exist_ok=True)
    output_files = {
        task: os.path.join(args.path_to_labels_dir, task, "all_labels.csv")
        for task in args.tasks
    }
    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    prediction_offset = datetime.timedelta(hours=args.prediction_hours_after_admission)
    long_los = datetime.timedelta(days=args.long_los_days)
    min_prior_history = datetime.timedelta(days=args.min_prior_history_days)

    n_subjects = 0
    n_labels = {task: 0 for task in args.tasks}
    with meds_reader.SubjectDatabase(args.path_to_meds_reader_extract, num_threads=1) as database:
        files = {task: open(path, "w", newline="") for task, path in output_files.items()}
        try:
            writers = {
                task: csv.DictWriter(f, fieldnames=["patient_id", "prediction_time", "value", "label_type"])
                for task, f in files.items()
            }
            for writer in writers.values():
                writer.writeheader()
            for idx, patient_id in enumerate(tqdm(database, desc="Generating MIMIC-IV labels")):
                if args.max_subjects is not None and idx >= args.max_subjects:
                    break
                subject = database[patient_id]
                n_subjects += 1
                for row in iter_labels(subject, prediction_offset, long_los, min_prior_history):
                    task = row.pop("task")
                    if task not in writers:
                        continue
                    writers[task].writerow(row)
                    n_labels[task] += 1
        finally:
            for f in files.values():
                f.close()

    for task, output_file in output_files.items():
        print(f"Wrote {n_labels[task]} {task} labels from {n_subjects} subjects to {output_file}")
    if any(count == 0 for count in n_labels.values()):
        raise RuntimeError(f"No labels generated for at least one task: {n_labels}")


if __name__ == "__main__":
    main()
