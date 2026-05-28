from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bus_departure_predictor import (
    DEFAULT_BINS,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    fill_time_since_event,
    load_model_bundle,
    normalize_schema,
    parse_probability_bins,
    predict_departure,
    predict_departure_with_model,
    summarize_prediction_row,
    write_prediction,
)


DEFAULT_MANUAL_TRAINING_DIR = Path("outputs/training")
DEFAULT_AUTO_TRAINING_DIR = Path("outputs/training_auto")
DEFAULT_REALTIME_MODEL_PATH = Path("models/realtime_departure_random_forest.joblib")
DEFAULT_REALTIME_RETRAIN_STATE = Path("models/realtime_departure_random_forest.retrain_state.json")
DEFAULT_AUTO_RETRAIN_THRESHOLD_EVENTS = 15


def log(message: str) -> None:
    print(message, flush=True)


def strict_gate_out_events(df: pd.DataFrame) -> np.ndarray:
    if "gate_out_event_count" not in df.columns:
        raise ValueError(
            "Feature CSV is missing gate_out_event_count. Run bus_yolo_analyzer.py "
            "with gate ROI logic first."
        )

    events = (
        pd.to_numeric(df["gate_out_event_count"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
        > 0
    )
    return events


def bucket_for_delay(delay_sec: float, bins: list) -> str:
    for bucket in bins:
        if bucket.end_sec is None:
            if delay_sec >= bucket.start_sec:
                return bucket.label
        elif bucket.start_sec <= delay_sec < bucket.end_sec:
            return bucket.label
    raise ValueError(f"Could not bucket delay: {delay_sec}")


def prepare_feature_state(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = normalize_schema(raw_df)
    events = strict_gate_out_events(df)
    times = df["time_second"].to_numpy(dtype=float)
    df["departure_event"] = events.astype(int)
    df["time_since_last_departure"] = fill_time_since_event(times, events)
    return df, events


def build_labeled_batch(
    prepared_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    bins: list,
) -> pd.DataFrame:
    if end_idx < start_idx:
        return pd.DataFrame()

    batch = prepared_df.iloc[start_idx : end_idx + 1].copy().reset_index(drop=True)
    event_time = float(prepared_df.iloc[end_idx]["time_second"])
    batch["source_feature_row_index"] = np.arange(start_idx, end_idx + 1)
    batch["next_departure_time_second"] = event_time
    batch["time_until_next_departure"] = event_time - batch["time_second"].to_numpy(dtype=float)
    local_events = np.zeros(len(batch), dtype=int)
    local_events[-1] = 1
    batch["departure_event"] = local_events
    batch["auto_departure_event"] = local_events
    batch["auto_label_source"] = "gate_out"
    batch["departure_bucket"] = [
        bucket_for_delay(float(delay_sec), bins)
        for delay_sec in batch["time_until_next_departure"].to_numpy(dtype=float)
    ]
    batch["is_trainable"] = 1
    return batch


def load_existing_training_data(
    manual_training_dir: Path,
    auto_training_dir: Path,
    exclude_auto_csv: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory in (manual_training_dir, auto_training_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            if path.resolve() == exclude_auto_csv.resolve():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["source_training_csv"] = str(path)
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def count_auto_departure_events(auto_training_dir: Path) -> int:
    if not auto_training_dir.exists():
        return 0

    total = 0
    for path in sorted(auto_training_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty or "departure_event" not in df.columns:
            continue
        total += int(pd.to_numeric(df["departure_event"], errors="coerce").fillna(0).sum())
    return total


def load_retrain_state(state_path: Path) -> dict[str, object]:
    if not state_path.exists():
        return {
            "trained_auto_event_total": 0,
            "retrain_count": 0,
        }

    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_retrain_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def write_auto_training_csv(training_df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def maybe_retrain_model(
    train_script: Path,
    manual_training_dir: Path,
    auto_training_dir: Path,
    realtime_model_path: Path,
    retrain_state_path: Path,
    retrain_state: dict[str, object],
    bins: str,
    auto_retrain_threshold_events: int,
) -> tuple[dict[str, object] | None, dict[str, object], bool, int, int]:
    total_auto_events = count_auto_departure_events(auto_training_dir)
    trained_auto_event_total = int(retrain_state.get("trained_auto_event_total", 0))
    pending_auto_events = max(0, total_auto_events - trained_auto_event_total)
    model_exists = realtime_model_path.exists()

    should_retrain = False
    if not model_exists:
        should_retrain = total_auto_events >= auto_retrain_threshold_events
    elif pending_auto_events >= auto_retrain_threshold_events:
        should_retrain = True

    if not should_retrain:
        bundle = load_model_bundle(realtime_model_path) if model_exists else None
        return bundle, retrain_state, False, pending_auto_events, total_auto_events

    command = [sys.executable, str(train_script)]
    for training_dir in (manual_training_dir, auto_training_dir):
        if training_dir.exists():
            command.extend(["--training-dir", str(training_dir)])
    command.extend(
        [
            "--model-output",
            str(realtime_model_path),
            "--bins",
            bins,
        ]
    )
    log(
        "Retraining realtime model from "
        f"{manual_training_dir} + {auto_training_dir} "
        f"(pending auto events: {pending_auto_events})"
    )
    subprocess.run(command, check=True)

    updated_state = dict(retrain_state)
    updated_state["trained_auto_event_total"] = total_auto_events
    updated_state["retrain_count"] = int(updated_state.get("retrain_count", 0)) + 1
    updated_state["last_retrained_model"] = str(realtime_model_path)
    save_retrain_state(retrain_state_path, updated_state)
    bundle = load_model_bundle(realtime_model_path)
    return bundle, updated_state, True, pending_auto_events, total_auto_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a realtime departure-learning pipeline by reading a full feature CSV, "
            "emitting cumulative prefix predictions, auto-labeling gate-out events, and "
            "retraining a separate realtime model when enough events accumulate."
        )
    )
    parser.add_argument("--csv", required=True, type=Path, help="YOLO feature CSV path.")
    parser.add_argument(
        "--progressive-output-csv",
        default=None,
        type=Path,
        help="Progressive prefix prediction CSV. Defaults to outputs/<input>_realtime_progressive.csv.",
    )
    parser.add_argument(
        "--final-output-csv",
        default=None,
        type=Path,
        help="Final departure probability CSV for the latest row.",
    )
    parser.add_argument(
        "--auto-training-output",
        default=None,
        type=Path,
        help="Auto-labeled training CSV path. Defaults to outputs/training_auto/<input>_gate_out_training.csv.",
    )
    parser.add_argument(
        "--manual-training-dir",
        default=DEFAULT_MANUAL_TRAINING_DIR,
        type=Path,
        help="Directory containing manually labeled training CSVs.",
    )
    parser.add_argument(
        "--auto-training-dir",
        default=DEFAULT_AUTO_TRAINING_DIR,
        type=Path,
        help="Directory where gate-out auto-labeled training CSVs are stored.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL_PATH,
        type=Path,
        help="Base offline model used before a realtime model exists.",
    )
    parser.add_argument(
        "--realtime-model",
        default=DEFAULT_REALTIME_MODEL_PATH,
        type=Path,
        help="Realtime-learning model output path.",
    )
    parser.add_argument(
        "--retrain-state",
        default=DEFAULT_REALTIME_RETRAIN_STATE,
        type=Path,
        help="JSON state file that tracks how many auto events were already retrained.",
    )
    parser.add_argument("--bins", default=DEFAULT_BINS)
    parser.add_argument("--min-samples", default=12, type=int)
    parser.add_argument("--neighbors", default=40, type=int)
    parser.add_argument("--default-interval-sec", default=300.0, type=float)
    parser.add_argument(
        "--auto-retrain-threshold-events",
        default=DEFAULT_AUTO_RETRAIN_THRESHOLD_EVENTS,
        type=int,
        help="Retrain after this many new gate-out departure events accumulate. Default is 15.",
    )
    parser.add_argument(
        "--train-script",
        default=Path("train_departure_model.py"),
        type=Path,
        help="Training script used for automatic retraining.",
    )
    parser.add_argument(
        "--no-auto-retrain",
        action="store_true",
        help="Disable automatic retraining and only write progressive predictions + auto labels.",
    )
    parser.add_argument(
        "--progress-every",
        default=100,
        type=int,
        help="Print progressive simulation progress every N rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV does not exist: {args.csv}")
    if args.min_samples < 1:
        raise ValueError("--min-samples must be greater than zero")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be greater than zero")
    if args.default_interval_sec <= 0:
        raise ValueError("--default-interval-sec must be greater than zero")
    if args.auto_retrain_threshold_events < 1:
        raise ValueError("--auto-retrain-threshold-events must be greater than zero")

    bins = parse_probability_bins(args.bins)
    raw_df = pd.read_csv(args.csv)
    if raw_df.empty:
        raise ValueError(f"CSV is empty: {args.csv}")

    progressive_output_csv = (
        args.progressive_output_csv
        if args.progressive_output_csv is not None
        else DEFAULT_OUTPUT_DIR / f"{args.csv.stem}_realtime_progressive.csv"
    )
    final_output_csv = (
        args.final_output_csv
        if args.final_output_csv is not None
        else DEFAULT_OUTPUT_DIR / f"{args.csv.stem}_realtime_probability.csv"
    )
    auto_training_output = (
        args.auto_training_output
        if args.auto_training_output is not None
        else args.auto_training_dir / f"{args.csv.stem}_gate_out_training.csv"
    )
    auto_training_output.parent.mkdir(parents=True, exist_ok=True)
    if auto_training_output.exists():
        auto_training_output.unlink()

    prepared_df, gate_out_events = prepare_feature_state(raw_df)
    historical_training_df = load_existing_training_data(
        manual_training_dir=args.manual_training_dir,
        auto_training_dir=args.auto_training_dir,
        exclude_auto_csv=auto_training_output,
    )

    retrain_state = load_retrain_state(args.retrain_state)
    active_model_path = None
    if args.realtime_model.exists():
        bundle = load_model_bundle(args.realtime_model)
        active_model_path = args.realtime_model
    elif args.base_model.exists():
        bundle = load_model_bundle(args.base_model)
        active_model_path = args.base_model
    else:
        bundle = None

    current_auto_training_df = pd.DataFrame()
    progressive_rows: list[dict[str, object]] = []
    final_prediction = None
    last_batch_start = 0

    for row_index in range(len(prepared_df)):
        if args.progress_every > 0 and (row_index == 0 or (row_index + 1) % args.progress_every == 0):
            log(f"Realtime simulation row {row_index + 1}/{len(prepared_df)}")

        current_row = prepared_df.iloc[[row_index]].copy().reset_index(drop=True)
        if bundle is not None:
            prediction = predict_departure_with_model(
                df=current_row,
                row_index=0,
                bundle=bundle,
            )
        else:
            labeled_frames = []
            if not historical_training_df.empty:
                labeled_frames.append(historical_training_df)
            if not current_auto_training_df.empty:
                labeled_frames.append(current_auto_training_df)

            current_row["time_until_next_departure"] = np.nan
            current_row["departure_bucket"] = ""
            fallback_df = pd.concat(labeled_frames + [current_row], ignore_index=True, sort=False)
            prediction = predict_departure(
                df=fallback_df,
                row_index=len(fallback_df) - 1,
                buckets=bins,
                min_samples=args.min_samples,
                neighbors=args.neighbors,
                default_interval_sec=args.default_interval_sec,
            )

        final_prediction = prediction
        row_summary = summarize_prediction_row(
            prediction,
            prefix_rows=row_index + 1,
            current_row_index=row_index,
        )
        row_summary["current_gate_out_event"] = int(gate_out_events[row_index])
        row_summary["seen_gate_out_events"] = int(gate_out_events[: row_index + 1].sum())
        row_summary["available_auto_training_rows"] = int(len(current_auto_training_df))
        row_summary["available_history_training_rows"] = int(len(historical_training_df))
        row_summary["active_model_path"] = "" if active_model_path is None else str(active_model_path)
        row_summary["retrain_count"] = int(retrain_state.get("retrain_count", 0))
        row_summary["retrained_after_row"] = 0

        if gate_out_events[row_index]:
            new_batch = build_labeled_batch(
                prepared_df=prepared_df,
                start_idx=last_batch_start,
                end_idx=row_index,
                bins=bins,
            )
            if not new_batch.empty:
                current_auto_training_df = pd.concat(
                    [current_auto_training_df, new_batch],
                    ignore_index=True,
                    sort=False,
                )
                write_auto_training_csv(current_auto_training_df, auto_training_output)

            if not args.no_auto_retrain:
                bundle, retrain_state, retrained, pending_events, total_auto_events = maybe_retrain_model(
                    train_script=args.train_script,
                    manual_training_dir=args.manual_training_dir,
                    auto_training_dir=args.auto_training_dir,
                    realtime_model_path=args.realtime_model,
                    retrain_state_path=args.retrain_state,
                    retrain_state=retrain_state,
                    bins=args.bins,
                    auto_retrain_threshold_events=args.auto_retrain_threshold_events,
                )
                row_summary["pending_auto_events"] = pending_events
                row_summary["total_auto_events"] = total_auto_events
                if retrained:
                    active_model_path = args.realtime_model
                    row_summary["active_model_path"] = str(active_model_path)
                    row_summary["retrain_count"] = int(retrain_state.get("retrain_count", 0))
                    row_summary["retrained_after_row"] = 1

            last_batch_start = row_index + 1

        progressive_rows.append(row_summary)

    if final_prediction is None:
        raise RuntimeError("No progressive prediction was produced.")

    progressive_df = pd.DataFrame(progressive_rows)
    progressive_output_csv.parent.mkdir(parents=True, exist_ok=True)
    progressive_df.to_csv(progressive_output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    write_prediction(final_prediction, final_output_csv)

    if current_auto_training_df.empty:
        log("No gate-out events were found, so no auto-labeled training rows were generated.")
    else:
        write_auto_training_csv(current_auto_training_df, auto_training_output)

    best_bucket = max(final_prediction.buckets, key=lambda row: float(row["probability"]))
    log(f"Input CSV: {args.csv}")
    log(f"Rows processed sequentially: {len(prepared_df)}")
    log(f"Gate-out events used as labels: {int(gate_out_events.sum())}")
    log(f"Most likely final departure window: {best_bucket['bucket']} ({best_bucket['probability_percent']}%)")
    log(f"Saved progressive prediction CSV to {progressive_output_csv}")
    log(f"Saved final probability CSV to {final_output_csv}")
    if not current_auto_training_df.empty:
        log(f"Saved auto-labeled gate-out training CSV to {auto_training_output}")
    if args.realtime_model.exists():
        log(f"Realtime model path: {args.realtime_model}")


if __name__ == "__main__":
    main()
