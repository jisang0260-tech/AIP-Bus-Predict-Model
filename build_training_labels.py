from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from bus_departure_predictor import (
    DEFAULT_BINS,
    fill_time_since_event,
    normalize_schema,
    parse_probability_bins,
    seconds_to_hhmmss,
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_time_to_seconds(value: str) -> float:
    parts = str(value).strip().split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Expected MM:SS or HH:MM:SS, got {value!r}")

    numbers = [float(part) for part in parts]
    if len(numbers) == 2:
        minute, second = numbers
        return minute * 60 + second

    hour, minute, second = numbers
    return hour * 3600 + minute * 60 + second


def read_departure_times(label_csv: Path) -> list[float]:
    labels = pd.read_csv(label_csv)
    if labels.empty:
        raise ValueError(f"Label CSV is empty: {label_csv}")

    if "event_time_second" in labels.columns:
        times = pd.to_numeric(labels["event_time_second"], errors="coerce")
    elif "event_time_hhmmss" in labels.columns:
        times = labels["event_time_hhmmss"].map(parse_time_to_seconds)
    else:
        raise ValueError(
            "Label CSV must contain event_time_second or event_time_hhmmss."
        )

    cleaned = sorted(float(value) for value in times.dropna().tolist())
    if not cleaned:
        raise ValueError(f"No valid departure times found in {label_csv}")

    return cleaned


def bucket_for_delay(delay_sec: float, bins: str) -> str:
    for bucket in parse_probability_bins(bins):
        if bucket.end_sec is None:
            if delay_sec >= bucket.start_sec:
                return bucket.label
        elif bucket.start_sec <= delay_sec < bucket.end_sec:
            return bucket.label
    raise ValueError(f"Could not bucket delay: {delay_sec}")


def add_manual_targets(
    features: pd.DataFrame,
    departure_times: list[float],
    bins: str,
    event_tolerance_sec: float,
    drop_after_last_event: bool,
) -> pd.DataFrame:
    df = normalize_schema(features)
    times = df["time_second"].to_numpy(dtype=float)
    event_times = np.asarray(departure_times, dtype=float)

    event_flags = np.zeros(len(df), dtype=bool)
    for event_time in event_times:
        nearest_idx = int(np.argmin(np.abs(times - event_time)))
        if abs(times[nearest_idx] - event_time) <= event_tolerance_sec:
            event_flags[nearest_idx] = True
        else:
            log(
                "Warning: no frame close to departure "
                f"{seconds_to_hhmmss(event_time)} within {event_tolerance_sec}s"
            )

    next_departure = np.full(len(df), np.nan)
    event_idx = 0
    for row_idx, current_time in enumerate(times):
        while event_idx < len(event_times) and event_times[event_idx] < current_time:
            event_idx += 1
        if event_idx < len(event_times):
            next_departure[row_idx] = event_times[event_idx]

    df["departure_event"] = event_flags.astype(int)
    df["manual_departure_event"] = df["departure_event"]
    df["time_since_last_departure"] = fill_time_since_event(times, event_flags)
    df["next_departure_time_second"] = next_departure
    df["next_departure_time_hhmmss"] = [
        "" if np.isnan(value) else seconds_to_hhmmss(float(value))
        for value in next_departure
    ]
    df["time_until_next_departure"] = next_departure - times
    df["departure_bucket"] = [
        "" if np.isnan(value) else bucket_for_delay(float(value), bins)
        for value in df["time_until_next_departure"]
    ]
    df["is_trainable"] = np.isfinite(df["time_until_next_departure"]).astype(int)

    if drop_after_last_event:
        df = df[df["is_trainable"] == 1].reset_index(drop=True)

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine YOLO feature CSV with manually labeled bus departure times "
            "to create a training CSV."
        )
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bins", default=DEFAULT_BINS)
    parser.add_argument("--event-tolerance-sec", default=0.6, type=float)
    parser.add_argument(
        "--keep-after-last-event",
        action="store_true",
        help="Keep rows after the last labeled departure with blank targets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Feature CSV does not exist: {args.features}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label CSV does not exist: {args.labels}")
    if args.event_tolerance_sec < 0:
        raise ValueError("--event-tolerance-sec cannot be negative")

    features = pd.read_csv(args.features)
    if features.empty:
        raise ValueError(f"Feature CSV is empty: {args.features}")

    departure_times = read_departure_times(args.labels)
    training_df = add_manual_targets(
        features=features,
        departure_times=departure_times,
        bins=args.bins,
        event_tolerance_sec=args.event_tolerance_sec,
        drop_after_last_event=not args.keep_after_last_event,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)

    log(f"Feature rows: {len(features)}")
    log(f"Manual departure labels: {len(departure_times)}")
    log(f"Training rows written: {len(training_df)}")
    log(f"Saved training CSV to {args.output}")


if __name__ == "__main__":
    main()
