from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_BINS = "0-30,30-60,60-120,120-180,180-300,300-600"
FEATURE_COLUMNS = (
    "bus_count_inside",
    "bus_count_diff",
    "total_waiting_time",
    "avg_waiting_time",
    "max_waiting_time",
    "avg_area",
    "avg_area_diff",
    "max_area",
    "seconds_since_last_new_bus",
    "time_since_last_departure",
)


@dataclass(frozen=True)
class ProbabilityBucket:
    label: str
    start_sec: float
    end_sec: float | None


@dataclass(frozen=True)
class Prediction:
    method: str
    row_index: int
    current_time_second: float
    expected_departure_in_sec: float
    buckets: list[dict[str, object]]


def log(message: str) -> None:
    print(message, flush=True)


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(round(seconds)) % (24 * 3600)
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def parse_probability_bins(value: str) -> list[ProbabilityBucket]:
    buckets: list[ProbabilityBucket] = []
    previous_end = 0.0

    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" not in part:
            raise argparse.ArgumentTypeError(
                "Each bucket must use start-end format, for example 0-30"
            )

        start_text, end_text = part.split("-", 1)
        start_sec = float(start_text)
        end_sec = float(end_text)

        if start_sec < 0 or end_sec <= start_sec:
            raise argparse.ArgumentTypeError(f"Invalid bucket: {part}")

        if buckets and start_sec < previous_end:
            raise argparse.ArgumentTypeError("Buckets must be sorted and non-overlapping")

        previous_end = end_sec
        buckets.append(
            ProbabilityBucket(
                label=f"{int(start_sec)}-{int(end_sec)} sec",
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    if not buckets:
        raise argparse.ArgumentTypeError("At least one probability bucket is required")

    last_end = buckets[-1].end_sec
    assert last_end is not None
    buckets.append(
        ProbabilityBucket(
            label=f">{int(last_end)} sec",
            start_sec=last_end,
            end_sec=None,
        )
    )
    return buckets


def find_latest_csv(output_dir: Path) -> Path:
    candidates = [
        path
        for path in output_dir.glob("*.csv")
        if "departure_probability" not in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {output_dir}. Run bus_yolo_analyzer.py first "
            "or pass --csv explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_ids(value: object) -> set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()

    text = str(value).strip()
    if not text:
        return set()

    ids: set[int] = set()
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(float(part)))
        except ValueError:
            continue
    return ids


def coerce_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> None:
    if column not in df.columns:
        df[column] = default
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)


def normalize_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    aliases = {
        "count": "bus_count_inside",
        "count_diff": "bus_count_diff",
    }
    for old_name, new_name in aliases.items():
        if new_name not in df.columns and old_name in df.columns:
            df[new_name] = df[old_name]

    if "frame_index" not in df.columns:
        df["frame_index"] = np.arange(len(df))

    if "time_second" not in df.columns:
        df["time_second"] = df["frame_index"]

    if "total_waiting_time" not in df.columns and "remaining_time" in df.columns:
        df["total_waiting_time"] = df["remaining_time"]

    numeric_defaults = {
        "frame_index": 0.0,
        "time_second": 0.0,
        "bus_count_inside": 0.0,
        "bus_count_diff": 0.0,
        "new_bus_count": 0.0,
        "exited_bus_count": np.nan,
        "avg_area": 0.0,
        "avg_area_diff": 0.0,
        "max_area": 0.0,
        "total_waiting_time": 0.0,
        "avg_waiting_time": 0.0,
        "max_waiting_time": 0.0,
        "seconds_since_last_new_bus": np.nan,
    }
    for column, default in numeric_defaults.items():
        coerce_numeric(df, column, default)

    if "exited_bus_count" not in raw_df.columns:
        df["exited_bus_count"] = np.maximum(-df["bus_count_diff"], 0)
    else:
        df["exited_bus_count"] = df["exited_bus_count"].fillna(0)

    if "new_bus_count" not in raw_df.columns:
        df["new_bus_count"] = np.maximum(df["bus_count_diff"], 0)

    if df["seconds_since_last_new_bus"].isna().any():
        df["seconds_since_last_new_bus"] = fill_time_since_event(
            times=df["time_second"].to_numpy(dtype=float),
            events=df["new_bus_count"].to_numpy(dtype=float) > 0,
        )

    return df.reset_index(drop=True)


def infer_departure_events(df: pd.DataFrame) -> np.ndarray:
    if "exited_bus_count" in df.columns and df["exited_bus_count"].sum() > 0:
        return df["exited_bus_count"].to_numpy(dtype=float) > 0

    if "bus_ids_inside" in df.columns:
        id_sets = [parse_ids(value) for value in df["bus_ids_inside"]]
        events = np.zeros(len(df), dtype=bool)
        for idx in range(1, len(id_sets)):
            events[idx] = len(id_sets[idx - 1] - id_sets[idx]) > 0
        if events.any():
            return events

    return df["bus_count_diff"].to_numpy(dtype=float) < 0


def fill_time_since_event(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    result = np.zeros(len(times), dtype=float)
    last_event_time: float | None = None

    for idx, current_time in enumerate(times):
        if events[idx]:
            last_event_time = current_time

        if last_event_time is None:
            result[idx] = max(0.0, current_time - times[0])
        else:
            result[idx] = max(0.0, current_time - last_event_time)

    return result


def add_departure_targets(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    times = enriched["time_second"].to_numpy(dtype=float)
    events = infer_departure_events(enriched)

    enriched["departure_event"] = events.astype(int)
    enriched["time_since_last_departure"] = fill_time_since_event(times, events)

    next_departure_times = np.full(len(enriched), np.nan)
    next_event_time: float | None = None
    for idx in range(len(enriched) - 1, -1, -1):
        if events[idx]:
            next_event_time = times[idx]
        if next_event_time is not None:
            next_departure_times[idx] = next_event_time

    enriched["time_until_next_departure"] = next_departure_times - times
    return enriched


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    features = df.copy()
    for column in FEATURE_COLUMNS:
        coerce_numeric(features, column, 0.0)
    return features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)


def robust_scaled_distance(
    train_features: np.ndarray,
    current_features: np.ndarray,
) -> np.ndarray:
    median = np.nanmedian(train_features, axis=0)
    q75 = np.nanpercentile(train_features, 75, axis=0)
    q25 = np.nanpercentile(train_features, 25, axis=0)
    scale = q75 - q25
    scale = np.where(scale < 1.0, 1.0, scale)

    train_scaled = (np.nan_to_num(train_features, nan=0.0) - median) / scale
    current_scaled = (np.nan_to_num(current_features, nan=0.0) - median) / scale
    return np.linalg.norm(train_scaled - current_scaled, axis=1)


def bucket_probabilities_from_samples(
    labels: np.ndarray,
    weights: np.ndarray,
    buckets: list[ProbabilityBucket],
) -> tuple[list[float], float]:
    probabilities = np.full(len(buckets), 1e-6, dtype=float)

    for label, weight in zip(labels, weights):
        for bucket_idx, bucket in enumerate(buckets):
            if bucket.end_sec is None:
                if label >= bucket.start_sec:
                    probabilities[bucket_idx] += weight
                    break
            elif bucket.start_sec <= label < bucket.end_sec:
                probabilities[bucket_idx] += weight
                break

    probabilities = probabilities / probabilities.sum()
    expected_delay = float(np.average(labels, weights=weights))
    return probabilities.tolist(), expected_delay


def weibull_cdf(seconds: float, scale: float, shape: float) -> float:
    if seconds <= 0:
        return 0.0
    return 1.0 - math.exp(-((seconds / scale) ** shape))


def fallback_distribution(
    df: pd.DataFrame,
    row_index: int,
    buckets: list[ProbabilityBucket],
    default_interval_sec: float,
) -> tuple[list[float], float]:
    row = df.iloc[row_index]
    event_times = df.loc[df["departure_event"] > 0, "time_second"].to_numpy(dtype=float)
    if len(event_times) >= 2:
        base_interval = float(np.median(np.diff(event_times)))
    else:
        base_interval = default_interval_sec

    count = max(0.0, float(row.get("bus_count_inside", 0.0)))
    total_wait = max(0.0, float(row.get("total_waiting_time", 0.0)))
    max_wait = max(0.0, float(row.get("max_waiting_time", 0.0)))
    since_departure = max(0.0, float(row.get("time_since_last_departure", 0.0)))

    if count <= 0:
        expected_delay = base_interval * 1.8
        shape = 1.05
    else:
        wait_pressure = min(max_wait / max(base_interval, 1.0), 2.5)
        total_pressure = min(total_wait / max(base_interval * count, 1.0), 2.5)
        count_pressure = min(count, 4.0) / 4.0
        recent_departure_penalty = 0.35 if since_departure < 20 else 0.0
        pressure = 0.55 * wait_pressure + 0.25 * total_pressure + 0.20 * count_pressure
        expected_delay = base_interval * math.exp(-(pressure - recent_departure_penalty))
        shape = 1.35

    max_bucket_start = buckets[-1].start_sec
    expected_delay = min(max(expected_delay, 5.0), max_bucket_start * 2.5)
    scale = expected_delay / math.gamma(1.0 + 1.0 / shape)

    probabilities: list[float] = []
    previous_cdf = 0.0
    for bucket in buckets:
        if bucket.end_sec is None:
            probabilities.append(max(0.0, 1.0 - previous_cdf))
        else:
            end_cdf = weibull_cdf(bucket.end_sec, scale, shape)
            probabilities.append(max(0.0, end_cdf - previous_cdf))
            previous_cdf = end_cdf

    total = sum(probabilities)
    if total <= 0:
        return [1.0 / len(probabilities)] * len(probabilities), expected_delay

    return [probability / total for probability in probabilities], expected_delay


def predict_departure(
    df: pd.DataFrame,
    row_index: int,
    buckets: list[ProbabilityBucket],
    min_samples: int,
    neighbors: int,
    default_interval_sec: float,
) -> Prediction:
    labels = df["time_until_next_departure"].to_numpy(dtype=float)
    valid_mask = np.isfinite(labels) & (labels >= 0)
    valid_mask[row_index] = False

    current_time = float(df.loc[row_index, "time_second"])
    method = "weighted_history"

    if int(valid_mask.sum()) >= min_samples:
        train_features = feature_matrix(df.loc[valid_mask])
        current_features = feature_matrix(df.iloc[[row_index]])
        distances = robust_scaled_distance(train_features, current_features)
        neighbor_count = min(max(1, neighbors), len(distances))
        neighbor_indices = np.argsort(distances)[:neighbor_count]

        neighbor_labels = labels[valid_mask][neighbor_indices]
        neighbor_distances = distances[neighbor_indices]
        weights = np.exp(-0.5 * np.square(neighbor_distances))
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)

        probabilities, expected_delay = bucket_probabilities_from_samples(
            neighbor_labels,
            weights,
            buckets,
        )
    else:
        method = "heuristic_fallback"
        probabilities, expected_delay = fallback_distribution(
            df=df,
            row_index=row_index,
            buckets=buckets,
            default_interval_sec=default_interval_sec,
        )

    rows = []
    for bucket, probability in zip(buckets, probabilities):
        eta_start = current_time + bucket.start_sec
        eta_end = current_time + bucket.end_sec if bucket.end_sec is not None else None
        rows.append(
            {
                "bucket": bucket.label,
                "start_sec": bucket.start_sec,
                "end_sec": "" if bucket.end_sec is None else bucket.end_sec,
                "probability": round(float(probability), 6),
                "probability_percent": round(float(probability) * 100, 2),
                "eta_start_hhmmss": seconds_to_hhmmss(eta_start),
                "eta_end_hhmmss": "" if eta_end is None else seconds_to_hhmmss(eta_end),
            }
        )

    return Prediction(
        method=method,
        row_index=row_index,
        current_time_second=current_time,
        expected_departure_in_sec=expected_delay,
        buckets=rows,
    )


def write_prediction(prediction: Prediction, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction.buckets).to_csv(
        output_csv,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read YOLO bus CSV features and predict a probability distribution "
            "for when the next bus will depart the ROI."
        )
    )
    parser.add_argument("--csv", default=None, type=Path, help="YOLO feature CSV path.")
    parser.add_argument(
        "--output-csv",
        default=None,
        type=Path,
        help="Prediction CSV path. Defaults to outputs/<input>_departure_probability.csv.",
    )
    parser.add_argument(
        "--row-index",
        default=-1,
        type=int,
        help="CSV row to predict from. Default is the latest row.",
    )
    parser.add_argument("--bins", default=DEFAULT_BINS, type=parse_probability_bins)
    parser.add_argument("--min-samples", default=12, type=int)
    parser.add_argument("--neighbors", default=40, type=int)
    parser.add_argument("--default-interval-sec", default=300.0, type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_csv = args.csv if args.csv is not None else find_latest_csv(DEFAULT_OUTPUT_DIR)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV does not exist: {input_csv}")

    if args.min_samples < 1:
        raise ValueError("--min-samples must be greater than zero")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be greater than zero")
    if args.default_interval_sec <= 0:
        raise ValueError("--default-interval-sec must be greater than zero")

    raw_df = pd.read_csv(input_csv)
    if raw_df.empty:
        raise ValueError(f"CSV is empty: {input_csv}")

    df = add_departure_targets(normalize_schema(raw_df))

    row_index = args.row_index
    if row_index < 0:
        row_index = len(df) + row_index
    if not 0 <= row_index < len(df):
        raise IndexError(f"--row-index is out of range for {len(df)} rows")

    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else DEFAULT_OUTPUT_DIR / f"{input_csv.stem}_departure_probability.csv"
    )

    prediction = predict_departure(
        df=df,
        row_index=row_index,
        buckets=args.bins,
        min_samples=args.min_samples,
        neighbors=args.neighbors,
        default_interval_sec=args.default_interval_sec,
    )
    write_prediction(prediction, output_csv)

    best_bucket = max(prediction.buckets, key=lambda row: float(row["probability"]))
    expected_eta = prediction.current_time_second + prediction.expected_departure_in_sec
    event_count = int(df["departure_event"].sum())

    log(f"Input CSV: {input_csv}")
    log(f"Rows: {len(df)}, detected departure events: {event_count}")
    log(f"Method: {prediction.method}")
    log(
        "Most likely departure window: "
        f"{best_bucket['bucket']} ({best_bucket['probability_percent']}%)"
    )
    log(
        "Expected departure: "
        f"in {prediction.expected_departure_in_sec:.1f}s "
        f"around {seconds_to_hhmmss(expected_eta)}"
    )
    log(f"Saved probability table to {output_csv}")


if __name__ == "__main__":
    main()
