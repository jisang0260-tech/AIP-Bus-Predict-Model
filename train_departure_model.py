from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from bus_departure_predictor import (
    DEFAULT_BINS,
    FEATURE_COLUMNS,
    normalize_schema,
    parse_probability_bins,
)


DEFAULT_MODEL_PATH = Path("models/departure_random_forest.joblib")


def log(message: str) -> None:
    print(message, flush=True)


def collect_training_csvs(training_csv: Path | None, training_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    if training_csv is not None:
        paths.append(training_csv)
    if training_dir is not None:
        paths.extend(sorted(training_dir.glob("*.csv")))

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    if not unique_paths:
        raise FileNotFoundError("Pass --training-csv or --training-dir with CSV files.")

    missing = [path for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Training CSV does not exist: {missing[0]}")

    return unique_paths


def load_training_data(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_training_csv"] = str(path)
        frames.append(df)

    if not frames:
        raise ValueError("No non-empty training rows found.")

    return pd.concat(frames, ignore_index=True)


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time_until_next_departure", "departure_bucket"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")

    training_df = normalize_schema(df.copy())
    training_df["time_until_next_departure"] = pd.to_numeric(
        training_df["time_until_next_departure"],
        errors="coerce",
    )
    training_df = training_df[
        training_df["time_until_next_departure"].notna()
        & (training_df["time_until_next_departure"] >= 0)
        & training_df["departure_bucket"].notna()
        & (training_df["departure_bucket"].astype(str).str.len() > 0)
    ].reset_index(drop=True)

    if len(training_df) < 10:
        raise ValueError(
            f"Need at least 10 trainable rows, got {len(training_df)}. "
            "Add more labeled departures or keep more rows before departures."
        )

    for column in FEATURE_COLUMNS:
        if column not in training_df.columns:
            training_df[column] = 0.0
        training_df[column] = pd.to_numeric(training_df[column], errors="coerce")

    return training_df


def split_train_validation(
    df: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if validation_fraction <= 0 or len(df) < 30:
        return df, None

    validation_size = max(1, int(round(len(df) * validation_fraction)))
    if validation_size >= len(df):
        return df, None

    train_df = df.iloc[:-validation_size].reset_index(drop=True)
    valid_df = df.iloc[-validation_size:].reset_index(drop=True)
    return train_df, valid_df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                list(FEATURE_COLUMNS),
            )
        ],
        remainder="drop",
    )


def train_models(
    train_df: pd.DataFrame,
    n_estimators: int,
    random_state: int,
    min_samples_leaf: int,
) -> tuple[Pipeline, Pipeline]:
    regressor = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classifier = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    x_train = train_df.loc[:, FEATURE_COLUMNS]
    y_seconds = train_df["time_until_next_departure"].to_numpy(dtype=float)
    y_bucket = train_df["departure_bucket"].astype(str).to_numpy()

    regressor.fit(x_train, y_seconds)
    classifier.fit(x_train, y_bucket)
    return regressor, classifier


def evaluate_models(
    regressor: Pipeline,
    classifier: Pipeline,
    valid_df: pd.DataFrame | None,
) -> dict[str, float | int | None]:
    if valid_df is None or valid_df.empty:
        return {
            "validation_rows": 0,
            "regression_mae_sec": None,
            "regression_rmse_sec": None,
            "classifier_accuracy": None,
        }

    x_valid = valid_df.loc[:, FEATURE_COLUMNS]
    y_seconds = valid_df["time_until_next_departure"].to_numpy(dtype=float)
    y_bucket = valid_df["departure_bucket"].astype(str).to_numpy()

    predicted_seconds = regressor.predict(x_valid)
    predicted_bucket = classifier.predict(x_valid)
    mse = mean_squared_error(y_seconds, predicted_seconds)
    return {
        "validation_rows": int(len(valid_df)),
        "regression_mae_sec": float(mean_absolute_error(y_seconds, predicted_seconds)),
        "regression_rmse_sec": float(np.sqrt(mse)),
        "classifier_accuracy": float(accuracy_score(y_bucket, predicted_bucket)),
    }


def feature_importance(model: Pipeline) -> list[dict[str, float | str]]:
    forest = model.named_steps["model"]
    importances = getattr(forest, "feature_importances_", None)
    if importances is None:
        return []

    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(FEATURE_COLUMNS, importances)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train RandomForest departure regressor and classifier."
    )
    parser.add_argument("--training-csv", default=None, type=Path)
    parser.add_argument("--training-dir", default=None, type=Path)
    parser.add_argument("--model-output", default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument("--bins", default=DEFAULT_BINS)
    parser.add_argument("--n-estimators", default=300, type=int)
    parser.add_argument("--min-samples-leaf", default=3, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--validation-fraction", default=0.2, type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be greater than zero")
    if args.min_samples_leaf < 1:
        raise ValueError("--min-samples-leaf must be greater than zero")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in [0, 1)")

    parse_probability_bins(args.bins)
    training_dir = args.training_dir
    if args.training_csv is None and training_dir is None:
        training_dir = Path("outputs/training")

    training_paths = collect_training_csvs(args.training_csv, training_dir)
    raw_df = load_training_data(training_paths)
    training_df = prepare_training_data(raw_df)
    train_df, valid_df = split_train_validation(training_df, args.validation_fraction)

    regressor, classifier = train_models(
        train_df=train_df,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        min_samples_leaf=args.min_samples_leaf,
    )
    metrics = evaluate_models(regressor, classifier, valid_df)
    bundle = {
        "version": 1,
        "model_type": "random_forest_departure_bundle",
        "feature_columns": list(FEATURE_COLUMNS),
        "bins": args.bins,
        "bucket_labels": [bucket.label for bucket in parse_probability_bins(args.bins)],
        "regressor": regressor,
        "classifier": classifier,
        "metrics": metrics,
        "training_rows": int(len(training_df)),
        "train_rows": int(len(train_df)),
        "training_sources": [str(path) for path in training_paths],
        "regressor_feature_importance": feature_importance(regressor),
        "classifier_feature_importance": feature_importance(classifier),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)

    metadata_path = args.model_output.with_suffix(".metadata.json")
    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in {"regressor", "classifier"}
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log(f"Training rows: {len(training_df)}")
    log(f"Train rows: {len(train_df)}")
    log(f"Validation rows: {metrics['validation_rows']}")
    log(f"Metrics: {metrics}")
    log(f"Saved model to {args.model_output}")
    log(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
