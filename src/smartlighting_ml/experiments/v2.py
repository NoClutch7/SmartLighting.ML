"""Leakage-safe development experiments for SmartLighting.ML version 2."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from smartlighting_ml.training.train_isolation_forest import calibrate_threshold


ORIGINAL_FEATURES = [
    "event_count_5m",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "events_last_15m",
    "events_last_60m",
    "rolling_mean_1h",
]
FEATURE_SETS = {
    "original_9": ORIGINAL_FEATURES,
    "without_rolling_mean_1h": [
        feature for feature in ORIGINAL_FEATURES if feature != "rolling_mean_1h"
    ],
    "count_with_history": [
        "event_count_5m",
        "events_last_15m",
        "events_last_60m",
    ],
    "count_with_temporal": [
        "event_count_5m",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
    ],
}
ANOMALY_TYPES = ("out_of_pattern_burst", "electrical_oscillation")
SENSORS = tuple(f"PIR-{index:02d}" for index in range(1, 7))


@dataclass(frozen=True)
class HyperParameters:
    n_estimators: int
    max_samples: int | str
    max_features: float

    @property
    def identifier(self) -> str:
        return (
            f"trees-{self.n_estimators}_samples-{self.max_samples}_"
            f"features-{self.max_features:g}"
        )


@dataclass(frozen=True)
class V2Config:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("artifacts/experiments_v2")
    random_state: int = 20261008
    initial_train_fraction: float = 0.40
    calibration_fraction: float = 0.20
    evaluation_fraction: float = 0.20
    step_fraction: float = 0.10
    minimum_positive_episodes: int = 10


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


def default_hyperparameter_grid() -> list[HyperParameters]:
    return [
        HyperParameters(trees, samples, features)
        for trees, samples, features in itertools.product(
            (100, 300), (128, "auto"), (0.5, 1.0)
        )
    ]


def load_development_data(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only development periods; the v1 test has no code path here."""

    development_path = processed_dir / "development_windows.csv"
    legacy_path = processed_dir / "all_windows.csv"
    windows = pd.read_csv(
        development_path if development_path.exists() else legacy_path
    )
    labels = pd.read_csv(processed_dir / "anomaly_windows_evaluation.csv")
    required = {
        "SensorId",
        "WindowStartUtc",
        "split",
        "telemetry_reliable",
        *ORIGINAL_FEATURES,
    }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"all_windows.csv is missing: {', '.join(missing)}")
    windows["WindowStartUtc"] = pd.to_datetime(
        windows["WindowStartUtc"], utc=True, errors="coerce"
    )
    labels["WindowStartUtc"] = pd.to_datetime(
        labels["WindowStartUtc"], utc=True, errors="coerce"
    )
    if windows["WindowStartUtc"].isna().any() or labels["WindowStartUtc"].isna().any():
        raise ValueError("development inputs contain invalid timestamps")

    if development_path.exists():
        development = windows.copy()
    else:
        development = windows.loc[
            windows["split"].isin(["train", "validation"])
        ].copy()
    development = development.loc[
        development["telemetry_reliable"].astype(bool)
        & development[ORIGINAL_FEATURES].notna().all(axis=1)
    ]
    values = development[ORIGINAL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("development features must be finite")
    if sorted(development["SensorId"].unique()) != list(SENSORS):
        raise ValueError("development data must contain the six expected sensors")

    development_end = development["WindowStartUtc"].max()
    labels = labels.loc[labels["WindowStartUtc"].le(development_end)].copy()
    return development.sort_values(["WindowStartUtc", "SensorId"]), labels


def build_progressive_folds(
    development: pd.DataFrame, config: V2Config
) -> list[TemporalFold]:
    times = development["WindowStartUtc"].drop_duplicates().sort_values().array
    total = len(times)
    train_size = int(total * config.initial_train_fraction)
    calibration_size = int(total * config.calibration_fraction)
    evaluation_size = int(total * config.evaluation_fraction)
    step_size = max(1, int(total * config.step_fraction))
    if min(train_size, calibration_size, evaluation_size) <= 0:
        raise ValueError("temporal fractions produce an empty block")

    folds: list[TemporalFold] = []
    fold_number = 0
    while train_size + calibration_size + evaluation_size <= total:
        calibration_end_index = train_size + calibration_size
        evaluation_end_index = calibration_end_index + evaluation_size
        evaluation_end = (
            times[evaluation_end_index]
            if evaluation_end_index < total
            else times[-1] + pd.Timedelta(minutes=5)
        )
        folds.append(
            TemporalFold(
                fold=fold_number,
                train_start=times[0],
                train_end=times[train_size],
                calibration_start=times[train_size],
                calibration_end=times[calibration_end_index],
                evaluation_start=times[calibration_end_index],
                evaluation_end=evaluation_end,
            )
        )
        fold_number += 1
        train_size += step_size
    return folds


def _positive_keys(labels: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return set(
        labels[["SensorId", "WindowStartUtc"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )


def _contaminated_keys(labels: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    contaminated: set[tuple[str, pd.Timestamp]] = set()
    step = pd.Timedelta(minutes=5)
    for sensor, timestamp in _positive_keys(labels):
        for offset in range(13):
            contaminated.add((sensor, timestamp + offset * step))
    return contaminated


def _labels_for(
    frame: pd.DataFrame, labels: pd.DataFrame, anomaly_type: str | None = None
) -> np.ndarray:
    selected = labels
    if anomaly_type is not None:
        selected = selected.loc[selected["AnomalyType"].eq(anomaly_type)]
    keys = _positive_keys(selected)
    return np.asarray(
        [key in keys for key in frame[["SensorId", "WindowStartUtc"]].itertuples(index=False, name=None)],
        dtype=bool,
    )


def _episode_count(frame: pd.DataFrame, labels: pd.DataFrame) -> int:
    keys = frame[["SensorId", "WindowStartUtc"]].drop_duplicates()
    selected = labels.merge(keys, on=["SensorId", "WindowStartUtc"], how="inner")
    if "EpisodeIdentifier" in selected.columns:
        return int(selected["EpisodeIdentifier"].nunique())
    return len(selected[["SensorId", "WindowStartUtc"]].drop_duplicates())


def _block(
    development: pd.DataFrame,
    sensor: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return development.loc[
        development["SensorId"].eq(sensor)
        & development["WindowStartUtc"].ge(start)
        & development["WindowStartUtc"].lt(end)
    ].copy()


def _metric_rows(
    base: dict[str, Any],
    evaluation: pd.DataFrame,
    predictions: np.ndarray,
    labels: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_positive = _labels_for(evaluation, labels)
    for anomaly_type in ("overall", *ANOMALY_TYPES):
        if anomaly_type == "overall":
            considered = np.ones(len(evaluation), dtype=bool)
            truth = all_positive
        else:
            type_truth = _labels_for(evaluation, labels, anomaly_type)
            # Exclude positives of the other synthetic type from type-specific metrics.
            considered = type_truth | ~all_positive
            truth = type_truth[considered]
        predicted = predictions[considered]
        tn, fp, fn, tp = confusion_matrix(
            truth, predicted, labels=[False, True]
        ).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        considered_frame = evaluation.loc[considered]
        normal_zero = (~truth) & considered_frame["event_count_5m"].eq(0).to_numpy()
        zero_false_positives = int((predicted & normal_zero).sum())
        row = {
            **base,
            "anomaly_type": anomaly_type,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "positive_windows": int(truth.sum()),
            "negative_windows": int((~truth).sum()),
            "false_positives_zero_count": zero_false_positives,
            "normal_zero_count_windows": int(normal_zero.sum()),
            "false_positive_rate_zero_count": (
                zero_false_positives / normal_zero.sum() if normal_zero.sum() else 0.0
            ),
        }
        rows.append(row)
    return rows


def _frequency_scores(training: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    mean = float(training["event_count_5m"].mean())
    std = float(training["event_count_5m"].std(ddof=0))
    return (target["event_count_5m"].to_numpy(dtype=float) - mean) / (std or 1.0)


def run_experiments(
    config: V2Config = V2Config(),
    feature_sets: dict[str, list[str]] | None = None,
    hyperparameters: list[HyperParameters] | None = None,
) -> pd.DataFrame:
    feature_sets = feature_sets or FEATURE_SETS
    hyperparameters = hyperparameters or default_hyperparameter_grid()
    development, labels = load_development_data(config.processed_dir)
    folds = build_progressive_folds(development, config)
    contaminated = _contaminated_keys(labels)
    rows: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []

    for fold in folds:
        for sensor in SENSORS:
            training = _block(
                development, sensor, fold.train_start, fold.train_end
            )
            training_keys = training[["SensorId", "WindowStartUtc"]].itertuples(
                index=False, name=None
            )
            clean_mask = np.asarray(
                [key not in contaminated for key in training_keys], dtype=bool
            )
            training = training.loc[clean_mask]
            calibration = _block(
                development, sensor, fold.calibration_start, fold.calibration_end
            )
            evaluation = _block(
                development, sensor, fold.evaluation_start, fold.evaluation_end
            )
            calibration_labels = _labels_for(calibration, labels)
            evaluation_labels = _labels_for(evaluation, labels)
            calibration_episodes = _episode_count(calibration, labels)
            evaluation_episodes = _episode_count(evaluation, labels)
            sufficient = (
                calibration_episodes >= config.minimum_positive_episodes
                and evaluation_episodes >= config.minimum_positive_episodes
            )
            if not sufficient:
                limitations.append(
                    {
                        "fold": fold.fold,
                        "SensorId": sensor,
                        "calibration_positives": int(calibration_labels.sum()),
                        "evaluation_positives": int(evaluation_labels.sum()),
                        "calibration_episodes": calibration_episodes,
                        "evaluation_episodes": evaluation_episodes,
                        "reason": "insufficient positive windows for stable comparison",
                    }
                )
            if not calibration_labels.any() or evaluation.empty or training.empty:
                continue

            # Original statistical detector, once per fold and sensor.
            calibration_frequency = _frequency_scores(training, calibration)
            frequency_threshold = calibrate_threshold(
                calibration_frequency, calibration_labels
            )
            evaluation_frequency = _frequency_scores(training, evaluation)
            rows.extend(
                _metric_rows(
                    {
                        "fold": fold.fold,
                        "SensorId": sensor,
                        "detector": "statistical_frequency",
                        "feature_set": "event_count_only",
                        "configuration": "original_statistical",
                        "threshold": frequency_threshold,
                        "calibration_positive_windows": int(calibration_labels.sum()),
                        "evaluation_positive_windows": int(evaluation_labels.sum()),
                        "calibration_positive_episodes": calibration_episodes,
                        "evaluation_positive_episodes": evaluation_episodes,
                        "sufficient_for_comparison": sufficient,
                    },
                    evaluation,
                    evaluation_frequency >= frequency_threshold,
                    labels,
                )
            )

            for feature_name, features in feature_sets.items():
                for parameters in hyperparameters:
                    model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "isolation_forest",
                                IsolationForest(
                                    n_estimators=parameters.n_estimators,
                                    max_samples=parameters.max_samples,
                                    max_features=parameters.max_features,
                                    contamination="auto",
                                    random_state=config.random_state,
                                    n_jobs=1,
                                ),
                            ),
                        ]
                    )
                    model.fit(training[features])
                    calibration_scores = -model.score_samples(calibration[features])
                    threshold = calibrate_threshold(
                        calibration_scores, calibration_labels
                    )
                    evaluation_scores = -model.score_samples(evaluation[features])
                    rows.extend(
                        _metric_rows(
                            {
                                "fold": fold.fold,
                                "SensorId": sensor,
                                "detector": "isolation_forest",
                                "feature_set": feature_name,
                                "configuration": parameters.identifier,
                                "threshold": threshold,
                                "calibration_positive_windows": int(
                                    calibration_labels.sum()
                                ),
                                "evaluation_positive_windows": int(
                                    evaluation_labels.sum()
                                ),
                                "calibration_positive_episodes": calibration_episodes,
                                "evaluation_positive_episodes": evaluation_episodes,
                                "sufficient_for_comparison": sufficient,
                            },
                            evaluation,
                            evaluation_scores >= threshold,
                            labels,
                        )
                    )

    metrics = pd.DataFrame(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.output_dir / "development_fold_metrics.csv", index=False)
    report = {
        "status": (
            "insufficient_positive_windows"
            if limitations
            else "ready_for_development_comparison"
        ),
        "test_data_accessed": False,
        "selection_permitted": "development folds only",
        "feature_sets": feature_sets,
        "hyperparameter_grid": [asdict(item) for item in hyperparameters],
        "config": {
            **asdict(config),
            "processed_dir": str(config.processed_dir),
            "output_dir": str(config.output_dir),
        },
        "folds": [
            {key: (value.isoformat() if isinstance(value, pd.Timestamp) else value)
             for key, value in asdict(fold).items()}
            for fold in folds
        ],
        "limitations": limitations,
        "interpretation": (
            "No configuration may be declared superior when folds have insufficient "
            "positive windows. The original v1 test remains excluded."
        ),
    }
    (config.output_dir / "development_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments_v2")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one small configuration to validate the experimental workflow",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grid = [HyperParameters(10, 128, 1.0)] if args.quick else None
    selected_feature_sets = {"original_9": FEATURE_SETS["original_9"]} if args.quick else None
    metrics = run_experiments(
        V2Config(processed_dir=args.processed_dir, output_dir=args.output_dir),
        feature_sets=selected_feature_sets,
        hyperparameters=grid,
    )
    eligible = metrics.loc[
        metrics["sufficient_for_comparison"] & metrics["anomaly_type"].eq("overall")
    ]
    print(f"metric_rows={len(metrics)}")
    print(f"eligible_overall_rows={len(eligible)}")
    print("No v1 test data was accessed.")


if __name__ == "__main__":
    main()
