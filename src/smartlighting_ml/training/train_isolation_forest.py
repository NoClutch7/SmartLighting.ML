"""Train, calibrate, and evaluate one Isolation Forest per sensor."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
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
KEY_COLUMNS = ["SensorId", "WindowStartUtc"]
EXPECTED_SENSORS = [f"SENSOR-{index:02d}" for index in range(1, 7)]
MODEL_VERSION = "isolation-forest-v1"


@dataclass(frozen=True)
class TrainingConfig:
    processed_dir: Path = Path("data/processed")
    artifact_dir: Path = Path("artifacts/isolation_forest_v1")
    random_state: int = 20261008
    n_estimators: int = 300
    max_samples: str = "auto"
    contamination: str = "auto"
    model_version: str = MODEL_VERSION


@dataclass(frozen=True)
class StatisticalDetector:
    mean: float
    standard_deviation: float

    def score(self, values: pd.Series | np.ndarray) -> np.ndarray:
        scale = self.standard_deviation if self.standard_deviation > 0 else 1.0
        return (np.asarray(values, dtype=float) - self.mean) / scale


def _load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["SensorId", "DeviceId", "WindowStartUtc", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    frame["WindowStartUtc"] = pd.to_datetime(
        frame["WindowStartUtc"], utc=True, errors="coerce"
    )
    if frame["WindowStartUtc"].isna().any():
        raise ValueError(f"{path.name} contains invalid WindowStartUtc values")
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError(f"{path.name} contains missing or infinite feature values")
    if "AnomalyType" in frame.columns or "is_anomaly" in frame.columns:
        raise ValueError(f"{path.name} contains an anomaly label column")
    return frame.sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_training_inputs(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _load_dataset(processed_dir / "train.csv")
    validation = _load_dataset(processed_dir / "validation.csv")
    test = _load_dataset(processed_dir / "test.csv")
    anomaly_windows = pd.read_csv(processed_dir / "anomaly_windows_evaluation.csv")
    required_labels = ["EventIdentifier", "SensorId", "WindowStartUtc", "AnomalyType"]
    missing = [column for column in required_labels if column not in anomaly_windows]
    if missing:
        raise ValueError(f"anomaly evaluation file is missing: {', '.join(missing)}")
    anomaly_windows["WindowStartUtc"] = pd.to_datetime(
        anomaly_windows["WindowStartUtc"], utc=True, errors="coerce"
    )
    if anomaly_windows["WindowStartUtc"].isna().any():
        raise ValueError("anomaly evaluation file contains invalid timestamps")
    return train, validation, test, anomaly_windows


def audit_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    anomaly_windows: pd.DataFrame,
) -> dict[str, Any]:
    """Fail before training if chronology, sensors, or contamination is invalid."""

    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        sensors = sorted(frame["SensorId"].unique().tolist())
        if sensors != EXPECTED_SENSORS:
            raise ValueError(f"{name} does not contain exactly the six expected sensors")

    if not train["WindowStartUtc"].max() < validation["WindowStartUtc"].min():
        raise ValueError("training and validation periods overlap")
    if not validation["WindowStartUtc"].max() < test["WindowStartUtc"].min():
        raise ValueError("validation and test periods overlap")

    positive_keys = set(
        anomaly_windows[KEY_COLUMNS].drop_duplicates().itertuples(index=False, name=None)
    )
    contaminated_keys: set[tuple[str, pd.Timestamp]] = set()
    step = pd.Timedelta(minutes=5)
    for sensor_id, window_start in positive_keys:
        for offset in range(13):
            contaminated_keys.add((sensor_id, window_start + offset * step))
    train_keys = set(train[KEY_COLUMNS].itertuples(index=False, name=None))
    overlap = train_keys & contaminated_keys
    if overlap:
        raise ValueError(
            "training contains known anomalies or windows contaminated by their history"
        )

    positive_counts: dict[str, dict[str, int]] = {}
    for name, frame in (("validation", validation), ("test", test)):
        frame_keys = set(frame[KEY_COLUMNS].itertuples(index=False, name=None))
        per_sensor = {
            sensor: len(
                {
                    key for key in positive_keys if key[0] == sensor and key in frame_keys
                }
            )
            for sensor in EXPECTED_SENSORS
        }
        if any(count == 0 for count in per_sensor.values()):
            raise ValueError(f"{name} lacks positive windows for at least one sensor")
        positive_counts[name] = per_sensor

    correlation = train[FEATURE_COLUMNS].corr().abs()
    redundant_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(FEATURE_COLUMNS):
        for right in FEATURE_COLUMNS[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            if value >= 0.999:
                redundant_pairs.append(
                    {"feature_a": left, "feature_b": right, "absolute_correlation": value}
                )
    scales = {
        feature: {
            "min": float(train[feature].min()),
            "max": float(train[feature].max()),
            "mean": float(train[feature].mean()),
            "standard_deviation": float(train[feature].std(ddof=0)),
        }
        for feature in FEATURE_COLUMNS
    }
    return {
        "periods": {
            name: {
                "start_utc": frame["WindowStartUtc"].min().isoformat(),
                "end_utc": frame["WindowStartUtc"].max().isoformat(),
                "rows": len(frame),
            }
            for name, frame in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "positive_windows": positive_counts,
        "feature_scales_training": scales,
        "redundant_feature_pairs": redundant_pairs,
        "known_anomaly_or_history_overlap_in_training": 0,
    }


def _labels_for(frame: pd.DataFrame, anomaly_windows: pd.DataFrame) -> np.ndarray:
    positive_keys = set(
        anomaly_windows[KEY_COLUMNS].drop_duplicates().itertuples(index=False, name=None)
    )
    return np.asarray(
        [key in positive_keys for key in frame[KEY_COLUMNS].itertuples(index=False, name=None)],
        dtype=bool,
    )


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Choose a validation-only threshold by F1, then recall, precision, specificity."""

    if scores.size == 0 or labels.size != scores.size:
        raise ValueError("scores and labels must be non-empty and aligned")
    if not labels.any():
        raise ValueError("threshold calibration requires positive validation windows")
    candidates = np.unique(scores)
    candidates = np.append(candidates, np.nextafter(candidates.max(), np.inf))
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    positive_suffix = np.zeros(scores.size + 1, dtype=np.int64)
    positive_suffix[:-1] = np.cumsum(sorted_labels[::-1])[::-1]
    total_positive = int(sorted_labels.sum())
    total_negative = int(scores.size - total_positive)
    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for threshold in candidates:
        first_predicted = int(np.searchsorted(sorted_scores, threshold, side="left"))
        predicted_count = scores.size - first_predicted
        tp = int(positive_suffix[first_predicted])
        fp = int(predicted_count - tp)
        fn = total_positive - tp
        tn = total_negative - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        specificity = tn / (tn + fp) if tn + fp else 0.0
        key = (float(f1), float(recall), float(precision), float(specificity))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _isolation_score(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    # score_samples is larger for normal observations; invert for alert semantics.
    return -model.score_samples(frame[FEATURE_COLUMNS])


def _metric_row(
    detector: str,
    sensor_id: str,
    labels: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "detector": detector,
        "SensorId": sensor_id,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positives": int(fp),
        "positive_windows": int(labels.sum()),
        "evaluated_windows": int(labels.size),
    }


def _prediction_rows(
    frame: pd.DataFrame,
    labels: np.ndarray,
    detector: str,
    scores: np.ndarray,
    threshold: float,
    model_version: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SensorId": frame["SensorId"].to_numpy(),
            "WindowStartUtc": frame["WindowStartUtc"].map(lambda value: value.isoformat()),
            "detector": detector,
            "anomaly_score": scores,
            "threshold": threshold,
            "is_anomaly": scores >= threshold,
            "is_known_anomaly": labels,
            "model_version": model_version,
        }
    )


def train_and_evaluate(config: TrainingConfig = TrainingConfig()) -> pd.DataFrame:
    train, validation, test, anomaly_windows = load_training_inputs(
        config.processed_dir
    )
    audit = audit_inputs(train, validation, test, anomaly_windows)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    models_dir = config.artifact_dir / "models"
    results_dir = config.artifact_dir / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    thresholds: dict[str, dict[str, float]] = {}
    statistical_parameters: dict[str, dict[str, float]] = {}
    validation_predictions: list[pd.DataFrame] = []
    test_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for sensor_id in EXPECTED_SENSORS:
        sensor_train = train.loc[train["SensorId"].eq(sensor_id)].copy()
        sensor_validation = validation.loc[validation["SensorId"].eq(sensor_id)].copy()
        sensor_test = test.loc[test["SensorId"].eq(sensor_id)].copy()
        validation_labels = _labels_for(sensor_validation, anomaly_windows)
        test_labels = _labels_for(sensor_test, anomaly_windows)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "isolation_forest",
                    IsolationForest(
                        n_estimators=config.n_estimators,
                        max_samples=config.max_samples,
                        contamination=config.contamination,
                        random_state=config.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(sensor_train[FEATURE_COLUMNS])
        validation_if_scores = _isolation_score(model, sensor_validation)
        if_threshold = calibrate_threshold(validation_if_scores, validation_labels)
        test_if_scores = _isolation_score(model, sensor_test)
        test_if_predicted = test_if_scores >= if_threshold
        joblib.dump(model, models_dir / f"{sensor_id}.joblib")

        frequency_mean = float(sensor_train["event_count_5m"].mean())
        frequency_std = float(sensor_train["event_count_5m"].std(ddof=0))
        statistical = StatisticalDetector(frequency_mean, frequency_std)
        validation_stat_scores = statistical.score(
            sensor_validation["event_count_5m"]
        )
        stat_threshold = calibrate_threshold(validation_stat_scores, validation_labels)
        test_stat_scores = statistical.score(sensor_test["event_count_5m"])
        test_stat_predicted = test_stat_scores >= stat_threshold

        thresholds[sensor_id] = {
            "isolation_forest": if_threshold,
            "statistical_frequency": stat_threshold,
        }
        statistical_parameters[sensor_id] = {
            "training_mean": frequency_mean,
            "training_standard_deviation": frequency_std,
        }
        validation_predictions.extend(
            [
                _prediction_rows(
                    sensor_validation,
                    validation_labels,
                    "isolation_forest",
                    validation_if_scores,
                    if_threshold,
                    config.model_version,
                ),
                _prediction_rows(
                    sensor_validation,
                    validation_labels,
                    "statistical_frequency",
                    validation_stat_scores,
                    stat_threshold,
                    config.model_version,
                ),
            ]
        )
        test_predictions.extend(
            [
                _prediction_rows(
                    sensor_test,
                    test_labels,
                    "isolation_forest",
                    test_if_scores,
                    if_threshold,
                    config.model_version,
                ),
                _prediction_rows(
                    sensor_test,
                    test_labels,
                    "statistical_frequency",
                    test_stat_scores,
                    stat_threshold,
                    config.model_version,
                ),
            ]
        )
        metric_rows.extend(
            [
                _metric_row(
                    "isolation_forest", sensor_id, test_labels, test_if_predicted
                ),
                _metric_row(
                    "statistical_frequency",
                    sensor_id,
                    test_labels,
                    test_stat_predicted,
                ),
            ]
        )

    validation_frame = pd.concat(validation_predictions, ignore_index=True)
    test_frame = pd.concat(test_predictions, ignore_index=True)
    for detector in ("isolation_forest", "statistical_frequency"):
        detector_test = test_frame.loc[test_frame["detector"].eq(detector)]
        metric_rows.append(
            _metric_row(
                detector,
                "TOTAL",
                detector_test["is_known_anomaly"].to_numpy(dtype=bool),
                detector_test["is_anomaly"].to_numpy(dtype=bool),
            )
        )
    metrics = pd.DataFrame(metric_rows)

    validation_frame.to_csv(results_dir / "validation_predictions.csv", index=False)
    test_frame.to_csv(results_dir / "test_predictions.csv", index=False)
    metrics.to_csv(results_dir / "test_metrics.csv", index=False)
    metrics_payload = {
        "model_version": config.model_version,
        "warning": (
            "Metrics have high uncertainty because each sensor has only two or "
            "three positive test windows. An anomaly is not evidence of intrusion."
        ),
        "metrics": metrics.to_dict(orient="records"),
    }
    (results_dir / "test_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata = {
        "model_version": config.model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_COLUMNS,
        "feature_order_is_required": True,
        "training_config": {
            **asdict(config),
            "processed_dir": str(config.processed_dir),
            "artifact_dir": str(config.artifact_dir),
        },
        "isolation_forest_hyperparameters": {
            "n_estimators": config.n_estimators,
            "max_samples": config.max_samples,
            "contamination": config.contamination,
            "random_state": config.random_state,
            "n_jobs": 1,
        },
        "preprocessing": "StandardScaler fitted independently per sensor",
        "threshold_calibration": (
            "Validation-only F1 maximization; ties resolved by recall, precision, "
            "and specificity. Test was evaluated only after calibration."
        ),
        "statistical_detector": (
            "Upper-tail standardized event_count_5m using training mean/std; "
            "threshold calibrated on validation only."
        ),
        "thresholds": thresholds,
        "statistical_parameters": statistical_parameters,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "data_audit": audit,
        "limitations": [
            "Training and evaluation use synthetic sensor pulses, not people counts.",
            "Only two or three positive validation and test windows exist per sensor.",
            "Telemetry outages are excluded upstream and must never be treated as zero.",
            "An anomaly does not imply an intrusion, threat, or security incident.",
            "events_last_60m and rolling_mean_1h are exactly linearly redundant, but "
            "both are retained to honor the nine-feature experiment specification.",
        ],
    }
    (config.artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (config.artifact_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/isolation_forest_v1")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = train_and_evaluate(
        TrainingConfig(
            processed_dir=args.processed_dir,
            artifact_dir=args.artifact_dir,
        )
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
