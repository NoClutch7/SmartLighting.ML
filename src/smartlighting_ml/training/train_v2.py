"""Train the frozen v2 configuration and evaluate a later synthetic holdout once."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from smartlighting_ml.experiments.v2 import (
    FEATURE_SETS,
    SENSORS,
    _contaminated_keys,
    _frequency_scores,
    _labels_for,
    load_development_data,
)
from smartlighting_ml.training.train_isolation_forest import (
    _metric_row,
    _prediction_rows,
    calibrate_threshold,
)


FEATURES = FEATURE_SETS["count_with_temporal"]
MODEL_VERSION = "isolation-forest-v2"


@dataclass(frozen=True)
class V2TrainingConfig:
    development_dir: Path = Path("data/processed/v2_development")
    holdout_dir: Path = Path("data/processed/v2_holdout")
    output_dir: Path = Path("artifacts/isolation_forest_v2")
    demo_dir: Path = Path("data/processed/v2_final")


def train_and_evaluate(config: V2TrainingConfig = V2TrainingConfig()) -> pd.DataFrame:
    development, development_labels = load_development_data(config.development_dir)
    holdout, holdout_labels = load_development_data(config.holdout_dir)
    if development["WindowStartUtc"].max() >= holdout["WindowStartUtc"].min():
        raise ValueError("the holdout must start after the development period")

    times = development["WindowStartUtc"].drop_duplicates().sort_values().array
    calibration_start = times[int(len(times) * 0.8)]
    training = development.loc[development["WindowStartUtc"].lt(calibration_start)]
    calibration = development.loc[development["WindowStartUtc"].ge(calibration_start)]
    contaminated = _contaminated_keys(development_labels)
    training = training.loc[
        [
            (row.SensorId, row.WindowStartUtc) not in contaminated
            for row in training[["SensorId", "WindowStartUtc"]].itertuples(index=False)
        ]
    ]

    models_dir = config.output_dir / "models"
    results_dir = config.output_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    config.demo_dir.mkdir(parents=True, exist_ok=True)

    thresholds: dict[str, dict[str, float]] = {}
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for sensor in SENSORS:
        sensor_train = training.loc[training["SensorId"].eq(sensor)]
        sensor_calibration = calibration.loc[calibration["SensorId"].eq(sensor)]
        sensor_holdout = holdout.loc[holdout["SensorId"].eq(sensor)]
        calibration_truth = _labels_for(sensor_calibration, development_labels)
        holdout_truth = _labels_for(sensor_holdout, holdout_labels)
        if min(len(sensor_train), len(sensor_calibration), len(sensor_holdout)) == 0:
            raise ValueError(f"empty training, calibration, or holdout for {sensor}")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "isolation_forest",
                    IsolationForest(
                        n_estimators=100,
                        max_samples=128,
                        max_features=1.0,
                        contamination="auto",
                        random_state=20261008,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(sensor_train[FEATURES])
        calibration_scores = -model.score_samples(sensor_calibration[FEATURES])
        threshold = calibrate_threshold(calibration_scores, calibration_truth)
        holdout_scores = -model.score_samples(sensor_holdout[FEATURES])
        predicted = holdout_scores >= threshold
        joblib.dump(model, models_dir / f"{sensor}.joblib")

        calibration_frequency = _frequency_scores(sensor_train, sensor_calibration)
        frequency_threshold = calibrate_threshold(
            calibration_frequency, calibration_truth
        )
        holdout_frequency = _frequency_scores(sensor_train, sensor_holdout)
        frequency_predicted = holdout_frequency >= frequency_threshold

        thresholds[sensor] = {
            "isolation_forest": threshold,
            "statistical_frequency": frequency_threshold,
        }
        for detector, scores, selected_threshold, selected in (
            ("isolation_forest", holdout_scores, threshold, predicted),
            (
                "statistical_frequency",
                holdout_frequency,
                frequency_threshold,
                frequency_predicted,
            ),
        ):
            rows.append(_metric_row(detector, sensor, holdout_truth, selected))
            predictions.append(
                _prediction_rows(
                    sensor_holdout,
                    holdout_truth,
                    detector,
                    scores,
                    selected_threshold,
                    MODEL_VERSION,
                )
            )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    for detector in ("isolation_forest", "statistical_frequency"):
        selected = prediction_frame.loc[prediction_frame["detector"].eq(detector)]
        rows.append(
            _metric_row(
                detector,
                "TOTAL",
                selected["is_known_anomaly"].to_numpy(dtype=bool),
                selected["is_anomaly"].to_numpy(dtype=bool),
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "test_metrics.csv", index=False)
    prediction_frame.to_csv(results_dir / "test_predictions.csv", index=False)
    holdout.to_csv(config.demo_dir / "test.csv", index=False)
    summary = {
        "trainingWindows": len(training),
        "validationWindows": len(calibration),
        "testWindows": len(holdout),
        "sensorCount": len(SENSORS),
    }
    (results_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "feature_names": FEATURES,
        "thresholds": thresholds,
        "configuration": "100 trees, 128 samples, count_with_temporal; chosen on v2 development folds",
        "calibration_start_utc": calibration_start.isoformat(),
        "holdout_start_utc": holdout["WindowStartUtc"].min().isoformat(),
        "holdout_end_utc": holdout["WindowStartUtc"].max().isoformat(),
        "source": "synthetic_v2_holdout",
        "limitations": [
            "Development and holdout data are synthetic and follow related generator assumptions.",
            "The results do not establish reliability on physical sensors.",
        ],
    }
    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, default=V2TrainingConfig.development_dir)
    parser.add_argument("--holdout-dir", type=Path, default=V2TrainingConfig.holdout_dir)
    parser.add_argument("--output-dir", type=Path, default=V2TrainingConfig.output_dir)
    parser.add_argument("--demo-dir", type=Path, default=V2TrainingConfig.demo_dir)
    args = parser.parse_args()
    metrics = train_and_evaluate(
        V2TrainingConfig(args.development_dir, args.holdout_dir, args.output_dir, args.demo_dir)
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
