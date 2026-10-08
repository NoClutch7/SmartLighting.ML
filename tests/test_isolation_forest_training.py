from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from smartlighting_ml.inference.predictor import AnomalyPredictor
from smartlighting_ml.training.train_isolation_forest import (
    EXPECTED_SENSORS,
    FEATURE_COLUMNS,
    TrainingConfig,
    calibrate_threshold,
    load_training_inputs,
    train_and_evaluate,
)


PROCESSED_DIR = Path("data/processed")


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("trained-artifacts")
    config = TrainingConfig(
        processed_dir=PROCESSED_DIR,
        artifact_dir=artifact_dir,
        n_estimators=30,
    )
    metrics = train_and_evaluate(config)
    return artifact_dir, metrics, config


def test_six_independent_models_and_feature_order_are_saved(trained_artifacts) -> None:
    artifact_dir, _, config = trained_artifacts
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["feature_names"] == FEATURE_COLUMNS
    assert metadata["isolation_forest_hyperparameters"]["random_state"] == 20261008
    assert metadata["isolation_forest_hyperparameters"]["n_estimators"] == 30
    assert sorted(metadata["thresholds"]) == EXPECTED_SENSORS
    assert metadata["data_audit"]["known_anomaly_or_history_overlap_in_training"] == 0

    models = []
    for sensor_id in EXPECTED_SENSORS:
        path = artifact_dir / "models" / f"{sensor_id}.joblib"
        assert path.exists()
        model = joblib.load(path)
        assert model.named_steps["isolation_forest"].n_features_in_ == len(FEATURE_COLUMNS)
        models.append(model)
    assert len({id(model) for model in models}) == 6
    assert config.model_version == metadata["model_version"]


def test_thresholds_are_reproducibly_calibrated_from_validation(trained_artifacts) -> None:
    artifact_dir, _, _ = trained_artifacts
    predictions = pd.read_csv(artifact_dir / "results" / "validation_predictions.csv")
    thresholds = json.loads((artifact_dir / "thresholds.json").read_text(encoding="utf-8"))

    for sensor_id in EXPECTED_SENSORS:
        for detector in ("isolation_forest", "statistical_frequency"):
            rows = predictions.loc[
                predictions["SensorId"].eq(sensor_id)
                & predictions["detector"].eq(detector)
            ]
            expected = calibrate_threshold(
                rows["anomaly_score"].to_numpy(),
                rows["is_known_anomaly"].to_numpy(dtype=bool),
            )
            assert thresholds[sensor_id][detector] == pytest.approx(expected)


def test_test_metrics_match_absolute_confusion_counts(trained_artifacts) -> None:
    artifact_dir, metrics, _ = trained_artifacts
    predictions = pd.read_csv(artifact_dir / "results" / "test_predictions.csv")

    assert set(metrics["SensorId"]) == {*EXPECTED_SENSORS, "TOTAL"}
    for row in metrics.itertuples(index=False):
        selected = predictions.loc[predictions["detector"].eq(row.detector)]
        if row.SensorId != "TOTAL":
            selected = selected.loc[selected["SensorId"].eq(row.SensorId)]
        assert row.TP + row.FP + row.TN + row.FN == len(selected)
        assert row.TP + row.FN == int(selected["is_known_anomaly"].sum())
        assert row.false_positives == row.FP
        assert row.positive_windows in (2, 3, 16)


def test_inference_returns_contract_and_uses_sensor_specific_model(
    trained_artifacts,
) -> None:
    artifact_dir, _, _ = trained_artifacts
    predictor = AnomalyPredictor(artifact_dir)
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    outputs = []
    for sensor_id in EXPECTED_SENSORS:
        window = test.loc[test["SensorId"].eq(sensor_id)].iloc[0]
        output = predictor.predict(window)
        assert set(output) == {
            "SensorId",
            "WindowStartUtc",
            "anomaly_score",
            "is_anomaly",
            "model_version",
        }
        assert output["SensorId"] == sensor_id
        assert np.isfinite(output["anomaly_score"])
        assert isinstance(output["is_anomaly"], bool)
        outputs.append(output)
    assert set(predictor._models) == set(EXPECTED_SENSORS)
    assert len(outputs) == 6


def test_training_is_reproducible(tmp_path, trained_artifacts) -> None:
    first_dir, first_metrics, first_config = trained_artifacts
    second_dir = tmp_path / "second"
    second_metrics = train_and_evaluate(
        TrainingConfig(
            processed_dir=PROCESSED_DIR,
            artifact_dir=second_dir,
            n_estimators=first_config.n_estimators,
        )
    )

    pd.testing.assert_frame_equal(first_metrics, second_metrics)
    first_predictions = pd.read_csv(first_dir / "results" / "test_predictions.csv")
    second_predictions = pd.read_csv(second_dir / "results" / "test_predictions.csv")
    pd.testing.assert_frame_equal(first_predictions, second_predictions)


def test_processed_inputs_remain_label_free_and_chronological() -> None:
    train, validation, test, _ = load_training_inputs(PROCESSED_DIR)

    for frame in (train, validation, test):
        assert "AnomalyType" not in frame.columns
        assert "is_anomaly" not in frame.columns
        assert np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    assert train["WindowStartUtc"].max() < validation["WindowStartUtc"].min()
    assert validation["WindowStartUtc"].max() < test["WindowStartUtc"].min()
