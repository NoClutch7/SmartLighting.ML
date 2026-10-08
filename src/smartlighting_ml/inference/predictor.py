"""Load sensor-specific artifacts and score one processed five-minute window."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class AnomalyPredictor:
    def __init__(self, artifact_dir: Path = Path("artifacts/isolation_forest_v1")):
        self.artifact_dir = artifact_dir
        self.metadata = json.loads(
            (artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.feature_names: list[str] = self.metadata["feature_names"]
        self.thresholds: dict[str, dict[str, float]] = self.metadata["thresholds"]
        self.model_version: str = self.metadata["model_version"]
        self._models: dict[str, Any] = {}

    def _model_for(self, sensor_id: str) -> Any:
        if sensor_id not in self.thresholds:
            raise ValueError(f"no trained model for SensorId={sensor_id}")
        if sensor_id not in self._models:
            self._models[sensor_id] = joblib.load(
                self.artifact_dir / "models" / f"{sensor_id}.joblib"
            )
        return self._models[sensor_id]

    def predict(self, window: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        sensor_id = str(window["SensorId"])
        timestamp = pd.to_datetime(window["WindowStartUtc"], utc=True, errors="coerce")
        if pd.isna(timestamp):
            raise ValueError("WindowStartUtc must be a valid timestamp")
        try:
            values = np.asarray([window[name] for name in self.feature_names], dtype=float)
        except KeyError as error:
            raise ValueError(f"missing feature: {error.args[0]}") from error
        if not np.isfinite(values).all():
            raise ValueError("all feature values must be finite")

        model = self._model_for(sensor_id)
        frame = pd.DataFrame([values], columns=self.feature_names)
        anomaly_score = float(-model.score_samples(frame)[0])
        threshold = float(self.thresholds[sensor_id]["isolation_forest"])
        return {
            "SensorId": sensor_id,
            "WindowStartUtc": timestamp.isoformat(),
            "anomaly_score": anomaly_score,
            "is_anomaly": bool(anomaly_score >= threshold),
            "model_version": self.model_version,
        }
