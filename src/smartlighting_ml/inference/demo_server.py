"""Serve model predictions for clearly identified synthetic test windows."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from smartlighting_ml.inference.predictor import AnomalyPredictor


def build_demo(artifact_dir: Path, test_file: Path) -> dict:
    predictor = AnomalyPredictor(artifact_dir)
    processed_dir = test_file.parent
    windows = pd.read_csv(test_file)
    predictions = pd.read_csv(artifact_dir / "results" / "test_predictions.csv")
    metrics = pd.read_csv(artifact_dir / "results" / "test_metrics.csv")
    windows["WindowStartUtc"] = pd.to_datetime(windows["WindowStartUtc"], utc=True)
    predictions["WindowStartUtc"] = pd.to_datetime(
        predictions["WindowStartUtc"], utc=True
    )
    isolation_predictions = predictions.loc[
        predictions["detector"].eq("isolation_forest")
    ].copy()
    isolation_predictions = isolation_predictions.merge(
        windows,
        on=["SensorId", "WindowStartUtc"],
        validate="one_to_one",
    )
    examples: list[dict] = []
    outcomes = (
        (True, True, "Acerto: anomalia detectada"),
        (False, True, "Alarme falso"),
        (True, False, "Anomalia não detectada"),
        (False, False, "Acerto: período normal"),
    )
    for expected, predicted, label in outcomes:
        matches = isolation_predictions.loc[
            isolation_predictions["is_known_anomaly"].eq(expected)
            & isolation_predictions["is_anomaly"].eq(predicted)
        ].sort_values(["SensorId", "WindowStartUtc"])
        if matches.empty:
            continue
        row = matches.iloc[0]
        prediction = predictor.predict(row)
        if prediction["is_anomaly"] != bool(row["is_anomaly"]):
            raise ValueError("saved test predictions do not match the trained model")
        examples.append(
            {
                "sensorId": row["SensorId"],
                "windowStartUtc": prediction["WindowStartUtc"],
                "eventCount5m": int(row["event_count_5m"]),
                "anomalyScore": prediction["anomaly_score"],
                "isAnomaly": prediction["is_anomaly"],
                "expectedAnomaly": bool(row["is_known_anomaly"]),
                "selection": label,
            }
        )

    total_metrics = metrics.loc[metrics["SensorId"].eq("TOTAL")]
    results = [
        {
            "detector": row.detector,
            "truePositives": int(row.TP),
            "falsePositives": int(row.FP),
            "trueNegatives": int(row.TN),
            "falseNegatives": int(row.FN),
        }
        for row in total_metrics.itertuples(index=False)
    ]
    summary_path = artifact_dir / "results" / "dataset_summary.json"
    if summary_path.exists():
        dataset = json.loads(summary_path.read_text(encoding="utf-8"))
        if dataset["testWindows"] != len(windows):
            raise ValueError("dataset summary does not match the test windows")
    else:
        dataset = {
            "trainingWindows": len(pd.read_csv(processed_dir / "train.csv")),
            "validationWindows": len(pd.read_csv(processed_dir / "validation.csv")),
            "testWindows": len(windows),
            "sensorCount": int(windows["SensorId"].nunique()),
        }

    return {
        "source": "simulado",
        "modelVersion": predictor.model_version,
        "dataset": dataset,
        "metrics": results,
        "examples": examples,
    }


def serve(demo: dict, host: str, port: int) -> None:
    body = json.dumps(demo, ensure_ascii=False, allow_nan=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/demo":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer((host, port), Handler) as server:
        print(f"ML experimental disponível em http://{host}:{port}/demo", flush=True)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/isolation_forest_v2"))
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/v2_final/test.csv"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(build_demo(args.artifact_dir, args.test_file), args.host, args.port)


if __name__ == "__main__":
    main()
