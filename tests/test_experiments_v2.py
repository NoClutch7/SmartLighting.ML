from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from smartlighting_ml.experiments.v2 import (
    FEATURE_SETS,
    HyperParameters,
    V2Config,
    build_progressive_folds,
    load_development_data,
    run_experiments,
)


PROCESSED_DIR = Path("data/processed")
V1_DIR = Path("artifacts/isolation_forest_v1")


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("experiments-v2")
    config = V2Config(processed_dir=PROCESSED_DIR, output_dir=output_dir)
    metrics = run_experiments(
        config,
        feature_sets={"count_with_history": FEATURE_SETS["count_with_history"]},
        hyperparameters=[HyperParameters(10, 64, 1.0)],
    )
    report = json.loads(
        (output_dir / "development_report.json").read_text(encoding="utf-8")
    )
    return metrics, report, output_dir


def test_feature_sets_cover_the_four_requested_variants() -> None:
    assert set(FEATURE_SETS) == {
        "original_9",
        "without_rolling_mean_1h",
        "count_with_history",
        "count_with_temporal",
    }
    assert len(FEATURE_SETS["original_9"]) == 9
    assert "rolling_mean_1h" not in FEATURE_SETS["without_rolling_mean_1h"]
    assert FEATURE_SETS["count_with_history"] == [
        "event_count_5m",
        "events_last_15m",
        "events_last_60m",
    ]


def test_loader_never_reads_v1_test_files(monkeypatch) -> None:
    paths: list[str] = []
    original = pd.read_csv

    def recording_read_csv(path, *args, **kwargs):
        paths.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    development, _ = load_development_data(PROCESSED_DIR)

    assert set(development["split"]) == {"train", "validation"}
    assert not any(Path(path).name in {"test.csv", "test_predictions.csv"} for path in paths)


def test_progressive_folds_are_strictly_ordered_and_do_not_overlap() -> None:
    development, _ = load_development_data(PROCESSED_DIR)
    folds = build_progressive_folds(development, V2Config())

    assert len(folds) >= 2
    for fold in folds:
        assert fold.train_start < fold.train_end
        assert fold.train_end == fold.calibration_start
        assert fold.calibration_start < fold.calibration_end
        assert fold.calibration_end == fold.evaluation_start
        assert fold.evaluation_start < fold.evaluation_end
    assert all(
        later.train_end > earlier.train_end
        for earlier, later in zip(folds, folds[1:], strict=False)
    )


def test_reports_metrics_by_type_and_zero_count_false_positives(v2_run) -> None:
    metrics, report, output_dir = v2_run

    assert not metrics.empty
    assert {"overall", "out_of_pattern_burst", "electrical_oscillation"}.issubset(
        set(metrics["anomaly_type"])
    )
    assert {
        "TP",
        "FP",
        "TN",
        "FN",
        "precision",
        "recall",
        "f1_score",
        "false_positives_zero_count",
        "false_positive_rate_zero_count",
    }.issubset(metrics.columns)
    assert (metrics[["TP", "FP", "TN", "FN"]] >= 0).all().all()
    assert report["test_data_accessed"] is False
    assert (output_dir / "development_fold_metrics.csv").exists()


def test_insufficient_positive_folds_are_explicitly_reported(v2_run) -> None:
    metrics, report, _ = v2_run

    assert report["status"] == "insufficient_positive_windows"
    assert report["limitations"]
    assert (~metrics["sufficient_for_comparison"]).any()
    assert "No configuration" in report["interpretation"]


def test_v1_artifacts_are_preserved_by_v2_run(tmp_path) -> None:
    before = _hash_tree(V1_DIR)
    run_experiments(
        V2Config(processed_dir=PROCESSED_DIR, output_dir=tmp_path / "v2"),
        feature_sets={"count_with_temporal": FEATURE_SETS["count_with_temporal"]},
        hyperparameters=[HyperParameters(5, 32, 0.5)],
    )
    after = _hash_tree(V1_DIR)

    assert before
    assert before == after
