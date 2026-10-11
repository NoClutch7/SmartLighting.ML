from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from smartlighting_ml.data_processing.preparation_pipeline import (
    EVENT_COLUMNS,
    FEATURE_COLUMNS,
    PipelineConfig,
    _training_contamination_keys,
    load_inputs,
    prepare_datasets,
    run_pipeline,
)


RAW_DIR = Path("data/raw/synthetic")


@pytest.fixture(scope="module")
def prepared():
    events, outages, labels = load_inputs(RAW_DIR)
    return prepare_datasets(events, outages, labels)


def test_load_validates_deduplicates_and_sorts(tmp_path) -> None:
    source_events = pd.read_csv(RAW_DIR / "sensor_events.csv", dtype="string")
    duplicate = source_events.iloc[[0]]
    shuffled = pd.concat([source_events.iloc[::-1], duplicate], ignore_index=True)
    shuffled.to_csv(tmp_path / "sensor_events.csv", index=False)
    pd.read_csv(RAW_DIR / "telemetry_outages.csv").to_csv(
        tmp_path / "telemetry_outages.csv", index=False
    )
    pd.read_csv(RAW_DIR / "anomaly_labels.csv").to_csv(
        tmp_path / "anomaly_labels.csv", index=False
    )

    events, _, _ = load_inputs(tmp_path)

    assert list(events.columns) == EVENT_COLUMNS
    assert events["EventIdentifier"].is_unique
    assert events["OccurredAtUtc"].is_monotonic_increasing
    assert len(events) == len(source_events)


def test_windows_include_reliable_zero_counts_and_unreliable_nulls(prepared) -> None:
    windows = prepared.all_windows

    assert len(windows) == 6 * 30 * 24 * 12
    assert ((windows["telemetry_reliable"]) & windows["event_count_5m"].eq(0)).any()
    unreliable = windows.loc[~windows["telemetry_reliable"]]
    assert not unreliable.empty
    assert unreliable["event_count_5m"].isna().all()


def test_local_time_and_cyclical_features_are_correct(prepared) -> None:
    first = prepared.all_windows.sort_values("WindowStartUtc").iloc[0]

    assert str(first["WindowStartUtc"]).startswith("2026-01-01 03:00:00+00:00")
    assert str(first["WindowStartLocal"]).startswith("2026-01-01 00:00:00-03:00")
    assert math.isclose(first["hour_sin"], 0.0, abs_tol=1e-12)
    assert math.isclose(first["hour_cos"], 1.0, abs_tol=1e-12)
    assert first["is_weekend"] == 0


def test_historical_features_use_only_strictly_prior_windows(prepared) -> None:
    sensor_windows = prepared.all_windows.loc[
        prepared.all_windows["SensorId"].eq("SENSOR-01")
    ].reset_index(drop=True)
    candidate_indexes = sensor_windows.index[
        sensor_windows[FEATURE_COLUMNS].notna().all(axis=1)
    ]
    index = int(candidate_indexes[0])
    previous = sensor_windows.loc[index - 12 : index - 1, "event_count_5m"]
    current = sensor_windows.loc[index]

    assert current["events_last_15m"] == previous.iloc[-3:].sum()
    assert current["events_last_60m"] == previous.sum()
    assert current["rolling_mean_1h"] == pytest.approx(previous.mean())

    changed_current = current["event_count_5m"] + 100
    assert current["events_last_60m"] != previous.iloc[1:].sum() + changed_current


def test_splits_are_chronological_and_have_no_feature_nulls(prepared) -> None:
    assert prepared.training["WindowStartUtc"].max() < prepared.validation[
        "WindowStartUtc"
    ].min()
    assert prepared.validation["WindowStartUtc"].max() < prepared.test[
        "WindowStartUtc"
    ].min()
    for dataset in (prepared.training, prepared.validation, prepared.test):
        assert not dataset[FEATURE_COLUMNS].isna().any().any()


def test_known_anomalies_and_their_lookback_effect_do_not_enter_training(
    prepared,
) -> None:
    excluded = _training_contamination_keys(prepared.anomaly_windows, 5)
    training_keys = set(
        prepared.training[["SensorId", "WindowStartUtc"]].itertuples(
            index=False, name=None
        )
    )

    assert training_keys.isdisjoint(excluded)
    assert "AnomalyType" not in prepared.training.columns


def test_pipeline_writes_all_datasets(tmp_path) -> None:
    result = run_pipeline(PipelineConfig(raw_dir=RAW_DIR, output_dir=tmp_path))

    assert len(result.all_windows) > 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "all_windows.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "anomaly_windows_evaluation.csv",
    }
