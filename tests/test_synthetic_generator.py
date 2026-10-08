from __future__ import annotations

from datetime import timedelta

import pandas as pd

from smartlighting_ml.data_processing.synthetic_generator import (
    EVENT_COLUMNS,
    LABEL_COLUMNS,
    OUTAGE_COLUMNS,
    SENSORS,
    GeneratorConfig,
    generate_data,
    write_csv_files,
)


def test_generates_expected_schema_sensors_and_period() -> None:
    config = GeneratorConfig()
    events, labels, outages = generate_data(config)

    assert list(events.columns) == EVENT_COLUMNS
    assert list(labels.columns) == LABEL_COLUMNS
    assert list(outages.columns) == OUTAGE_COLUMNS
    assert set(events["SensorId"]) == {sensor.sensor_id for sensor in SENSORS}
    assert {sensor.sensor_id for sensor in SENSORS} == {
        "PIR-01",
        "PIR-02",
        "PIR-03",
        "PIR-04",
        "PIR-05",
        "PIR-06",
    }
    assert len({sensor.device_id for sensor in SENSORS}) == 6
    assert events["EventIdentifier"].is_unique
    assert not events.empty

    timestamps = pd.to_datetime(events["OccurredAtUtc"], utc=True)
    expected_start = pd.Timestamp(
        config.start_local, tz=config.local_timezone
    ).tz_convert("UTC")
    assert timestamps.min() >= expected_start
    assert timestamps.max() < expected_start + timedelta(days=config.days)
    assert events["OccurredAtUtc"].str.endswith("Z").all()


def test_anomaly_labels_are_separate_and_reference_events() -> None:
    events, labels, _ = generate_data()

    assert not labels.empty
    assert set(labels["AnomalyType"]) == {
        "out_of_pattern_burst",
        "electrical_oscillation",
    }
    assert set(labels["EventIdentifier"]).issubset(set(events["EventIdentifier"]))
    assert "AnomalyType" not in events.columns


def test_sensor_must_return_inactive_and_rearm_before_next_event() -> None:
    config = GeneratorConfig(include_outages=False)
    events, _, _ = generate_data(config)
    timestamps = pd.to_datetime(events["OccurredAtUtc"], utc=True)
    minimum_gap = (
        config.electrical_pulse_min_seconds + config.electrical_rearm_seconds
    )

    for sensor_id in events["SensorId"].unique():
        sensor_times = timestamps[events["SensorId"].eq(sensor_id)].sort_values()
        gaps = sensor_times.diff().dropna().dt.total_seconds()
        assert gaps.ge(minimum_gap).all()


def test_outages_can_be_disabled() -> None:
    events, _, outages = generate_data(GeneratorConfig(include_outages=False))

    assert not events.empty
    assert outages.empty
    assert list(outages.columns) == OUTAGE_COLUMNS


def test_each_sensor_has_an_individual_probabilistic_pattern() -> None:
    events, _, _ = generate_data(GeneratorConfig(include_outages=False))
    counts = events.groupby("SensorId").size()

    assert len(counts) == 6
    assert counts.nunique() == 6


def test_no_observed_event_occurs_during_an_outage() -> None:
    events, _, outages = generate_data()
    event_times = pd.to_datetime(events["OccurredAtUtc"], utc=True)

    for outage in outages.itertuples(index=False):
        unavailable_from = pd.Timestamp(outage.UnavailableFromUtc)
        unavailable_to = pd.Timestamp(outage.UnavailableToUtc)
        matching_sensor = events["SensorId"].eq(outage.SensorId)
        inside_window = event_times.ge(unavailable_from) & event_times.lt(unavailable_to)
        assert not (matching_sensor & inside_window).any()


def test_generation_is_reproducible(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_paths = write_csv_files(first_dir)
    second_paths = write_csv_files(second_dir)

    for first, second in zip(first_paths, second_paths, strict=True):
        assert first.read_bytes() == second.read_bytes()
