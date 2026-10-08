from __future__ import annotations

from pathlib import Path

import pandas as pd

from smartlighting_ml.data_processing.preparation_pipeline import load_inputs
from smartlighting_ml.experiments.v2 import (
    V2Config,
    build_progressive_folds,
    load_development_data,
)


RAW_DIR = Path("data/raw/synthetic_v2_development")
PROCESSED_DIR = Path("data/processed/v2_development")


def _episodes() -> pd.DataFrame:
    events = pd.read_csv(RAW_DIR / "pir_events.csv", parse_dates=["OccurredAtUtc"])
    labels = pd.read_csv(RAW_DIR / "anomaly_labels.csv")
    return (
        labels.merge(
            events[["EventIdentifier", "SensorId", "OccurredAtUtc"]],
            on="EventIdentifier",
            validate="many_to_one",
        )
        .groupby(["EpisodeIdentifier", "SensorId", "AnomalyType"])["OccurredAtUtc"]
        .agg(["min", "max"])
        .reset_index()
    )


def test_v2_scenario_covers_120_days_and_six_fixed_sensors() -> None:
    events = pd.read_csv(RAW_DIR / "pir_events.csv", parse_dates=["OccurredAtUtc"])

    assert events["EventIdentifier"].is_unique
    assert set(events["SensorId"]) == {f"PIR-{index:02d}" for index in range(1, 7)}
    assert set(events["DeviceId"]) == {
        f"ESP32-SENSOR-{index:02d}" for index in range(1, 7)
    }
    start = events["OccurredAtUtc"].min().tz_convert("America/Sao_Paulo").normalize()
    end = events["OccurredAtUtc"].max().tz_convert("America/Sao_Paulo").normalize()
    assert (end - start).days == 119


def test_labels_are_consistent_and_episodes_are_independent() -> None:
    events = pd.read_csv(RAW_DIR / "pir_events.csv")
    labels = pd.read_csv(RAW_DIR / "anomaly_labels.csv")
    episodes = _episodes()

    assert labels["EventIdentifier"].is_unique
    assert set(labels["EventIdentifier"]).issubset(set(events["EventIdentifier"]))
    assert episodes.groupby("EpisodeIdentifier")["SensorId"].nunique().max() == 1
    assert episodes.groupby("EpisodeIdentifier")["AnomalyType"].nunique().max() == 1
    assert set(episodes["AnomalyType"]) == {
        "out_of_pattern_burst",
        "electrical_oscillation",
    }
    assert episodes["EpisodeIdentifier"].nunique() >= 700


def test_anomalies_cover_hours_weekdays_and_required_fold_episodes() -> None:
    episodes = _episodes()
    local_start = episodes["min"].dt.tz_convert("America/Sao_Paulo")
    assert local_start.dt.hour.nunique() == 24
    assert set(local_start.dt.weekday) == set(range(7))

    development, _ = load_development_data(PROCESSED_DIR)
    folds = build_progressive_folds(
        development, V2Config(processed_dir=PROCESSED_DIR)
    )
    for fold in folds:
        crossing = episodes.loc[
            (
                episodes["min"].lt(fold.train_end)
                & episodes["max"].ge(fold.train_end)
            )
            | (
                episodes["min"].lt(fold.calibration_end)
                & episodes["max"].ge(fold.calibration_end)
            )
        ]
        assert crossing.empty
        for start, end in (
            (fold.calibration_start, fold.calibration_end),
            (fold.evaluation_start, fold.evaluation_end),
        ):
            selected = episodes.loc[
                episodes["min"].ge(start) & episodes["max"].lt(end)
            ]
            counts = selected.groupby(["SensorId", "AnomalyType"]).size()
            assert counts.min() >= 10


def test_legitimate_intense_activity_is_not_an_anomaly() -> None:
    legitimate = pd.read_csv(
        RAW_DIR / "legitimate_activity_episodes.csv",
        parse_dates=["StartUtc", "EndUtc"],
    )
    labels = pd.read_csv(RAW_DIR / "anomaly_labels.csv")
    episodes = _episodes()

    assert set(legitimate["EpisodeIdentifier"]).isdisjoint(
        set(labels["EpisodeIdentifier"])
    )
    overlap = legitimate.merge(episodes, on="SensorId")
    overlap = overlap.loc[
        overlap["StartUtc"].le(overlap["max"])
        & overlap["EndUtc"].ge(overlap["min"])
    ]
    assert overlap.empty


def test_outages_produce_unreliable_null_windows_and_no_observed_events() -> None:
    events = pd.read_csv(RAW_DIR / "pir_events.csv", parse_dates=["OccurredAtUtc"])
    outages = pd.read_csv(
        RAW_DIR / "telemetry_outages.csv",
        parse_dates=["UnavailableFromUtc", "UnavailableToUtc"],
    )
    windows = pd.read_csv(
        PROCESSED_DIR / "development_windows.csv",
        parse_dates=["WindowStartUtc"],
    )

    unreliable = windows.loc[~windows["telemetry_reliable"]]
    assert not unreliable.empty
    assert unreliable["event_count_5m"].isna().all()
    for outage in outages.itertuples(index=False):
        inside = events["SensorId"].eq(outage.SensorId) & events["OccurredAtUtc"].ge(
            outage.UnavailableFromUtc
        ) & events["OccurredAtUtc"].lt(outage.UnavailableToUtc)
        assert not inside.any()


def test_v2_input_deduplication_preserves_episode_identifier(tmp_path) -> None:
    events = pd.read_csv(RAW_DIR / "pir_events.csv").iloc[:20].copy()
    labelled_ids = set(pd.read_csv(RAW_DIR / "anomaly_labels.csv")["EventIdentifier"])
    labelled = events.loc[events["EventIdentifier"].isin(labelled_ids)]
    if labelled.empty:
        first_label = pd.read_csv(RAW_DIR / "anomaly_labels.csv").iloc[0]
        source = pd.read_csv(RAW_DIR / "pir_events.csv")
        events = source.loc[source["EventIdentifier"].eq(first_label.EventIdentifier)]
    events = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    labels = pd.read_csv(RAW_DIR / "anomaly_labels.csv")
    labels = labels.loc[labels["EventIdentifier"].isin(events["EventIdentifier"])].copy()
    outages = pd.DataFrame(
        columns=[
            "SensorId",
            "DeviceId",
            "UnavailableFromUtc",
            "UnavailableToUtc",
            "Reason",
        ]
    )
    events.to_csv(tmp_path / "pir_events.csv", index=False)
    labels.to_csv(tmp_path / "anomaly_labels.csv", index=False)
    outages.to_csv(tmp_path / "telemetry_outages.csv", index=False)

    loaded_events, _, loaded_labels = load_inputs(tmp_path)

    assert loaded_events["EventIdentifier"].is_unique
    assert "EpisodeIdentifier" in loaded_labels.columns
