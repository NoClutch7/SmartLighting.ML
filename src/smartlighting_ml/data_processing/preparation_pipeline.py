"""Prepare five-minute PIR features without training an anomaly detector."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_COLUMNS = ["EventIdentifier", "SensorId", "DeviceId", "OccurredAtUtc"]
OUTAGE_COLUMNS = [
    "SensorId",
    "DeviceId",
    "UnavailableFromUtc",
    "UnavailableToUtc",
    "Reason",
]
LABEL_COLUMNS = ["EventIdentifier", "AnomalyType"]
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
MODEL_DATASET_COLUMNS = [
    "SensorId",
    "DeviceId",
    "WindowStartUtc",
    "WindowStartLocal",
    *FEATURE_COLUMNS,
]


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path = Path("data/raw/synthetic")
    output_dir: Path = Path("data/processed")
    local_timezone: str = "America/Sao_Paulo"
    window_minutes: int = 5
    train_ratio: float = 0.70
    validation_ratio: float = 0.15


@dataclass(frozen=True)
class PipelineResult:
    all_windows: pd.DataFrame
    training: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    anomaly_windows: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _parse_utc(series: pd.Series, column: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())
        raise ValueError(f"{column} contains {invalid_count} invalid UTC timestamp(s)")
    return parsed


def load_inputs(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate events, outages, and evaluation-only anomaly labels."""

    events = pd.read_csv(raw_dir / "pir_events.csv", dtype="string")
    outages = pd.read_csv(raw_dir / "telemetry_outages.csv", dtype="string")
    labels = pd.read_csv(raw_dir / "anomaly_labels.csv", dtype="string")

    _require_columns(events, EVENT_COLUMNS, "pir_events.csv")
    _require_columns(outages, OUTAGE_COLUMNS, "telemetry_outages.csv")
    _require_columns(labels, LABEL_COLUMNS, "anomaly_labels.csv")

    for column in ("EventIdentifier", "SensorId", "DeviceId"):
        if events[column].isna().any() or events[column].str.strip().eq("").any():
            raise ValueError(f"pir_events.csv contains an empty {column}")

    events = events.loc[:, EVENT_COLUMNS].copy()
    events["OccurredAtUtc"] = _parse_utc(events["OccurredAtUtc"], "OccurredAtUtc")
    events = events.drop_duplicates(subset="EventIdentifier", keep="first")
    events = events.sort_values("OccurredAtUtc", kind="stable").reset_index(drop=True)
    if events.empty:
        raise ValueError("pir_events.csv contains no events")

    device_counts = events.groupby("SensorId")["DeviceId"].nunique()
    if device_counts.gt(1).any():
        sensors = ", ".join(device_counts[device_counts.gt(1)].index.astype(str))
        raise ValueError(f"a sensor is associated with multiple devices: {sensors}")

    outages = outages.loc[:, OUTAGE_COLUMNS].copy()
    outages["UnavailableFromUtc"] = _parse_utc(
        outages["UnavailableFromUtc"], "UnavailableFromUtc"
    )
    outages["UnavailableToUtc"] = _parse_utc(
        outages["UnavailableToUtc"], "UnavailableToUtc"
    )
    if (outages["UnavailableToUtc"] <= outages["UnavailableFromUtc"]).any():
        raise ValueError("telemetry_outages.csv contains a non-positive interval")
    unknown_outage_sensors = set(outages["SensorId"]) - set(events["SensorId"])
    if unknown_outage_sensors:
        raise ValueError("telemetry_outages.csv references an unknown sensor")

    optional_label_columns = [
        column for column in ("EpisodeIdentifier",) if column in labels.columns
    ]
    labels = labels.loc[:, [*LABEL_COLUMNS, *optional_label_columns]].copy()
    if labels["EventIdentifier"].duplicated().any():
        raise ValueError("anomaly_labels.csv contains duplicate EventIdentifier values")
    unknown_labels = set(labels["EventIdentifier"]) - set(events["EventIdentifier"])
    if unknown_labels:
        raise ValueError("anomaly_labels.csv references an unknown event")
    return events, outages, labels


def _build_window_grid(
    events: pd.DataFrame, window_minutes: int, local_timezone: str
) -> pd.DataFrame:
    frequency = f"{window_minutes}min"
    local_events = events["OccurredAtUtc"].dt.tz_convert(local_timezone)
    # Complete local calendar days preserve empty boundary windows as real silence.
    start = local_events.min().normalize().tz_convert("UTC")
    end = (local_events.max().normalize() + pd.Timedelta(days=1)).tz_convert("UTC")
    starts = pd.date_range(start=start, end=end, freq=frequency, inclusive="left")
    devices = events[["SensorId", "DeviceId"]].drop_duplicates().sort_values("SensorId")
    grid = devices.merge(pd.DataFrame({"WindowStartUtc": starts}), how="cross")

    counted = events.assign(
        WindowStartUtc=events["OccurredAtUtc"].dt.floor(frequency)
    ).groupby(["SensorId", "WindowStartUtc"], as_index=False).size()
    counted = counted.rename(columns={"size": "event_count_5m"})
    grid = grid.merge(counted, on=["SensorId", "WindowStartUtc"], how="left")
    grid["event_count_5m"] = grid["event_count_5m"].fillna(0).astype("Float64")
    return grid.sort_values(["SensorId", "WindowStartUtc"]).reset_index(drop=True)


def _mark_reliability(
    windows: pd.DataFrame, outages: pd.DataFrame, window_minutes: int
) -> pd.DataFrame:
    result = windows.copy()
    result["telemetry_reliable"] = True
    window_end = result["WindowStartUtc"] + pd.Timedelta(minutes=window_minutes)
    for outage in outages.itertuples(index=False):
        affected = (
            result["SensorId"].eq(outage.SensorId)
            & result["WindowStartUtc"].lt(outage.UnavailableToUtc)
            & window_end.gt(outage.UnavailableFromUtc)
        )
        result.loc[affected, "telemetry_reliable"] = False
    result.loc[~result["telemetry_reliable"], "event_count_5m"] = pd.NA
    return result


def _add_features(windows: pd.DataFrame, local_timezone: str) -> pd.DataFrame:
    result = windows.copy()
    local_time = result["WindowStartUtc"].dt.tz_convert(local_timezone)
    result["WindowStartLocal"] = local_time
    hour = local_time.dt.hour + local_time.dt.minute / 60.0
    weekday = local_time.dt.weekday
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["is_weekend"] = weekday.ge(5).astype("int8")

    result["events_last_15m"] = np.nan
    result["events_last_60m"] = np.nan
    result["rolling_mean_1h"] = np.nan
    for _, indexes in result.groupby("SensorId", sort=False).groups.items():
        counts = result.loc[indexes, "event_count_5m"].astype("float64")
        prior = counts.shift(1)
        result.loc[indexes, "events_last_15m"] = (
            prior.rolling(3, min_periods=3).sum().to_numpy()
        )
        prior_hour = prior.rolling(12, min_periods=12)
        result.loc[indexes, "events_last_60m"] = prior_hour.sum().to_numpy()
        result.loc[indexes, "rolling_mean_1h"] = prior_hour.mean().to_numpy()
    return result


def _map_anomaly_windows(
    events: pd.DataFrame, labels: pd.DataFrame, window_minutes: int
) -> pd.DataFrame:
    frequency = f"{window_minutes}min"
    mapped = labels.merge(
        events[["EventIdentifier", "SensorId", "DeviceId", "OccurredAtUtc"]],
        on="EventIdentifier",
        how="left",
        validate="one_to_one",
    )
    mapped["WindowStartUtc"] = mapped["OccurredAtUtc"].dt.floor(frequency)
    columns = [
        "EventIdentifier",
        "SensorId",
        "DeviceId",
        "WindowStartUtc",
        "AnomalyType",
    ]
    if "EpisodeIdentifier" in mapped.columns:
        columns.append("EpisodeIdentifier")
    return mapped[columns].sort_values(
        ["WindowStartUtc", "SensorId", "EventIdentifier"]
    )


def _training_contamination_keys(
    anomaly_windows: pd.DataFrame, window_minutes: int
) -> set[tuple[str, pd.Timestamp]]:
    keys: set[tuple[str, pd.Timestamp]] = set()
    step = pd.Timedelta(minutes=window_minutes)
    lookback_windows = 60 // window_minutes
    unique_windows = anomaly_windows[["SensorId", "WindowStartUtc"]].drop_duplicates()
    for row in unique_windows.itertuples(index=False):
        # Current anomaly plus every later window whose one-hour history sees it.
        for offset in range(lookback_windows + 1):
            keys.add((row.SensorId, row.WindowStartUtc + offset * step))
    return keys


def prepare_datasets(
    events: pd.DataFrame,
    outages: pd.DataFrame,
    labels: pd.DataFrame,
    config: PipelineConfig = PipelineConfig(),
) -> PipelineResult:
    """Create window features and chronological model datasets."""

    if config.window_minutes != 5:
        raise ValueError("this pipeline currently requires five-minute windows")
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")
    if not 0 < config.validation_ratio < 1 - config.train_ratio:
        raise ValueError("validation_ratio leaves no room for the test split")

    windows = _build_window_grid(
        events, config.window_minutes, config.local_timezone
    )
    windows = _mark_reliability(windows, outages, config.window_minutes)
    windows = _add_features(windows, config.local_timezone)
    anomaly_windows = _map_anomaly_windows(events, labels, config.window_minutes)

    unique_times = windows["WindowStartUtc"].drop_duplicates().sort_values().array
    train_index = int(len(unique_times) * config.train_ratio)
    validation_index = int(
        len(unique_times) * (config.train_ratio + config.validation_ratio)
    )
    train_end = unique_times[train_index]
    validation_end = unique_times[validation_index]
    windows["split"] = np.select(
        [
            windows["WindowStartUtc"].lt(train_end),
            windows["WindowStartUtc"].lt(validation_end),
        ],
        ["train", "validation"],
        default="test",
    )

    eligible = windows["telemetry_reliable"] & windows[FEATURE_COLUMNS].notna().all(axis=1)
    contamination = _training_contamination_keys(
        anomaly_windows, config.window_minutes
    )
    contaminated = pd.Series(
        [
            (row.SensorId, row.WindowStartUtc) in contamination
            for row in windows[["SensorId", "WindowStartUtc"]].itertuples(index=False)
        ],
        index=windows.index,
    )
    training = windows.loc[eligible & windows["split"].eq("train") & ~contaminated]
    validation = windows.loc[eligible & windows["split"].eq("validation")]
    test = windows.loc[eligible & windows["split"].eq("test")]

    return PipelineResult(
        all_windows=windows,
        training=training[MODEL_DATASET_COLUMNS].reset_index(drop=True),
        validation=validation[MODEL_DATASET_COLUMNS].reset_index(drop=True),
        test=test[MODEL_DATASET_COLUMNS].reset_index(drop=True),
        anomaly_windows=anomaly_windows.reset_index(drop=True),
    )


def run_pipeline(config: PipelineConfig = PipelineConfig()) -> PipelineResult:
    events, outages, labels = load_inputs(config.raw_dir)
    result = prepare_datasets(events, outages, labels, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result.all_windows.to_csv(config.output_dir / "all_windows.csv", index=False)
    result.training.to_csv(config.output_dir / "train.csv", index=False)
    result.validation.to_csv(config.output_dir / "validation.csv", index=False)
    result.test.to_csv(config.output_dir / "test.csv", index=False)
    result.anomaly_windows.to_csv(
        config.output_dir / "anomaly_windows_evaluation.csv", index=False
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pipeline(PipelineConfig(raw_dir=args.raw_dir, output_dir=args.output_dir))
    print(f"windows={len(result.all_windows)}")
    print(f"unreliable={int((~result.all_windows['telemetry_reliable']).sum())}")
    print(f"train={len(result.training)}")
    print(f"validation={len(result.validation)}")
    print(f"test={len(result.test)}")


if __name__ == "__main__":
    main()
