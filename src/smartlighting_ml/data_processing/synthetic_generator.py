"""Generate reproducible PIR telemetry for the Smart Lighting scale model."""

from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


EVENT_COLUMNS = ["EventIdentifier", "SensorId", "DeviceId", "OccurredAtUtc"]
LABEL_COLUMNS = ["EventIdentifier", "AnomalyType"]
OUTAGE_COLUMNS = [
    "SensorId",
    "DeviceId",
    "UnavailableFromUtc",
    "UnavailableToUtc",
    "Reason",
]
IDENTIFIER_NAMESPACE = uuid.UUID("4904498c-1822-4bc8-a2d6-5460de8e5ad1")


@dataclass(frozen=True)
class Sensor:
    sensor_id: str
    device_id: str
    profile_id: int


SENSORS = (
    Sensor("PIR-01", "ESP32-SENSOR-01", 0),
    Sensor("PIR-02", "ESP32-SENSOR-02", 1),
    Sensor("PIR-03", "ESP32-SENSOR-03", 2),
    Sensor("PIR-04", "ESP32-SENSOR-04", 3),
    Sensor("PIR-05", "ESP32-SENSOR-05", 4),
    Sensor("PIR-06", "ESP32-SENSOR-06", 5),
)

# Expected pulses per five-minute window for four periods of the day. These are
# abstract sensor profiles; they deliberately do not encode unconfirmed places.
_PROFILE_RATES = (
    (0.02, 0.75, 0.25, 0.85, 1.10),
    (0.04, 0.45, 0.35, 0.70, 1.20),
    (0.01, 0.30, 0.20, 1.00, 1.25),
    (0.03, 0.65, 0.40, 0.60, 0.95),
    (0.02, 0.40, 0.30, 0.90, 1.15),
    (0.05, 0.55, 0.45, 0.75, 1.05),
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Assumptions used by the synthetic electrical signal simulation."""

    days: int = 30
    seed: int = 20261008
    start_local: datetime = datetime(2026, 1, 1)
    local_timezone: str = "America/Sao_Paulo"
    window_minutes: int = 5
    pulse_duration_min_seconds: float = 2.0
    pulse_duration_max_seconds: float = 6.0
    rearm_seconds: float = 1.5
    electrical_pulse_min_seconds: float = 0.15
    electrical_pulse_max_seconds: float = 0.45
    electrical_rearm_seconds: float = 0.25
    include_outages: bool = True


@dataclass(frozen=True)
class _Candidate:
    occurred_at: datetime
    sensor: Sensor
    active_seconds: float
    rearm_seconds: float
    anomaly_type: str | None = None


def _expected_events(sensor: Sensor, local_time: datetime) -> float:
    """Return the expected pulse count for one five-minute window."""

    hour = local_time.hour
    weekend = local_time.weekday() >= 5

    night, morning, daytime, evening, weekend_factor = _PROFILE_RATES[
        sensor.profile_id
    ]
    if hour < 5:
        rate = night
    elif hour < 10:
        rate = morning
    elif hour < 17:
        rate = daytime
    elif hour < 23:
        rate = evening
    else:
        rate = night
    return rate * weekend_factor if weekend else rate


def _normal_candidates(
    config: GeneratorConfig, rng: np.random.Generator
) -> list[_Candidate]:
    local_zone = ZoneInfo(config.local_timezone)
    start = config.start_local.replace(tzinfo=local_zone)
    end = start + timedelta(days=config.days)
    candidates: list[_Candidate] = []
    cursor = start

    while cursor < end:
        for sensor in SENSORS:
            count = int(rng.poisson(_expected_events(sensor, cursor)))
            for _ in range(count):
                offset = float(rng.uniform(0, config.window_minutes * 60))
                duration = float(
                    rng.uniform(
                        config.pulse_duration_min_seconds,
                        config.pulse_duration_max_seconds,
                    )
                )
                candidates.append(
                    _Candidate(
                        (cursor + timedelta(seconds=offset)).astimezone(timezone.utc),
                        sensor,
                        duration,
                        config.rearm_seconds,
                    )
                )
        cursor += timedelta(minutes=config.window_minutes)

    return candidates


def _anomaly_candidates(
    config: GeneratorConfig, rng: np.random.Generator
) -> list[_Candidate]:
    """Create controlled burst and electrical-oscillation anomalies."""

    local_zone = ZoneInfo(config.local_timezone)
    start = config.start_local.replace(tzinfo=local_zone)
    candidates: list[_Candidate] = []

    for index, sensor in enumerate(SENSORS):
        # One controlled burst in each chronological evaluation period.
        burst_days = (4 + index % 2, 21 + index % 2, 26 + index % 2)
        for day_offset in burst_days:
            burst_start = start + timedelta(
                days=day_offset, hours=2, minutes=10 + index * 2
            )
            elapsed = 0.0
            while elapsed < config.window_minutes * 60:
                elapsed += float(rng.uniform(9.0, 24.0))
                if elapsed >= config.window_minutes * 60:
                    break
                duration = float(
                    rng.uniform(
                        config.pulse_duration_min_seconds,
                        config.pulse_duration_max_seconds,
                    )
                )
                candidates.append(
                    _Candidate(
                        (burst_start + timedelta(seconds=elapsed)).astimezone(
                            timezone.utc
                        ),
                        sensor,
                        duration,
                        config.rearm_seconds,
                        "out_of_pattern_burst",
                    )
                )

        # One rapid LOW/HIGH oscillation in each chronological period.
        oscillation_days = (10 + index % 3, 22 + index % 2, 27 + index % 2)
        for day_offset in oscillation_days:
            oscillation_start = start + timedelta(
                days=day_offset, hours=3, minutes=30 + index * 2
            )
            elapsed = 0.0
            for _ in range(12):
                elapsed += float(rng.uniform(0.8, 1.8))
                duration = float(
                    rng.uniform(
                        config.electrical_pulse_min_seconds,
                        config.electrical_pulse_max_seconds,
                    )
                )
                candidates.append(
                    _Candidate(
                        (oscillation_start + timedelta(seconds=elapsed)).astimezone(
                            timezone.utc
                        ),
                        sensor,
                        duration,
                        config.electrical_rearm_seconds,
                        "electrical_oscillation",
                    )
                )

    return candidates


def _outages(
    config: GeneratorConfig, rng: np.random.Generator
) -> list[dict[str, object]]:
    if not config.include_outages:
        return []

    local_zone = ZoneInfo(config.local_timezone)
    start = config.start_local.replace(tzinfo=local_zone)
    rows: list[dict[str, object]] = []
    for index, sensor in enumerate(SENSORS):
        outage_start = start + timedelta(
            days=6 + index * 3,
            hours=8 + index * 2,
            minutes=int(rng.integers(0, 30)),
        )
        outage_end = outage_start + timedelta(minutes=int(rng.integers(35, 121)))
        rows.append(
            {
                "SensorId": sensor.sensor_id,
                "DeviceId": sensor.device_id,
                "UnavailableFromUtc": outage_start.astimezone(timezone.utc),
                "UnavailableToUtc": outage_end.astimezone(timezone.utc),
                "Reason": "simulated_telemetry_unavailability",
            }
        )
    return rows


def _is_unavailable(candidate: _Candidate, outages: list[dict[str, object]]) -> bool:
    return any(
        row["SensorId"] == candidate.sensor.sensor_id
        and row["UnavailableFromUtc"] <= candidate.occurred_at < row["UnavailableToUtc"]
        for row in outages
    )


def generate_data(
    config: GeneratorConfig = GeneratorConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate observable events, evaluation-only labels, and outage windows."""

    if config.days <= 0:
        raise ValueError("days must be greater than zero")
    if config.window_minutes <= 0:
        raise ValueError("window_minutes must be greater than zero")

    rng = np.random.default_rng(config.seed)
    candidates = _normal_candidates(config, rng) + _anomaly_candidates(config, rng)
    candidates.sort(key=lambda item: (item.sensor.sensor_id, item.occurred_at))
    outages = _outages(config, rng)

    accepted: list[_Candidate] = []
    next_available: dict[str, datetime] = {}
    for candidate in candidates:
        available_at = next_available.get(
            candidate.sensor.sensor_id, datetime.min.replace(tzinfo=timezone.utc)
        )
        if candidate.occurred_at < available_at:
            continue
        next_available[candidate.sensor.sensor_id] = candidate.occurred_at + timedelta(
            seconds=candidate.active_seconds + candidate.rearm_seconds
        )
        if not _is_unavailable(candidate, outages):
            accepted.append(candidate)

    accepted.sort(key=lambda item: item.occurred_at)
    event_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []
    for sequence, candidate in enumerate(accepted):
        identity_source = (
            f"{config.seed}:{candidate.sensor.sensor_id}:"
            f"{candidate.occurred_at.isoformat()}:{sequence}"
        )
        event_id = str(uuid.uuid5(IDENTIFIER_NAMESPACE, identity_source))
        event_rows.append(
            {
                "EventIdentifier": event_id,
                "SensorId": candidate.sensor.sensor_id,
                "DeviceId": candidate.sensor.device_id,
                "OccurredAtUtc": candidate.occurred_at.isoformat().replace("+00:00", "Z"),
            }
        )
        if candidate.anomaly_type:
            label_rows.append(
                {
                    "EventIdentifier": event_id,
                    "AnomalyType": candidate.anomaly_type,
                }
            )

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    labels = pd.DataFrame(label_rows, columns=LABEL_COLUMNS)
    outage_frame = pd.DataFrame(outages, columns=OUTAGE_COLUMNS)
    for column in ("UnavailableFromUtc", "UnavailableToUtc"):
        if not outage_frame.empty:
            outage_frame[column] = outage_frame[column].map(
                lambda value: value.isoformat().replace("+00:00", "Z")
            )
    return events, labels, outage_frame


def write_csv_files(
    output_dir: Path, config: GeneratorConfig = GeneratorConfig()
) -> tuple[Path, Path, Path]:
    """Generate the dataset and write its three independent CSV files."""

    events, labels, outages = generate_data(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "pir_events.csv",
        output_dir / "anomaly_labels.csv",
        output_dir / "telemetry_outages.csv",
    )
    events.to_csv(paths[0], index=False, lineterminator="\n")
    labels.to_csv(paths[1], index=False, lineterminator="\n")
    outages.to_csv(paths[2], index=False, lineterminator="\n")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument(
        "--include-outages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate telemetry-unavailability windows (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = GeneratorConfig(
        days=args.days,
        seed=args.seed,
        include_outages=args.include_outages,
    )
    paths = write_csv_files(args.output_dir, config)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
