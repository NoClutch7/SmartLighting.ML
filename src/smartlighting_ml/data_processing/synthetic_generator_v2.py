"""Generate synthetic development or later holdout scenarios for v2 experiments."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from smartlighting_ml.data_processing.synthetic_generator import (
    EVENT_COLUMNS,
    OUTAGE_COLUMNS,
    SENSORS,
    _PROFILE_RATES,
)


V2_NAMESPACE = uuid.UUID("45bc78cb-7152-49ac-baf8-a80d74924ed4")
V2_LABEL_COLUMNS = ["EventIdentifier", "AnomalyType", "EpisodeIdentifier"]
LEGITIMATE_COLUMNS = [
    "EpisodeIdentifier",
    "SensorId",
    "DeviceId",
    "StartUtc",
    "EndUtc",
    "ScenarioType",
]


@dataclass(frozen=True)
class V2GeneratorConfig:
    scenario: str = "synthetic_v2_development"
    days: int = 120
    seed: int = 20261008
    start_local: datetime = datetime(2026, 2, 1)
    local_timezone: str = "America/Sao_Paulo"
    window_minutes: int = 5
    pulse_duration_min_seconds: float = 2.0
    pulse_duration_max_seconds: float = 6.0
    rearm_seconds: float = 1.5
    electrical_pulse_min_seconds: float = 0.15
    electrical_pulse_max_seconds: float = 0.45
    electrical_rearm_seconds: float = 0.25


@dataclass(frozen=True)
class _Candidate:
    occurred_at: datetime
    sensor_id: str
    device_id: str
    active_seconds: float
    rearm_seconds: float
    anomaly_type: str | None = None
    episode_id: str | None = None


def _episode_id(kind: str, sensor_id: str, start: datetime) -> str:
    return str(uuid.uuid5(V2_NAMESPACE, f"{kind}:{sensor_id}:{start.isoformat()}"))


def _normal_candidates(
    config: V2GeneratorConfig, rng: np.random.Generator
) -> list[_Candidate]:
    zone = ZoneInfo(config.local_timezone)
    start = config.start_local.replace(tzinfo=zone)
    candidates: list[_Candidate] = []
    for day_offset in range(config.days):
        day_start = start + timedelta(days=day_offset)
        # Stable within a day but variable across days and sensors.
        daily_factors = rng.lognormal(mean=0.0, sigma=0.18, size=len(SENSORS))
        cursor = day_start
        day_end = day_start + timedelta(days=1)
        while cursor < day_end:
            for index, sensor in enumerate(SENSORS):
                night, morning, daytime, evening, weekend_factor = _PROFILE_RATES[
                    sensor.profile_id
                ]
                hour = cursor.hour
                if hour < 5 or hour >= 23:
                    base_rate = night
                elif hour < 10:
                    base_rate = morning
                elif hour < 17:
                    base_rate = daytime
                else:
                    base_rate = evening
                if cursor.weekday() >= 5:
                    base_rate *= weekend_factor
                # Slow periodic variation avoids identical weeks.
                seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * day_offset / 28 + index)
                count = int(rng.poisson(base_rate * daily_factors[index] * seasonal))
                for _ in range(count):
                    candidates.append(
                        _Candidate(
                            occurred_at=(
                                cursor
                                + timedelta(
                                    seconds=float(
                                        rng.uniform(0, config.window_minutes * 60)
                                    )
                                )
                            ).astimezone(timezone.utc),
                            sensor_id=sensor.sensor_id,
                            device_id=sensor.device_id,
                            active_seconds=float(
                                rng.uniform(
                                    config.pulse_duration_min_seconds,
                                    config.pulse_duration_max_seconds,
                                )
                            ),
                            rearm_seconds=config.rearm_seconds,
                        )
                    )
            cursor += timedelta(minutes=config.window_minutes)
    return candidates


def _anomaly_candidates(
    config: V2GeneratorConfig, rng: np.random.Generator
) -> list[_Candidate]:
    """Generate one independent, varied anomaly episode per sensor per day."""

    zone = ZoneInfo(config.local_timezone)
    start = config.start_local.replace(tzinfo=zone)
    candidates: list[_Candidate] = []
    for day_offset in range(config.days):
        for sensor_index, sensor in enumerate(SENSORS):
            hour = (day_offset * 7 + sensor_index * 3) % 24
            minute = ((day_offset * 11 + sensor_index * 7) % 10) * 5
            episode_start = start + timedelta(
                days=day_offset, hours=hour, minutes=minute
            )
            anomaly_type = (
                "out_of_pattern_burst"
                if (day_offset + sensor_index) % 2 == 0
                else "electrical_oscillation"
            )
            episode = _episode_id(anomaly_type, sensor.sensor_id, episode_start)
            if anomaly_type == "out_of_pattern_burst":
                elapsed = 0.0
                while elapsed < 180:
                    elapsed += float(rng.uniform(9.0, 22.0))
                    if elapsed >= 180:
                        break
                    candidates.append(
                        _Candidate(
                            (episode_start + timedelta(seconds=elapsed)).astimezone(
                                timezone.utc
                            ),
                            sensor.sensor_id,
                            sensor.device_id,
                            float(
                                rng.uniform(
                                    config.pulse_duration_min_seconds,
                                    config.pulse_duration_max_seconds,
                                )
                            ),
                            config.rearm_seconds,
                            anomaly_type,
                            episode,
                        )
                    )
            else:
                elapsed = 0.0
                for _ in range(12):
                    elapsed += float(rng.uniform(0.8, 1.8))
                    candidates.append(
                        _Candidate(
                            (episode_start + timedelta(seconds=elapsed)).astimezone(
                                timezone.utc
                            ),
                            sensor.sensor_id,
                            sensor.device_id,
                            float(
                                rng.uniform(
                                    config.electrical_pulse_min_seconds,
                                    config.electrical_pulse_max_seconds,
                                )
                            ),
                            config.electrical_rearm_seconds,
                            anomaly_type,
                            episode,
                        )
                    )
    return candidates


def _legitimate_intense_candidates(
    config: V2GeneratorConfig, rng: np.random.Generator
) -> tuple[list[_Candidate], list[dict[str, str]]]:
    """Add realistic hard negatives without anomaly labels."""

    zone = ZoneInfo(config.local_timezone)
    start = config.start_local.replace(tzinfo=zone)
    candidates: list[_Candidate] = []
    episodes: list[dict[str, str]] = []
    for day_offset in range(2, config.days, 7):
        for sensor_index, sensor in enumerate(SENSORS):
            local_start = start + timedelta(
                days=day_offset,
                hours=8 + (sensor_index * 2 + day_offset) % 14,
                minutes=(sensor_index % 3) * 10,
            )
            duration_minutes = int(rng.integers(10, 21))
            anomaly_start = start + timedelta(
                days=day_offset,
                hours=(day_offset * 7 + sensor_index * 3) % 24,
                minutes=((day_offset * 11 + sensor_index * 7) % 10) * 5,
            )
            anomaly_end = anomaly_start + timedelta(minutes=3)
            # Keep hard negatives semantically separate from labelled anomalies.
            while (
                local_start < anomaly_end
                and local_start + timedelta(minutes=duration_minutes) > anomaly_start
            ):
                local_start += timedelta(hours=3)
            local_end = local_start + timedelta(minutes=duration_minutes)
            episode = _episode_id("legitimate_intense_activity", sensor.sensor_id, local_start)
            episodes.append(
                {
                    "EpisodeIdentifier": episode,
                    "SensorId": sensor.sensor_id,
                    "DeviceId": sensor.device_id,
                    "StartUtc": local_start.astimezone(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "EndUtc": local_end.astimezone(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "ScenarioType": "legitimate_intense_activity",
                }
            )
            elapsed = 0.0
            while elapsed < duration_minutes * 60:
                elapsed += float(rng.uniform(18.0, 45.0))
                if elapsed >= duration_minutes * 60:
                    break
                candidates.append(
                    _Candidate(
                        (local_start + timedelta(seconds=elapsed)).astimezone(timezone.utc),
                        sensor.sensor_id,
                        sensor.device_id,
                        float(
                            rng.uniform(
                                config.pulse_duration_min_seconds,
                                config.pulse_duration_max_seconds,
                            )
                        ),
                        config.rearm_seconds,
                    )
                )
    return candidates, episodes


def _outages(
    config: V2GeneratorConfig, rng: np.random.Generator
) -> list[dict[str, object]]:
    zone = ZoneInfo(config.local_timezone)
    start = config.start_local.replace(tzinfo=zone)
    rows: list[dict[str, object]] = []
    for sensor_index, sensor in enumerate(SENSORS):
        for sequence, day_offset in enumerate(range(9 + sensor_index, config.days, 24)):
            local_start = start + timedelta(
                days=day_offset,
                hours=(6 + sensor_index * 3 + sequence * 5) % 24,
                minutes=17,
            )
            local_end = local_start + timedelta(minutes=int(rng.integers(35, 121)))
            rows.append(
                {
                    "SensorId": sensor.sensor_id,
                    "DeviceId": sensor.device_id,
                    "UnavailableFromUtc": local_start.astimezone(timezone.utc),
                    "UnavailableToUtc": local_end.astimezone(timezone.utc),
                    "Reason": "simulated_telemetry_unavailability",
                }
            )
    return rows


def _unavailable(candidate: _Candidate, outages: list[dict[str, object]]) -> bool:
    return any(
        row["SensorId"] == candidate.sensor_id
        and row["UnavailableFromUtc"] <= candidate.occurred_at < row["UnavailableToUtc"]
        for row in outages
    )


def generate_v2_data(
    config: V2GeneratorConfig = V2GeneratorConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if config.days < 1:
        raise ValueError("the synthetic scenario must cover at least one day")
    rng = np.random.default_rng(config.seed)
    normal = _normal_candidates(config, rng)
    anomalies = _anomaly_candidates(config, rng)
    legitimate, legitimate_episodes = _legitimate_intense_candidates(config, rng)
    outages = _outages(config, rng)
    candidates = normal + anomalies + legitimate
    candidates.sort(key=lambda item: (item.sensor_id, item.occurred_at))

    accepted: list[_Candidate] = []
    next_available: dict[str, datetime] = {}
    for candidate in candidates:
        available_at = next_available.get(
            candidate.sensor_id, datetime.min.replace(tzinfo=timezone.utc)
        )
        if candidate.occurred_at < available_at:
            continue
        next_available[candidate.sensor_id] = candidate.occurred_at + timedelta(
            seconds=candidate.active_seconds + candidate.rearm_seconds
        )
        if not _unavailable(candidate, outages):
            accepted.append(candidate)
    accepted.sort(key=lambda item: item.occurred_at)

    event_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []
    for sequence, candidate in enumerate(accepted):
        identifier = str(
            uuid.uuid5(
                V2_NAMESPACE,
                f"{config.seed}:{candidate.sensor_id}:{candidate.occurred_at.isoformat()}:{sequence}",
            )
        )
        event_rows.append(
            {
                "EventIdentifier": identifier,
                "SensorId": candidate.sensor_id,
                "DeviceId": candidate.device_id,
                "OccurredAtUtc": candidate.occurred_at.isoformat().replace("+00:00", "Z"),
            }
        )
        if candidate.anomaly_type and candidate.episode_id:
            label_rows.append(
                {
                    "EventIdentifier": identifier,
                    "AnomalyType": candidate.anomaly_type,
                    "EpisodeIdentifier": candidate.episode_id,
                }
            )

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    labels = pd.DataFrame(label_rows, columns=V2_LABEL_COLUMNS)
    outage_frame = pd.DataFrame(outages, columns=OUTAGE_COLUMNS)
    for column in ("UnavailableFromUtc", "UnavailableToUtc"):
        outage_frame[column] = outage_frame[column].map(
            lambda value: value.isoformat().replace("+00:00", "Z")
        )
    legitimate_frame = pd.DataFrame(legitimate_episodes, columns=LEGITIMATE_COLUMNS)
    metadata = {
        "scenario": config.scenario,
        "days": config.days,
        "seed": config.seed,
        "start_local": config.start_local.isoformat(),
        "local_timezone": config.local_timezone,
        "sensor_ids": [sensor.sensor_id for sensor in SENSORS],
        "planned_anomaly_episodes_per_sensor": config.days,
        "normal_variation": (
            "sensor-specific periods, daily lognormal factor sigma=0.18, and a "
            "28-day sinusoidal factor with amplitude 0.12"
        ),
        "anomaly_schedule": (
            "one independent episode per sensor/day, alternating burst and "
            "electrical oscillation, with deterministic varied local times"
        ),
        "legitimate_intense_activity": (
            "unlabelled 10-20 minute high-activity episodes every seven days"
        ),
        "pulse_assumptions": {
            "normal_duration_seconds": [
                config.pulse_duration_min_seconds,
                config.pulse_duration_max_seconds,
            ],
            "normal_rearm_seconds": config.rearm_seconds,
            "electrical_duration_seconds": [
                config.electrical_pulse_min_seconds,
                config.electrical_pulse_max_seconds,
            ],
            "electrical_rearm_seconds": config.electrical_rearm_seconds,
        },
    }
    return events, labels, outage_frame, legitimate_frame, metadata


def write_v2_files(
    output_dir: Path, config: V2GeneratorConfig = V2GeneratorConfig()
) -> None:
    events, labels, outages, legitimate, metadata = generate_v2_data(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "sensor_events.csv", index=False, lineterminator="\n")
    labels.to_csv(output_dir / "anomaly_labels.csv", index=False, lineterminator="\n")
    outages.to_csv(output_dir / "telemetry_outages.csv", index=False, lineterminator="\n")
    legitimate.to_csv(
        output_dir / "legitimate_activity_episodes.csv", index=False, lineterminator="\n"
    )
    (output_dir / "scenario_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/synthetic_v2_development"),
    )
    parser.add_argument("--scenario", default="synthetic_v2_development")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--start-local", default="2026-02-01")
    args = parser.parse_args()
    write_v2_files(
        args.output_dir,
        V2GeneratorConfig(
            scenario=args.scenario,
            days=args.days,
            seed=args.seed,
            start_local=datetime.fromisoformat(args.start_local),
        ),
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
