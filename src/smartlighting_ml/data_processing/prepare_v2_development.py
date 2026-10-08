"""Prepare all 120 v2 days as development data without creating a test set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from smartlighting_ml.data_processing.preparation_pipeline import (
    _add_features,
    _build_window_grid,
    _map_anomaly_windows,
    _mark_reliability,
    load_inputs,
)


@dataclass(frozen=True)
class V2PreparationConfig:
    raw_dir: Path = Path("data/raw/synthetic_v2_development")
    output_dir: Path = Path("data/processed/v2_development")
    local_timezone: str = "America/Sao_Paulo"
    window_minutes: int = 5


def prepare_v2_development(config: V2PreparationConfig = V2PreparationConfig()):
    events, outages, labels = load_inputs(config.raw_dir)
    windows = _build_window_grid(
        events, config.window_minutes, config.local_timezone
    )
    windows = _mark_reliability(windows, outages, config.window_minutes)
    windows = _add_features(windows, config.local_timezone)
    windows["split"] = "development"
    anomaly_windows = _map_anomaly_windows(events, labels, config.window_minutes)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(config.output_dir / "development_windows.csv", index=False)
    anomaly_windows.to_csv(
        config.output_dir / "anomaly_windows_evaluation.csv", index=False
    )
    return windows, anomaly_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/synthetic_v2_development")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/v2_development")
    )
    args = parser.parse_args()
    windows, anomaly_windows = prepare_v2_development(
        V2PreparationConfig(raw_dir=args.raw_dir, output_dir=args.output_dir)
    )
    print(f"windows={len(windows)}")
    print(f"unreliable={int((~windows['telemetry_reliable']).sum())}")
    print(
        "episodes="
        f"{anomaly_windows['EpisodeIdentifier'].nunique() if 'EpisodeIdentifier' in anomaly_windows else 0}"
    )


if __name__ == "__main__":
    main()
