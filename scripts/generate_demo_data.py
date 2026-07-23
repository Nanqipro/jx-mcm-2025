#!/usr/bin/env python3
"""Generate deterministic, synthetic data for repository smoke tests.

The generated values resemble the shape of the competition inputs, but they
do not reproduce or disclose any row from the original dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 2025


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _state(index: int, count: int, rng: random.Random) -> dict[str, float]:
    """Create one internally consistent synthetic weather/visibility state."""
    x = index / max(count - 1, 1)
    fog = 0.85 * math.exp(-((x - 0.48) / 0.18) ** 2)
    fog += 0.28 * math.exp(-((x - 0.82) / 0.09) ** 2)
    fog = _clip(fog + rng.gauss(0, 0.018), 0.0, 1.0)

    temperature = 11.5 + 4.2 * math.sin(2 * math.pi * (x - 0.2))
    temperature += rng.gauss(0, 0.12)
    humidity = _clip(54 + 43 * fog - 0.7 * (temperature - 11.5), 35, 100)
    dewpoint = temperature - (100 - humidity) / 5.0
    wind_speed = _clip(2.8 - 1.9 * fog + rng.gauss(0, 0.12), 0.1, 5.0)
    wind_direction = (210 + 55 * math.sin(4 * math.pi * x)) % 360
    pressure = 1014.2 + 1.8 * math.cos(2 * math.pi * x)
    visibility = _clip(9400 - 8100 * fog + rng.gauss(0, 90), 350, 10000)
    blur_index = _clip(0.08 + 0.84 * fog + rng.gauss(0, 0.012), 0.03, 0.98)

    return {
        "temperature": temperature,
        "humidity": humidity,
        "dewpoint": dewpoint,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "pressure": pressure,
        "visibility": visibility,
        "blur_index": blur_index,
    }


def generate_demo_data(output_dir: Path, rows: int = 360) -> tuple[Path, Path]:
    """Write the two CSV files used by the documented demo commands."""
    if rows < 60:
        raise ValueError("rows must be at least 60")

    output_dir.mkdir(parents=True, exist_ok=True)
    blur_path = output_dir / "blur.csv"
    synced_path = output_dir / "complete_synced_data.csv"
    rng = random.Random(SEED)
    start = datetime(2020, 3, 12, 8, 0)

    blur_fields = [
        "CREATEDATE", "MOR_1A", "RVR_1A", "TEMP", "RH", "DEWPOINT",
        "WS2A", "WD2A", "CW2A", "PAINS (HPA)", "VIS1K", "LIGHTS",
        "BL1A", "TREND", "laplacian_var", "sobel_mean", "brenner",
        "std_contrast", "blur_index",
    ]
    synced_fields = [
        "laplacian_var", "sobel_mean", "sobel_std", "contrast_std",
        "high_freq_ratio", "edge_density", "entropy", "local_var_mean",
        "brenner", "timestamp", "frame_number", "time_offset_seconds",
        "visibility_vis_1a", "visibility_vis_10a", "visibility_mor_raw",
        "visibility_vis_raw", "weather_pressure_hpa", "weather_qnh_hpa",
        "weather_temperature_c", "weather_humidity_pct", "weather_dewpoint_c",
        "wind_wind_speed_2m", "wind_wind_speed_10m", "wind_wind_dir_2m",
        "wind_wind_dir_10m", "wind_gust_speed", "vis_time_diff",
        "ptu_time_diff", "wind_time_diff",
    ]

    states = [_state(i, rows, rng) for i in range(rows)]

    with blur_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=blur_fields)
        writer.writeheader()
        previous_visibility = states[0]["visibility"]
        for index, state in enumerate(states):
            timestamp = start + timedelta(minutes=index)
            visibility = state["visibility"]
            blur = state["blur_index"]
            writer.writerow({
                "CREATEDATE": timestamp.isoformat(sep=" "),
                "MOR_1A": round(visibility, 2),
                "RVR_1A": round(min(visibility, 2000), 2),
                "TEMP": round(state["temperature"], 3),
                "RH": round(state["humidity"], 3),
                "DEWPOINT": round(state["dewpoint"], 3),
                "WS2A": round(state["wind_speed"], 3),
                "WD2A": round(state["wind_direction"], 3),
                "CW2A": round(state["wind_speed"] * 0.2, 3),
                "PAINS (HPA)": round(state["pressure"], 3),
                "VIS1K": round(visibility / 1000, 4),
                "LIGHTS": 100 if visibility < 1500 else 0,
                "BL1A": round(0.55 * visibility, 3),
                "TREND": round(visibility - previous_visibility, 3),
                "laplacian_var": round(90 + 620 * (1 - blur), 4),
                "sobel_mean": round(5 + 23 * (1 - blur), 4),
                "brenner": round(2e5 + 2.1e6 * (1 - blur), 3),
                "std_contrast": round(5 + 28 * (1 - blur), 4),
                "blur_index": round(blur, 6),
            })
            previous_visibility = visibility

    with synced_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=synced_fields)
        writer.writeheader()
        for index, state in enumerate(states):
            timestamp = start + timedelta(seconds=15 * index)
            blur = state["blur_index"]
            clarity = 1 - blur
            visibility = state["visibility"]
            writer.writerow({
                "laplacian_var": round(90 + 620 * clarity, 4),
                "sobel_mean": round(5 + 23 * clarity, 4),
                "sobel_std": round(18 + 52 * clarity, 4),
                "contrast_std": round(5 + 28 * clarity, 4),
                "high_freq_ratio": round(0.08 + 0.42 * clarity, 6),
                "edge_density": round(0.004 + 0.032 * clarity, 6),
                "entropy": round(4.5 + 1.6 * clarity, 5),
                "local_var_mean": round(8 + 52 * clarity, 4),
                "brenner": round(2e5 + 2.1e6 * clarity, 3),
                "timestamp": timestamp.isoformat(sep=" "),
                "frame_number": index * 375,
                "time_offset_seconds": index * 15,
                "visibility_vis_1a": round(visibility * 0.98, 2),
                "visibility_vis_10a": round(visibility * 1.01, 2),
                "visibility_mor_raw": round(visibility, 2),
                "visibility_vis_raw": round(visibility * 0.995, 2),
                "weather_pressure_hpa": round(state["pressure"], 3),
                "weather_qnh_hpa": round(state["pressure"] + 1.5, 3),
                "weather_temperature_c": round(state["temperature"], 3),
                "weather_humidity_pct": round(state["humidity"], 3),
                "weather_dewpoint_c": round(state["dewpoint"], 3),
                "wind_wind_speed_2m": round(state["wind_speed"], 3),
                "wind_wind_speed_10m": round(state["wind_speed"] * 1.18, 3),
                "wind_wind_dir_2m": round(state["wind_direction"], 3),
                "wind_wind_dir_10m": round((state["wind_direction"] + 8) % 360, 3),
                "wind_gust_speed": round(state["wind_speed"] * 1.5, 3),
                "vis_time_diff": 0,
                "ptu_time_diff": 0,
                "wind_time_diff": 0,
            })

    return blur_path, synced_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "demo")
    parser.add_argument("--rows", type=int, default=360)
    args = parser.parse_args()
    blur_path, synced_path = generate_demo_data(args.output_dir, args.rows)
    print(f"Generated: {blur_path}")
    print(f"Generated: {synced_path}")


if __name__ == "__main__":
    main()
