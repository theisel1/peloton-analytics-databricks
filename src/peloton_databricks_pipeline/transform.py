from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def _iso_utc(epoch_seconds: int | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat()


def workouts_to_dataframe(workouts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for workout in workouts:
        ride = workout.get("ride") or {}
        instructor = ride.get("instructor") or {}

        rows.append(
            {
                "workout_id": workout.get("id"),
                "status": workout.get("status"),
                "fitness_discipline": workout.get("fitness_discipline"),
                "created_at": _iso_utc(workout.get("created_at")),
                "start_time": _iso_utc(workout.get("start_time")),
                "end_time": _iso_utc(workout.get("end_time")),
                "total_work": workout.get("total_work"),
                "distance": workout.get("distance"),
                "calories": workout.get("total_calories"),
                "device_type": workout.get("device_type"),
                "ride_id": ride.get("id"),
                "ride_title": ride.get("title"),
                "ride_duration": ride.get("duration"),
                "instructor_name": instructor.get("name"),
            }
        )

    return pd.DataFrame(rows)


def performance_to_dataframe(workout_id: str, performance: dict[str, Any]) -> pd.DataFrame:
    metrics = performance.get("metrics", [])
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        name = metric.get("slug") or metric.get("display_name")
        values = metric.get("values") or []
        average_value = metric.get("average_value")
        max_value = metric.get("max_value")

        rows.append(
            {
                "workout_id": workout_id,
                "metric_name": name,
                "average_value": average_value,
                "max_value": max_value,
                "sample_count": len(values),
            }
        )

    return pd.DataFrame(rows)


def aggregate_metrics(performance_df: pd.DataFrame) -> pd.DataFrame:
    if performance_df.empty:
        return performance_df

    pivot = (
        performance_df.pivot_table(
            index="workout_id",
            columns="metric_name",
            values="average_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    rename_map = {
        "cadence": "avg_cadence",
        "heart_rate": "avg_heart_rate",
        "output": "avg_output",
        "resistance": "avg_resistance",
        "speed": "avg_speed",
    }
    return pivot.rename(columns=rename_map)


def build_training_frame(workouts_df: pd.DataFrame, metrics_agg_df: pd.DataFrame) -> pd.DataFrame:
    if workouts_df.empty:
        return workouts_df

    merged = workouts_df.merge(metrics_agg_df, on="workout_id", how="left")

    numeric_columns = [
        "total_work",
        "distance",
        "calories",
        "ride_duration",
        "avg_cadence",
        "avg_heart_rate",
        "avg_output",
        "avg_resistance",
        "avg_speed",
    ]

    for column in numeric_columns:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    return merged
