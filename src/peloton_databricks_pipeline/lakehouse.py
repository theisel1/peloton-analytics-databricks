from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .databricks_spark_loader import DatabricksSparkLoader


@dataclass
class LakehouseIngestionResult:
    workouts_raw_rows: int
    metrics_raw_rows: int
    workouts_rows: int
    metrics_rows: int


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _to_iso_utc(epoch_seconds: int | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _q(identifier: str) -> str:
    safe = identifier.replace("`", "``")
    return f"`{safe}`"


def _build_workouts_bronze_df(workouts_raw: list[dict[str, Any]]) -> pd.DataFrame:
    ingested_at = _utc_now_iso()
    rows = []
    for workout in workouts_raw:
        workout_id = workout.get("id")
        if not workout_id:
            continue
        rows.append(
            {
                "workout_id": str(workout_id),
                "workout_created_at": _to_iso_utc(workout.get("created_at")),
                "source_payload": json.dumps(workout, separators=(",", ":")),
                "ingested_at": ingested_at,
            }
        )
    return pd.DataFrame(rows)


def _build_metrics_bronze_df(performance_raw: list[dict[str, Any]]) -> pd.DataFrame:
    ingested_at = _utc_now_iso()
    rows = []
    for item in performance_raw:
        workout_id = item.get("workout_id")
        if not workout_id:
            continue
        rows.append(
            {
                "workout_id": str(workout_id),
                "source_payload": json.dumps(item.get("performance") or {}, separators=(",", ":")),
                "ingested_at": ingested_at,
            }
        )
    return pd.DataFrame(rows)


def ensure_lakehouse_objects(loader: "DatabricksSparkLoader") -> None:
    loader.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_q(loader.catalog)}.{_q(loader.schema)}")

    bronze_workouts = loader.table_name("bronze_peloton_workouts_raw")
    bronze_metrics = loader.table_name("bronze_peloton_metrics_raw")

    loader.spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {bronze_workouts} (
            workout_id STRING,
            workout_created_at TIMESTAMP,
            source_payload STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
        """
    )

    loader.spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {bronze_metrics} (
            workout_id STRING,
            source_payload STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
        """
    )

    loader.ensure_schema_and_tables()

    silver_workouts_view = loader.table_name("silver_peloton_workouts")
    silver_metrics_view = loader.table_name("silver_peloton_metrics")
    workouts_table = loader.table_name("peloton_workouts")
    metrics_table = loader.table_name("peloton_metrics")

    loader.spark.sql(f"CREATE OR REPLACE VIEW {silver_workouts_view} AS SELECT * FROM {workouts_table}")
    loader.spark.sql(f"CREATE OR REPLACE VIEW {silver_metrics_view} AS SELECT * FROM {metrics_table}")


def create_gold_views(loader: "DatabricksSparkLoader") -> None:
    workouts_table = loader.table_name("peloton_workouts")
    gold_daily = loader.table_name("gold_peloton_daily_summary")
    gold_discipline = loader.table_name("gold_peloton_discipline_summary")
    gold_instructor = loader.table_name("gold_peloton_instructor_summary")

    loader.spark.sql(
        f"""
        CREATE OR REPLACE VIEW {gold_daily} AS
        SELECT
            DATE(start_time) AS workout_date,
            COUNT(*) AS workout_count,
            SUM(calories) AS total_calories,
            AVG(total_work) AS avg_total_work,
            AVG(ride_duration) AS avg_duration
        FROM {workouts_table}
        GROUP BY DATE(start_time)
        """
    )

    loader.spark.sql(
        f"""
        CREATE OR REPLACE VIEW {gold_discipline} AS
        SELECT
            fitness_discipline,
            COUNT(*) AS workouts,
            AVG(calories) AS avg_calories,
            AVG(total_work) AS avg_total_work,
            AVG(ride_duration) AS avg_duration
        FROM {workouts_table}
        GROUP BY fitness_discipline
        """
    )

    loader.spark.sql(
        f"""
        CREATE OR REPLACE VIEW {gold_instructor} AS
        SELECT
            instructor_name,
            COUNT(*) AS workouts,
            AVG(total_work) AS avg_total_work,
            AVG(calories) AS avg_calories
        FROM {workouts_table}
        WHERE instructor_name IS NOT NULL
        GROUP BY instructor_name
        """
    )


def ingest_to_lakehouse(
    loader: "DatabricksSparkLoader",
    workouts_raw: list[dict[str, Any]],
    performance_raw: list[dict[str, Any]],
    workouts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> LakehouseIngestionResult:
    ensure_lakehouse_objects(loader)

    workouts_bronze_df = _build_workouts_bronze_df(workouts_raw)
    metrics_bronze_df = _build_metrics_bronze_df(performance_raw)

    loader.upsert_dataframe("bronze_peloton_workouts_raw", workouts_bronze_df, key_columns=["workout_id"])
    loader.upsert_dataframe("bronze_peloton_metrics_raw", metrics_bronze_df, key_columns=["workout_id"])

    loader.load(workouts_df=workouts_df, metrics_df=metrics_df)
    create_gold_views(loader)

    return LakehouseIngestionResult(
        workouts_raw_rows=len(workouts_bronze_df),
        metrics_raw_rows=len(metrics_bronze_df),
        workouts_rows=len(workouts_df),
        metrics_rows=len(metrics_df),
    )
