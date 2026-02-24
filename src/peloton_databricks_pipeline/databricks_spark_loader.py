from __future__ import annotations

from typing import Iterable
from uuid import uuid4

import pandas as pd
from pyspark.sql import SparkSession


class DatabricksSparkLoader:
    """Databricks-native loader that writes directly to Delta tables via Spark SQL."""

    def __init__(self, catalog: str, schema: str, spark: SparkSession | None = None) -> None:
        self.catalog = catalog
        self.schema = schema
        self.spark = spark or SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

    def _q(self, identifier: str) -> str:
        safe = identifier.replace("`", "``")
        return f"`{safe}`"

    def _table_name(self, table: str) -> str:
        return f"{self._q(self.catalog)}.{self._q(self.schema)}.{self._q(table)}"

    def table_name(self, table: str) -> str:
        return self._table_name(table)

    def ensure_schema_and_tables(self) -> None:
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self._q(self.catalog)}.{self._q(self.schema)}")

        workouts_table = self._table_name("peloton_workouts")
        metrics_table = self._table_name("peloton_metrics")

        self.spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {workouts_table} (
                workout_id STRING,
                status STRING,
                fitness_discipline STRING,
                created_at TIMESTAMP,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_work DOUBLE,
                distance DOUBLE,
                calories DOUBLE,
                device_type STRING,
                ride_id STRING,
                ride_title STRING,
                ride_duration DOUBLE,
                instructor_name STRING
            ) USING DELTA
            """
        )

        self.spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {metrics_table} (
                workout_id STRING,
                metric_name STRING,
                average_value DOUBLE,
                max_value DOUBLE,
                sample_count BIGINT
            ) USING DELTA
            """
        )

    def upsert_dataframe(self, table: str, df: pd.DataFrame, key_columns: Iterable[str]) -> None:
        if df.empty:
            return

        prepared = df.where(pd.notna(df), None)
        spark_df = self.spark.createDataFrame(prepared)

        staging_view = f"peloton_staging_{uuid4().hex[:8]}"
        spark_df.createOrReplaceTempView(staging_view)

        target_table = self._table_name(table)
        columns = spark_df.columns

        on_clause = " AND ".join([f"t.{self._q(col)} <=> s.{self._q(col)}" for col in key_columns])
        update_clause = ", ".join([f"t.{self._q(col)} = s.{self._q(col)}" for col in columns])
        insert_columns = ", ".join([self._q(col) for col in columns])
        insert_values = ", ".join([f"s.{self._q(col)}" for col in columns])

        self.spark.sql(
            f"""
            MERGE INTO {target_table} t
            USING {self._q(staging_view)} s
            ON {on_clause}
            WHEN MATCHED THEN UPDATE SET {update_clause}
            WHEN NOT MATCHED THEN INSERT ({insert_columns}) VALUES ({insert_values})
            """
        )

        self.spark.catalog.dropTempView(staging_view)

    def load(self, workouts_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
        self.ensure_schema_and_tables()
        self.upsert_dataframe("peloton_workouts", workouts_df, key_columns=["workout_id"])
        self.upsert_dataframe("peloton_metrics", metrics_df, key_columns=["workout_id", "metric_name"])

    def read_training_frame(self) -> pd.DataFrame:
        workouts_table = self._table_name("peloton_workouts")
        metrics_table = self._table_name("peloton_metrics")

        query = f"""
        WITH metric_agg AS (
            SELECT
                workout_id,
                MAX(CASE WHEN metric_name = 'cadence' THEN average_value END) AS avg_cadence,
                MAX(CASE WHEN metric_name = 'heart_rate' THEN average_value END) AS avg_heart_rate,
                MAX(CASE WHEN metric_name = 'output' THEN average_value END) AS avg_output,
                MAX(CASE WHEN metric_name = 'resistance' THEN average_value END) AS avg_resistance,
                MAX(CASE WHEN metric_name = 'speed' THEN average_value END) AS avg_speed
            FROM {metrics_table}
            GROUP BY workout_id
        )
        SELECT
            w.*, m.avg_cadence, m.avg_heart_rate, m.avg_output, m.avg_resistance, m.avg_speed
        FROM {workouts_table} w
        LEFT JOIN metric_agg m ON w.workout_id = m.workout_id
        """

        return self.spark.sql(query).toPandas()
