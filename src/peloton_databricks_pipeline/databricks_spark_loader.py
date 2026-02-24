from __future__ import annotations

from datetime import date, datetime
from typing import Iterable
from uuid import uuid4

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)


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

    def _coerce_for_spark_type(self, value: object | None, data_type: object) -> object | None:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        try:
            if isinstance(data_type, StringType):
                return str(value)
            if isinstance(data_type, (DoubleType, FloatType, DecimalType)):
                return float(value)
            if isinstance(data_type, (LongType, IntegerType, ShortType, ByteType)):
                return int(value)
            if isinstance(data_type, BooleanType):
                return bool(value)
            if isinstance(data_type, TimestampType):
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(data_type, DateType):
                if isinstance(value, date):
                    return value
                if isinstance(value, datetime):
                    return value.date()
                if isinstance(value, str):
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except Exception:
            return value

        return value

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

        target_table = self._table_name(table)
        target_schema = self.spark.table(target_table).schema

        # Spark Connect + pandas can fail on Arrow-backed columns (ChunkedArray).
        # Convert rows to plain Python scalars before creating a Spark DataFrame.
        records: list[tuple[object | None, ...]] = []
        for row in df.to_dict(orient="records"):
            coerced_row = tuple(
                self._coerce_for_spark_type(row.get(field.name), field.dataType) for field in target_schema.fields
            )
            records.append(coerced_row)

        if not records:
            return

        spark_df = self.spark.createDataFrame(records, schema=target_schema)

        staging_view = f"peloton_staging_{uuid4().hex[:8]}"
        spark_df.createOrReplaceTempView(staging_view)

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
