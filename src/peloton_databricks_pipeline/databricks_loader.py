from __future__ import annotations

from typing import Iterable

import pandas as pd
from databricks import sql


class DatabricksLoader:
    def __init__(self, server_hostname: str, http_path: str, access_token: str, catalog: str, schema: str) -> None:
        self.server_hostname = server_hostname
        self.http_path = http_path
        self.access_token = access_token
        self.catalog = catalog
        self.schema = schema

    def _connect(self):
        return sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
        )

    def ensure_schema_and_tables(self) -> None:
        statements = [
            f"CREATE CATALOG IF NOT EXISTS {self.catalog}",
            f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}",
            f"""
            CREATE TABLE IF NOT EXISTS {self.catalog}.{self.schema}.peloton_workouts (
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
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.catalog}.{self.schema}.peloton_metrics (
                workout_id STRING,
                metric_name STRING,
                average_value DOUBLE,
                max_value DOUBLE,
                sample_count BIGINT
            )
            """,
        ]

        with self._connect() as conn:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    def _insert_dataframe(self, table: str, df: pd.DataFrame, key_columns: Iterable[str]) -> None:
        if df.empty:
            return

        columns = df.columns.tolist()
        column_list = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))

        key_predicate = " AND ".join([f"t.{col} <=> s.{col}" for col in key_columns])

        # Insert into temp table and merge for idempotency.
        temp_table = f"{table}_staging"

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
                cursor.execute(f"CREATE TABLE {temp_table} AS SELECT * FROM {table} WHERE 1 = 0")
                insert_sql = f"INSERT INTO {temp_table} ({column_list}) VALUES ({placeholders})"
                cursor.executemany(insert_sql, [tuple(x) for x in df.itertuples(index=False, name=None)])

                merge_sql = f"""
                MERGE INTO {table} t
                USING {temp_table} s
                ON {key_predicate}
                WHEN MATCHED THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
                """
                cursor.execute(merge_sql)
                cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")

    def load(self, workouts_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
        self.ensure_schema_and_tables()
        workouts_table = f"{self.catalog}.{self.schema}.peloton_workouts"
        metrics_table = f"{self.catalog}.{self.schema}.peloton_metrics"
        self._insert_dataframe(workouts_table, workouts_df, key_columns=["workout_id"])
        self._insert_dataframe(metrics_table, metrics_df, key_columns=["workout_id", "metric_name"])

    def read_training_frame(self) -> pd.DataFrame:
        query = f"""
        WITH metric_agg AS (
            SELECT
                workout_id,
                MAX(CASE WHEN metric_name = 'cadence' THEN average_value END) AS avg_cadence,
                MAX(CASE WHEN metric_name = 'heart_rate' THEN average_value END) AS avg_heart_rate,
                MAX(CASE WHEN metric_name = 'output' THEN average_value END) AS avg_output,
                MAX(CASE WHEN metric_name = 'resistance' THEN average_value END) AS avg_resistance,
                MAX(CASE WHEN metric_name = 'speed' THEN average_value END) AS avg_speed
            FROM {self.catalog}.{self.schema}.peloton_metrics
            GROUP BY workout_id
        )
        SELECT
            w.*, m.avg_cadence, m.avg_heart_rate, m.avg_output, m.avg_resistance, m.avg_speed
        FROM {self.catalog}.{self.schema}.peloton_workouts w
        LEFT JOIN metric_agg m ON w.workout_id = m.workout_id
        """

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]

        return pd.DataFrame(rows, columns=columns)
