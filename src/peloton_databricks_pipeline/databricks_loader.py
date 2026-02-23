from __future__ import annotations

import math
import time
from numbers import Real
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd
import requests


class DatabricksLoader:
    def __init__(self, server_hostname: str, http_path: str, access_token: str, catalog: str, schema: str) -> None:
        self.server_hostname = server_hostname
        self.http_path = http_path
        self.access_token = access_token
        self.catalog = catalog
        self.schema = schema
        self.warehouse_id = self.http_path.rstrip("/").split("/")[-1]

    def _statement_endpoint(self) -> str:
        return f"https://{self.server_hostname}/api/2.0/sql/statements"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _execute_statement(self, statement: str, wait_timeout: str = "30s") -> dict[str, Any]:
        submit_response = requests.post(
            self._statement_endpoint(),
            headers=self._headers(),
            json={
                "statement": statement,
                "warehouse_id": self.warehouse_id,
                "wait_timeout": wait_timeout,
                "format": "JSON_ARRAY",
            },
            timeout=60,
        )
        if submit_response.status_code >= 400:
            snippet = submit_response.text[:1000]
            raise RuntimeError(f"Databricks SQL submit failed ({submit_response.status_code}): {snippet}")
        payload = submit_response.json()

        statement_id = payload.get("statement_id")
        if not statement_id:
            raise RuntimeError(f"Databricks SQL statement did not return statement_id: {payload}")

        state = (payload.get("status") or {}).get("state")
        while state in {"PENDING", "RUNNING"}:
            time.sleep(2)
            poll_response = requests.get(
                f"{self._statement_endpoint()}/{statement_id}",
                headers=self._headers(),
                timeout=60,
            )
            poll_response.raise_for_status()
            payload = poll_response.json()
            state = (payload.get("status") or {}).get("state")

        if state != "SUCCEEDED":
            err = (payload.get("status") or {}).get("error") or {}
            message = err.get("message") or str(payload.get("status"))
            raise RuntimeError(f"Databricks SQL failed: {message}")

        return payload

    def _query_rows(self, statement: str) -> tuple[list[str], list[list[Any]]]:
        payload = self._execute_statement(statement)
        manifest = payload.get("manifest") or {}
        schema = manifest.get("schema") or {}
        columns_meta = schema.get("columns") or []
        columns = [col.get("name", f"col_{idx}") for idx, col in enumerate(columns_meta)]

        result = payload.get("result") or {}
        rows = result.get("data_array") or []

        if not columns and rows:
            columns = [f"col_{idx}" for idx in range(len(rows[0]))]

        return columns, rows

    def _choose_catalog(self) -> str:
        try:
            _, rows = self._query_rows("SHOW CATALOGS")
        except Exception:
            return self.catalog

        catalogs = [str(row[0]) for row in rows if row and row[0]]
        if self.catalog in catalogs:
            return self.catalog
        if "hive_metastore" in catalogs:
            return "hive_metastore"
        if catalogs:
            return catalogs[0]
        return self.catalog

    def ensure_schema_and_tables(self) -> None:
        self.catalog = self._choose_catalog()
        statements = [
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
            ) USING DELTA
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.catalog}.{self.schema}.peloton_metrics (
                workout_id STRING,
                metric_name STRING,
                average_value DOUBLE,
                max_value DOUBLE,
                sample_count BIGINT
            ) USING DELTA
            """,
        ]

        for statement in statements:
            self._execute_statement(statement)

    def _sql_literal(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, Real):
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return "NULL"
            return str(value)

        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    def _insert_dataframe(self, table: str, df: pd.DataFrame, key_columns: Iterable[str], batch_size: int = 100) -> None:
        if df.empty:
            return

        columns = df.columns.tolist()
        column_list = ", ".join(columns)
        key_predicate = " AND ".join([f"t.{col} <=> s.{col}" for col in key_columns])

        staging_table = f"{table}_staging_{uuid4().hex[:8]}"

        records = [tuple(x) for x in df.itertuples(index=False, name=None)]

        try:
            self._execute_statement(f"CREATE TABLE {staging_table} AS SELECT * FROM {table} WHERE 1 = 0")

            for idx in range(0, len(records), batch_size):
                batch = records[idx : idx + batch_size]
                values_sql = ", ".join(
                    "(" + ", ".join(self._sql_literal(value) for value in row) + ")" for row in batch
                )
                insert_sql = f"INSERT INTO {staging_table} ({column_list}) VALUES {values_sql}"
                self._execute_statement(insert_sql, wait_timeout="50s")

            merge_sql = f"""
            MERGE INTO {table} t
            USING {staging_table} s
            ON {key_predicate}
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
            self._execute_statement(merge_sql, wait_timeout="50s")
        finally:
            self._execute_statement(f"DROP TABLE IF EXISTS {staging_table}")

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

        columns, rows = self._query_rows(query)
        return pd.DataFrame(rows, columns=columns)
