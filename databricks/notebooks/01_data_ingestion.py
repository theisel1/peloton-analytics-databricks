# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Ingestion (Bronze + Silver)
# MAGIC Pull data from Peloton API and write directly to Delta (no local staging).

# COMMAND ----------

import os
import sys
from pathlib import Path

if "dbutils" not in globals():
    raise RuntimeError("This notebook is intended to run in Databricks.")

# COMMAND ----------

dbutils.widgets.text("repo_root", "/Workspace/Repos/<user>/peloton-analytics-databricks")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "fitness")
dbutils.widgets.text("peloton_since", "")
dbutils.widgets.text("peloton_max_workouts", "150")
dbutils.widgets.text("peloton_username", "")
dbutils.widgets.text("peloton_password", "")
dbutils.widgets.text("peloton_secret_scope", "peloton")
dbutils.widgets.text("peloton_username_key", "username")
dbutils.widgets.text("peloton_password_key", "password")

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
peloton_since = dbutils.widgets.get("peloton_since")
peloton_max_workouts = dbutils.widgets.get("peloton_max_workouts")
peloton_username = dbutils.widgets.get("peloton_username")
peloton_password = dbutils.widgets.get("peloton_password")
peloton_secret_scope = dbutils.widgets.get("peloton_secret_scope").strip()
peloton_username_key = dbutils.widgets.get("peloton_username_key").strip()
peloton_password_key = dbutils.widgets.get("peloton_password_key").strip()

# COMMAND ----------

def _normalize_workspace_root(path: str) -> str:
    if path.startswith("/Workspace/"):
        return path
    if path.startswith("/Users/") or path.startswith("/Repos/"):
        return f"/Workspace{path}"
    return path

repo_root = _normalize_workspace_root(repo_root)
src_path = str(Path(repo_root) / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ["DATABRICKS_CATALOG"] = catalog
os.environ["DATABRICKS_SCHEMA"] = schema
os.environ["USE_DATABRICKS_SPARK"] = "true"
os.environ["WRITE_LOCAL_STAGING"] = "false"

if peloton_since:
    os.environ["PELOTON_SINCE"] = peloton_since
if peloton_max_workouts:
    os.environ["PELOTON_MAX_WORKOUTS"] = peloton_max_workouts


def _resolve_secret(scope: str, key: str, label: str) -> str | None:
    if not scope or not key:
        return None
    try:
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve {label} from secret scope '{scope}' using key '{key}'."
        ) from exc


resolved_username = peloton_username or _resolve_secret(
    peloton_secret_scope, peloton_username_key, "Peloton username"
)
resolved_password = peloton_password or _resolve_secret(
    peloton_secret_scope, peloton_password_key, "Peloton password"
)

if not resolved_username or not resolved_password:
    raise ValueError(
        "Peloton credentials are required. Pass peloton_username/peloton_password "
        "or configure peloton_secret_scope + secret keys."
    )

os.environ["PELOTON_USERNAME"] = resolved_username
os.environ["PELOTON_PASSWORD"] = resolved_password

# COMMAND ----------

from peloton_databricks_pipeline.databricks_spark_loader import DatabricksSparkLoader
from peloton_databricks_pipeline.lakehouse import ingest_to_lakehouse
from peloton_databricks_pipeline.pipeline import extract_with_raw_payloads

workouts_raw, performance_raw, workouts_df, metrics_df, _ = extract_with_raw_payloads(write_local_staging=False)

loader = DatabricksSparkLoader(catalog=catalog, schema=schema, spark=spark)
result = ingest_to_lakehouse(
    loader=loader,
    workouts_raw=workouts_raw,
    performance_raw=performance_raw,
    workouts_df=workouts_df,
    metrics_df=metrics_df,
)

print(result)

# COMMAND ----------

spark.sql(
    f"""
    SELECT
      (SELECT COUNT(*) FROM `{catalog}`.`{schema}`.`bronze_peloton_workouts_raw`) AS bronze_workouts,
      (SELECT COUNT(*) FROM `{catalog}`.`{schema}`.`bronze_peloton_metrics_raw`) AS bronze_metrics,
      (SELECT COUNT(*) FROM `{catalog}`.`{schema}`.`peloton_workouts`) AS silver_workouts,
      (SELECT COUNT(*) FROM `{catalog}`.`{schema}`.`peloton_metrics`) AS silver_metrics
    """
).show(truncate=False)
