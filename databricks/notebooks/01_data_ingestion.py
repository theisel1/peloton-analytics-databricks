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

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
peloton_since = dbutils.widgets.get("peloton_since")
peloton_max_workouts = dbutils.widgets.get("peloton_max_workouts")
peloton_username = dbutils.widgets.get("peloton_username")
peloton_password = dbutils.widgets.get("peloton_password")

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
if peloton_username:
    os.environ["PELOTON_USERNAME"] = peloton_username
if peloton_password:
    os.environ["PELOTON_PASSWORD"] = peloton_password

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
