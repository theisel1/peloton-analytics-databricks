# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - BI / Gold Layer
# MAGIC Build analytics views and run sample dashboard queries.

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

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

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

from peloton_databricks_pipeline.databricks_spark_loader import DatabricksSparkLoader
from peloton_databricks_pipeline.lakehouse import create_gold_views

loader = DatabricksSparkLoader(catalog=catalog, schema=schema, spark=spark)
create_gold_views(loader)

# COMMAND ----------

spark.sql(
    f"SELECT * FROM `{catalog}`.`{schema}`.`gold_peloton_daily_summary` ORDER BY workout_date DESC LIMIT 30"
).show(truncate=False)

spark.sql(
    f"SELECT * FROM `{catalog}`.`{schema}`.`gold_peloton_discipline_summary` ORDER BY workouts DESC"
).show(truncate=False)

spark.sql(
    f"SELECT * FROM `{catalog}`.`{schema}`.`gold_peloton_instructor_summary` ORDER BY workouts DESC LIMIT 20"
).show(truncate=False)
