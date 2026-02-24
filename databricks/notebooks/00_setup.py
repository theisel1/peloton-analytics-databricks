# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup
# MAGIC Configure catalog/schema parameters for the Peloton Lakehouse demo pipeline.

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
dbutils.widgets.text("artifact_base_path", "/dbfs/FileStore/peloton_analytics")
dbutils.widgets.text("peloton_since", "")
dbutils.widgets.text("peloton_max_workouts", "150")

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
artifact_base_path = dbutils.widgets.get("artifact_base_path")
peloton_since = dbutils.widgets.get("peloton_since")
peloton_max_workouts = dbutils.widgets.get("peloton_max_workouts")

# COMMAND ----------

src_path = str(Path(repo_root) / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ["DATABRICKS_CATALOG"] = catalog
os.environ["DATABRICKS_SCHEMA"] = schema
os.environ["USE_DATABRICKS_SPARK"] = "true"
os.environ["WRITE_LOCAL_STAGING"] = "false"
os.environ["DATABRICKS_ARTIFACT_BASE_PATH"] = artifact_base_path

if peloton_since:
    os.environ["PELOTON_SINCE"] = peloton_since
if peloton_max_workouts:
    os.environ["PELOTON_MAX_WORKOUTS"] = peloton_max_workouts

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")

print(
    {
        "repo_root": repo_root,
        "catalog": catalog,
        "schema": schema,
        "artifact_base_path": artifact_base_path,
        "use_databricks_spark": os.environ["USE_DATABRICKS_SPARK"],
        "write_local_staging": os.environ["WRITE_LOCAL_STAGING"],
    }
)
