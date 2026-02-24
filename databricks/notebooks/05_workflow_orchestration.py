# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Workflow Orchestration
# MAGIC Single-notebook orchestration entrypoint (setup + ingestion + governance + BI + ML).

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

from peloton_databricks_pipeline.pipeline import run_lakehouse

results = run_lakehouse(write_local_staging=False, model_base_path=artifact_base_path)
print(results)
