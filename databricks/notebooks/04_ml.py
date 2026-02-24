# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - ML
# MAGIC Train model from Silver tables and persist artifacts to DBFS.

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
default_mlflow_experiment = f"/Users/{spark.sql('SELECT current_user() AS user').first()['user']}/peloton-analytics"
dbutils.widgets.text("mlflow_enabled", "true")
dbutils.widgets.text("mlflow_experiment_name", default_mlflow_experiment)
dbutils.widgets.text("mlflow_run_name", "peloton-ml-training")
dbutils.widgets.text("mlflow_registered_model_name", "")

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
artifact_base_path = dbutils.widgets.get("artifact_base_path")
mlflow_enabled = dbutils.widgets.get("mlflow_enabled").strip()
mlflow_experiment_name = dbutils.widgets.get("mlflow_experiment_name").strip()
mlflow_run_name = dbutils.widgets.get("mlflow_run_name").strip()
mlflow_registered_model_name = dbutils.widgets.get("mlflow_registered_model_name").strip()

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
os.environ["DATABRICKS_ARTIFACT_BASE_PATH"] = artifact_base_path
if mlflow_enabled:
    os.environ["MLFLOW_ENABLED"] = mlflow_enabled
if mlflow_experiment_name:
    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name
if mlflow_run_name:
    os.environ["MLFLOW_RUN_NAME"] = mlflow_run_name
if mlflow_registered_model_name:
    os.environ["MLFLOW_REGISTERED_MODEL_NAME"] = mlflow_registered_model_name

from peloton_databricks_pipeline.pipeline import run_train

results = run_train(use_spark_loader=True, model_base_path=artifact_base_path)
print(results)

# COMMAND ----------

spark.sql(
    f"SELECT * FROM `{catalog}`.`{schema}`.`gold_peloton_discipline_summary` ORDER BY workouts DESC"
).show(truncate=False)
