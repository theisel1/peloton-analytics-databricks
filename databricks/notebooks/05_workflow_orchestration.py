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
dbutils.widgets.text("peloton_username", "")
dbutils.widgets.text("peloton_password", "")
dbutils.widgets.text("peloton_secret_scope", "peloton")
dbutils.widgets.text("peloton_username_key", "username")
dbutils.widgets.text("peloton_password_key", "password")
dbutils.widgets.text("mlflow_enabled", "true")
dbutils.widgets.text("mlflow_experiment_name", "")
dbutils.widgets.text("mlflow_run_name", "peloton-lakehouse-train")
dbutils.widgets.text("mlflow_registered_model_name", "")
dbutils.widgets.text("mlflow_model_alias", "Champion")
dbutils.widgets.text("optuna_enabled", "false")
dbutils.widgets.text("optuna_trials", "20")

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
artifact_base_path = dbutils.widgets.get("artifact_base_path")
peloton_since = dbutils.widgets.get("peloton_since")
peloton_max_workouts = dbutils.widgets.get("peloton_max_workouts")
peloton_username = dbutils.widgets.get("peloton_username")
peloton_password = dbutils.widgets.get("peloton_password")
peloton_secret_scope = dbutils.widgets.get("peloton_secret_scope").strip()
peloton_username_key = dbutils.widgets.get("peloton_username_key").strip()
peloton_password_key = dbutils.widgets.get("peloton_password_key").strip()
mlflow_enabled = dbutils.widgets.get("mlflow_enabled").strip()
mlflow_experiment_name = dbutils.widgets.get("mlflow_experiment_name").strip()
mlflow_run_name = dbutils.widgets.get("mlflow_run_name").strip()
mlflow_registered_model_name = dbutils.widgets.get("mlflow_registered_model_name").strip()
mlflow_model_alias = dbutils.widgets.get("mlflow_model_alias").strip() or "Champion"
optuna_enabled = dbutils.widgets.get("optuna_enabled").strip()
optuna_trials = dbutils.widgets.get("optuna_trials").strip()
if not mlflow_registered_model_name:
    mlflow_registered_model_name = f"{catalog}.{schema}.peloton_total_work_model"

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
os.environ["DATABRICKS_ARTIFACT_BASE_PATH"] = artifact_base_path
if peloton_since:
    os.environ["PELOTON_SINCE"] = peloton_since
if peloton_max_workouts:
    os.environ["PELOTON_MAX_WORKOUTS"] = peloton_max_workouts
if mlflow_enabled:
    os.environ["MLFLOW_ENABLED"] = mlflow_enabled
if mlflow_experiment_name:
    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name
if mlflow_run_name:
    os.environ["MLFLOW_RUN_NAME"] = mlflow_run_name
if mlflow_registered_model_name:
    os.environ["MLFLOW_REGISTERED_MODEL_NAME"] = mlflow_registered_model_name
if mlflow_model_alias:
    os.environ["MLFLOW_MODEL_ALIAS"] = mlflow_model_alias
if optuna_enabled:
    os.environ["OPTUNA_ENABLED"] = optuna_enabled
if optuna_trials:
    os.environ["OPTUNA_TRIALS"] = optuna_trials


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

from peloton_databricks_pipeline.pipeline import run_lakehouse

results = run_lakehouse(write_local_staging=False, model_base_path=artifact_base_path)
print(results)
