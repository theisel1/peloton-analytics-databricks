# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Model Scoring
# MAGIC Use trained model artifacts to score a single workout or batch-score recent workouts.

# COMMAND ----------

import os
import sys
from pathlib import Path

import pandas as pd

if "dbutils" not in globals():
    raise RuntimeError("This notebook is intended to run in Databricks.")

# COMMAND ----------

dbutils.widgets.text("repo_root", "/Workspace/Repos/<user>/peloton-analytics-databricks")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "fitness")
dbutils.widgets.text("artifact_base_path", "/dbfs/FileStore/peloton_analytics")
dbutils.widgets.dropdown("model_source", "mlflow_latest", ["mlflow_latest", "path"])
dbutils.widgets.text("model_path", "")
default_mlflow_experiment = f"/Users/{spark.sql('SELECT current_user() AS user').first()['user']}/peloton-analytics"
dbutils.widgets.text("mlflow_experiment_name", default_mlflow_experiment)
dbutils.widgets.text("mlflow_run_name", "peloton-ml-training")
dbutils.widgets.text("mlflow_run_id", "")
dbutils.widgets.dropdown("mode", "batch", ["batch", "single"])
dbutils.widgets.text("score_limit", "250")
dbutils.widgets.text("output_table", "main.fitness.gold_peloton_total_work_predictions")

dbutils.widgets.text("fitness_discipline", "cycling")
dbutils.widgets.text("ride_duration", "1800")
dbutils.widgets.text("distance", "")
dbutils.widgets.text("calories", "")
dbutils.widgets.text("avg_cadence", "")
dbutils.widgets.text("avg_heart_rate", "")
dbutils.widgets.text("avg_resistance", "")
dbutils.widgets.text("avg_speed", "")

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
artifact_base_path = dbutils.widgets.get("artifact_base_path").strip()
model_source = dbutils.widgets.get("model_source").strip().lower()
model_path_widget = dbutils.widgets.get("model_path").strip()
mlflow_experiment_name = dbutils.widgets.get("mlflow_experiment_name").strip()
mlflow_run_name = dbutils.widgets.get("mlflow_run_name").strip()
mlflow_run_id = dbutils.widgets.get("mlflow_run_id").strip()
mode = dbutils.widgets.get("mode").strip().lower()
score_limit = dbutils.widgets.get("score_limit").strip()
output_table = dbutils.widgets.get("output_table").strip()

# COMMAND ----------

def _normalize_workspace_root(path: str) -> str:
    if path.startswith("/Workspace/"):
        return path
    if path.startswith("/Users/") or path.startswith("/Repos/"):
        return f"/Workspace{path}"
    return path


def _to_float_or_none(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    return float(text)


def _stage_model_locally(path: str) -> str:
    model_path = path.strip()
    if model_path.startswith("/dbfs/"):
        dbfs_uri = model_path.replace("/dbfs/", "dbfs:/", 1)
    elif model_path.startswith("dbfs:/"):
        dbfs_uri = model_path
    else:
        return model_path

    local_path = f"/tmp/{Path(model_path).name}"
    dbutils.fs.cp(dbfs_uri, f"file:{local_path}", True)
    return local_path


repo_root = _normalize_workspace_root(repo_root)
src_path = str(Path(repo_root) / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ["DATABRICKS_CATALOG"] = catalog
os.environ["DATABRICKS_SCHEMA"] = schema

from peloton_databricks_pipeline.databricks_spark_loader import DatabricksSparkLoader
from peloton_databricks_pipeline.scoring import FEATURE_COLUMNS, load_model_bundle, predict_total_work

default_model_path = f"{artifact_base_path.rstrip('/')}/models/peloton_work_model.joblib"
model_path = model_path_widget or default_model_path
bundle: dict
if model_source == "mlflow_latest":
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if mlflow_run_id:
        run = client.get_run(mlflow_run_id)
    else:
        experiment = client.get_experiment_by_name(mlflow_experiment_name)
        if experiment is None:
            raise ValueError(f"MLflow experiment not found: {mlflow_experiment_name}")
        filter_string = ""
        if mlflow_run_name:
            filter_string = f"tags.`mlflow.runName` = '{mlflow_run_name}'"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(
                "No MLflow runs found for scoring. Run the training job first or provide mlflow_run_id."
            )
        run = runs[0]

    run_id = run.info.run_id
    stage2 = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    stage1 = mlflow.sklearn.load_model(f"runs:/{run_id}/stage1_classifier_model")
    threshold = float(run.data.params.get("stage1_threshold", "0.5"))
    bundle = {
        "stage1_classifier": stage1,
        "stage2_regressor": stage2,
        "stage1_threshold": threshold,
    }
    print(f"Loaded MLflow run: {run_id}")
else:
    local_model_path = _stage_model_locally(model_path)
    bundle = load_model_bundle(local_model_path)
    print(f"Loaded model artifact: {model_path} (local copy: {local_model_path})")

print(f"Scoring mode: {mode}")

# COMMAND ----------

if mode == "single":
    single_input = pd.DataFrame(
        [
            {
                "fitness_discipline": dbutils.widgets.get("fitness_discipline"),
                "ride_duration": _to_float_or_none(dbutils.widgets.get("ride_duration")),
                "distance": _to_float_or_none(dbutils.widgets.get("distance")),
                "calories": _to_float_or_none(dbutils.widgets.get("calories")),
                "avg_cadence": _to_float_or_none(dbutils.widgets.get("avg_cadence")),
                "avg_heart_rate": _to_float_or_none(dbutils.widgets.get("avg_heart_rate")),
                "avg_resistance": _to_float_or_none(dbutils.widgets.get("avg_resistance")),
                "avg_speed": _to_float_or_none(dbutils.widgets.get("avg_speed")),
            }
        ]
    )
    prediction = predict_total_work(single_input, bundle)
    result = pd.concat([single_input, prediction], axis=1)
    display(result)
else:
    loader = DatabricksSparkLoader(catalog=catalog, schema=schema, spark=spark)
    training_frame = loader.read_training_frame()
    if training_frame.empty:
        raise ValueError("Training frame is empty; run the ingestion + ML pipeline first.")

    if "created_at" in training_frame.columns:
        training_frame["created_at"] = pd.to_datetime(training_frame["created_at"], errors="coerce")
        training_frame = training_frame.sort_values("created_at", ascending=False)

    limit = int(score_limit) if score_limit else 0
    if limit > 0:
        training_frame = training_frame.head(limit).copy()

    prediction = predict_total_work(training_frame[["fitness_discipline", *FEATURE_COLUMNS]], bundle)
    scored = training_frame.copy()
    scored["predicted_total_work"] = prediction["predicted_total_work"]
    if "stage1_positive_probability" in prediction.columns:
        scored["stage1_positive_probability"] = prediction["stage1_positive_probability"]
    if "stage1_predicted_nonzero" in prediction.columns:
        scored["stage1_predicted_nonzero"] = prediction["stage1_predicted_nonzero"]
    if "total_work" in scored.columns:
        scored["prediction_error"] = scored["total_work"] - scored["predicted_total_work"]

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    spark_df = spark.createDataFrame(scored)
    spark_df.write.mode("overwrite").format("delta").saveAsTable(output_table)

    print(f"Wrote {len(scored)} scored rows to {output_table}")
    display(spark_df.limit(50))
