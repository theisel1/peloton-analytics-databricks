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

repo_root = dbutils.widgets.get("repo_root")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
artifact_base_path = dbutils.widgets.get("artifact_base_path")

# COMMAND ----------

src_path = str(Path(repo_root) / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ["DATABRICKS_CATALOG"] = catalog
os.environ["DATABRICKS_SCHEMA"] = schema
os.environ["USE_DATABRICKS_SPARK"] = "true"
os.environ["DATABRICKS_ARTIFACT_BASE_PATH"] = artifact_base_path

from peloton_databricks_pipeline.pipeline import run_train

results = run_train(use_spark_loader=True, model_base_path=artifact_base_path)
print(results)

# COMMAND ----------

spark.sql(
    f"SELECT * FROM `{catalog}`.`{schema}`.`gold_peloton_discipline_summary` ORDER BY workouts DESC"
).show(truncate=False)
