# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Governance
# MAGIC Apply basic medallion metadata and create a curated secure view.

# COMMAND ----------

if "dbutils" not in globals():
    raise RuntimeError("This notebook is intended to run in Databricks.")

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "fitness")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(
    f"ALTER TABLE `{catalog}`.`{schema}`.`bronze_peloton_workouts_raw` "
    "SET TBLPROPERTIES ('quality'='bronze','source'='peloton_api')"
)
spark.sql(
    f"ALTER TABLE `{catalog}`.`{schema}`.`bronze_peloton_metrics_raw` "
    "SET TBLPROPERTIES ('quality'='bronze','source'='peloton_api')"
)
spark.sql(
    f"ALTER TABLE `{catalog}`.`{schema}`.`peloton_workouts` "
    "SET TBLPROPERTIES ('quality'='silver','contains_pii'='false')"
)
spark.sql(
    f"ALTER TABLE `{catalog}`.`{schema}`.`peloton_metrics` "
    "SET TBLPROPERTIES ('quality'='silver','contains_pii'='false')"
)

spark.sql(
    f"""
    CREATE OR REPLACE VIEW `{catalog}`.`{schema}`.`secure_peloton_workouts` AS
    SELECT
      sha2(workout_id, 256) AS workout_key,
      status,
      fitness_discipline,
      date(start_time) AS workout_date,
      total_work,
      distance,
      calories,
      ride_title,
      ride_duration,
      instructor_name
    FROM `{catalog}`.`{schema}`.`peloton_workouts`
    """
)

# COMMAND ----------

spark.sql(
    f"SHOW TBLPROPERTIES `{catalog}`.`{schema}`.`peloton_workouts`"
).show(truncate=False)
