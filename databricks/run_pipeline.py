"""Databricks Job entrypoint: run Peloton ingestion directly into Delta tables."""

from peloton_databricks_pipeline.pipeline import run_all


if __name__ == "__main__":
    results = run_all(use_spark_loader=True, write_local_staging=False)
    print(results)
