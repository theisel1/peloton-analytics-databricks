"""Databricks Job entrypoint: run Peloton medallion + ML pipeline."""

from peloton_databricks_pipeline.pipeline import run_lakehouse


if __name__ == "__main__":
    results = run_lakehouse(write_local_staging=False)
    print(results)
