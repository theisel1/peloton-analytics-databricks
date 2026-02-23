# Peloton -> Databricks Python Pipeline

This project pulls Peloton workout data and writes it directly to Databricks Delta tables, then trains ML models for insights.

## Architecture

- Extract from Peloton API (OAuth flow)
- Load directly to Databricks Delta tables:
  - `main.fitness.peloton_workouts`
  - `main.fitness.peloton_metrics`
- Train ML model + clustering and write artifacts

Local staging files are **off by default**.

## Recommended: Run Entirely In Databricks

Run this code as a Databricks Workflow job (Python task).

### 1. Job environment variables

Set these in the job/task environment:

- `PELOTON_USERNAME`
- `PELOTON_PASSWORD`
- optional `PELOTON_SINCE`
- optional `PELOTON_MAX_WORKOUTS`
- `USE_DATABRICKS_SPARK=true`
- `WRITE_LOCAL_STAGING=false`
- optional `DATABRICKS_ARTIFACT_BASE_PATH=/dbfs/FileStore/peloton_analytics`
- optional `DATABRICKS_CATALOG=main`
- optional `DATABRICKS_SCHEMA=fitness`

### 2. Job command

Use either:

```bash
peloton-pipeline run-all --use-spark-loader
```

or the included Databricks entrypoint:

```bash
python databricks/run_pipeline.py
```

This path writes directly to Delta tables via Spark (no local CSV/JSON staging).

## Local Execution (Fallback)

Use this only when you want to run outside Databricks.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

For local SQL API mode, `.env` must include:

- `DATABRICKS_SERVER_HOSTNAME`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_ACCESS_TOKEN`

### Commands

```bash
peloton-pipeline run-all
```

If you want local staging files:

```bash
peloton-pipeline run-all --write-local-staging
```

## ML Artifacts

- Databricks mode default: `/dbfs/FileStore/peloton_analytics/models` and `/dbfs/FileStore/peloton_analytics/reports`
- Local mode default: `./models` and `./reports`

## Notes

- Peloton endpoints are unofficial and may change.
- In Databricks mode, Spark is used directly for table writes.
- In local mode, Databricks SQL Statements API is used.
