# Peloton Lakehouse on Databricks (Demo-Style)

This project now follows a Databricks demo-style structure similar to the Lakehouse tutorials:

1. Ingestion (Bronze + Silver)
2. Governance
3. BI / Gold layer
4. ML
5. Workflow orchestration

It is designed to run inside Databricks and write directly to Delta tables with no local staging by default.

## Architecture

### Medallion objects

- Bronze:
  - `main.fitness.bronze_peloton_workouts_raw`
  - `main.fitness.bronze_peloton_metrics_raw`
- Silver:
  - `main.fitness.peloton_workouts`
  - `main.fitness.peloton_metrics`
  - views: `main.fitness.silver_peloton_workouts`, `main.fitness.silver_peloton_metrics`
- Gold:
  - `main.fitness.gold_peloton_daily_summary`
  - `main.fitness.gold_peloton_discipline_summary`
  - `main.fitness.gold_peloton_instructor_summary`

### Databricks notebooks

- `/Users/thomasheisel/Documents/New project/databricks/notebooks/00_setup.py`
- `/Users/thomasheisel/Documents/New project/databricks/notebooks/01_data_ingestion.py`
- `/Users/thomasheisel/Documents/New project/databricks/notebooks/02_data_governance.py`
- `/Users/thomasheisel/Documents/New project/databricks/notebooks/03_bi.py`
- `/Users/thomasheisel/Documents/New project/databricks/notebooks/04_ml.py`
- `/Users/thomasheisel/Documents/New project/databricks/notebooks/05_workflow_orchestration.py`

### Workflow spec

- `/Users/thomasheisel/Documents/New project/databricks/workflows/peloton_lakehouse_job.json`

## Run In Databricks (Recommended)

### 1. Configure secrets/env for the Job

Required:
- `PELOTON_USERNAME`
- `PELOTON_PASSWORD`

Recommended:
- `USE_DATABRICKS_SPARK=true`
- `WRITE_LOCAL_STAGING=false`
- `DATABRICKS_CATALOG=main`
- `DATABRICKS_SCHEMA=fitness`
- `DATABRICKS_ARTIFACT_BASE_PATH=/dbfs/FileStore/peloton_analytics`
- `PELOTON_MAX_WORKOUTS=150` (for quicker first run)
- optional `PELOTON_SINCE=2025-01-01T00:00:00Z`

### 2. Run one of these

Notebook workflow style:
- Run notebooks in order `00` -> `01` -> `02` -> `03` -> `04`
- Or import/update `/databricks/workflows/peloton_lakehouse_job.json` and run as a job

Single entrypoint:
- `/Users/thomasheisel/Documents/New project/databricks/notebooks/05_workflow_orchestration.py`
- `/Users/thomasheisel/Documents/New project/databricks/run_pipeline.py`

## CLI Commands

From this repo:

```bash
peloton-pipeline run-lakehouse
```

Other commands:

```bash
peloton-pipeline extract
peloton-pipeline load
peloton-pipeline train
peloton-pipeline run-all
```

Notes:
- In Databricks runtime, `run-all` automatically switches to Lakehouse mode.
- Local staging is off by default; enable with `--write-local-staging` or `WRITE_LOCAL_STAGING=true`.

## Local Fallback

If you run outside Databricks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Local mode uses Databricks SQL API for load/read unless Spark mode is enabled.

## SQL for BI

Reusable BI queries:
- `/Users/thomasheisel/Documents/New project/databricks/sql/bi_queries.sql`
