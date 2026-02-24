# Peloton Lakehouse on Databricks

This repository contains an end-to-end Peloton analytics pipeline I built for Databricks.

The pipeline does five things:
1. Ingestion (Bronze + Silver)
2. Governance
3. BI / Gold layer
4. ML training and insight generation
5. Workflow orchestration

It is designed to run inside Databricks and write directly to Delta tables, with no local staging by default.

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

- `databricks/notebooks/00_setup.py`
- `databricks/notebooks/01_data_ingestion.py`
- `databricks/notebooks/02_data_governance.py`
- `databricks/notebooks/03_bi.py`
- `databricks/notebooks/04_ml.py`
- `databricks/notebooks/05_workflow_orchestration.py`

### Workflow definition

- `databricks/workflows/peloton_lakehouse_job.json`

## Run In Databricks (Recommended)

### 1. Configure secrets/env for the job

Required:
- `PELOTON_USERNAME`
- `PELOTON_PASSWORD`

Recommended:
- `USE_DATABRICKS_SPARK=true`
- `WRITE_LOCAL_STAGING=false`
- `DATABRICKS_CATALOG=main`
- `DATABRICKS_SCHEMA=fitness`
- `DATABRICKS_ARTIFACT_BASE_PATH=/dbfs/FileStore/peloton_analytics`
- `PELOTON_MAX_WORKOUTS=150` (for a faster first run)
- optional `PELOTON_SINCE=2025-01-01T00:00:00Z`

### 2. Run one of these options

Notebook pipeline:
- Run notebooks in order: `00` -> `01` -> `02` -> `03` -> `04`
- Or import `databricks/workflows/peloton_lakehouse_job.json` and run as a Databricks job

Single entrypoint:
- `databricks/notebooks/05_workflow_orchestration.py`
- `databricks/run_pipeline.py`

## CLI Commands

Main command:

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
- In Databricks runtime, `run-all` automatically switches to lakehouse mode.
- Local staging is off by default; enable with `--write-local-staging` or `WRITE_LOCAL_STAGING=true`.

## Local Fallback

If you run outside Databricks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Local mode uses the Databricks SQL API for load/read unless Spark mode is enabled.

## SQL for BI

Reusable BI queries:
- `databricks/sql/bi_queries.sql`
