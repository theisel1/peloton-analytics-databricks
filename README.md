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
- `databricks/notebooks/06_scoring.py`

### Workflow definition

- `databricks/workflows/peloton_lakehouse_job.json`

## Run In Databricks (Recommended)

### 1. Configure Databricks secrets for Peloton credentials

Create a Databricks secret scope and keys used by the ingestion notebook:
- scope: `peloton`
- key: `username`
- key: `password`

The workflow reads these via job parameters:
- `peloton_secret_scope=peloton`
- `peloton_username_key=username`
- `peloton_password_key=password`

Do not store plaintext credentials in job parameters.

### 2. Configure runtime settings

Recommended:
- `USE_DATABRICKS_SPARK=true`
- `WRITE_LOCAL_STAGING=false`
- `DATABRICKS_CATALOG=main`
- `DATABRICKS_SCHEMA=fitness`
- `DATABRICKS_ARTIFACT_BASE_PATH=/dbfs/FileStore/peloton_analytics`
- `PELOTON_MAX_WORKOUTS=150` (for a faster first run)
- optional `PELOTON_SINCE=2025-01-01T00:00:00Z`
- `MLFLOW_ENABLED=true`
- `MLFLOW_EXPERIMENT_NAME=/Users/<you>/peloton-analytics`
- optional `MLFLOW_RUN_NAME=peloton-ml-training`
- `MLFLOW_REGISTERED_MODEL_NAME=main.fitness.peloton_total_work_model`
- `MLFLOW_MODEL_ALIAS=Champion`
- `OPTUNA_ENABLED=false`
- `OPTUNA_TRIALS=20`

### 3. Deploy with Databricks Asset Bundle (recommended)

Bundle files:
- `databricks.yml`
- `databricks/resources/peloton_lakehouse_pipeline.yml`
- `databricks/resources/peloton_workout_scoring.yml`

Use the Databricks CLI bundle workflow:

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev peloton_lakehouse_pipeline
databricks bundle run -t dev peloton_workout_scoring
```

For production:

```bash
databricks bundle validate -t prod
databricks bundle deploy -t prod
databricks bundle run -t prod peloton_lakehouse_pipeline
databricks bundle run -t prod peloton_workout_scoring
```

### 4. Alternate run options

Notebook pipeline:
- Run notebooks in order: `00` -> `01` -> `02` -> `03` -> `04`
- Or import `databricks/workflows/peloton_lakehouse_job.json` and run as a Databricks job

Single entrypoint:
- `databricks/notebooks/05_workflow_orchestration.py`
- `databricks/run_pipeline.py`

MLflow metrics and model artifacts are logged during `04_ml.py` / `run_train`.
The stage-2 regressor is registered to Unity Catalog Model Registry and the configured alias is promoted automatically for the new version.
Optuna tuning is available for the two-stage random forest and can be enabled when you want to run a tuning experiment.
Logged quality metrics include:
- `mae`, `r2`
- `baseline_mae`, `baseline_r2`
- `mae_improvement_vs_baseline`, `r2_improvement_vs_baseline`
- `split_type` (`time_ordered` when timestamps are available, otherwise `random`)

Scoring output:
- batch scoring writes predictions to `main.fitness.gold_peloton_total_work_predictions`
- scoring notebook/job loads from `models:/<registered_model_name>@<alias>` by default
- single scoring mode in `06_scoring.py` predicts from widget inputs

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
