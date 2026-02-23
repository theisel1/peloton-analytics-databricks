# Peloton -> Databricks Python Pipeline

This project extracts workout data from Peloton, loads it into Databricks tables, and runs ML to generate workout insights.

## What This Pipeline Does

1. Authenticates to Peloton and extracts workouts + performance metrics.
2. Normalizes data into tabular CSVs.
3. Loads tables into Databricks SQL Warehouse (`peloton_workouts`, `peloton_metrics`).
4. Trains ML models to:
   - Predict workout `total_work`.
   - Cluster workouts into segments.
5. Writes artifacts:
   - `models/peloton_work_model.joblib`
   - `models/cluster_summary.csv`
   - `reports/insights.md`

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill `.env` with:
- `PELOTON_USERNAME`
- `PELOTON_PASSWORD`
- `DATABRICKS_SERVER_HOSTNAME`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_ACCESS_TOKEN`
- optional `PELOTON_SINCE`

## Run Commands

Extract only:

```bash
peloton-pipeline extract
```

Load only (reads processed CSVs):

```bash
peloton-pipeline load
```

Train only (reads from Databricks):

```bash
peloton-pipeline train
```

Run full pipeline:

```bash
peloton-pipeline run-all
```

## Output Data

- Raw JSON: `data/raw/`
- Processed CSV: `data/processed/`
- SQL DDL + views: `sql/`
- ML artifacts: `models/` and `reports/`

## Notes

- Peloton endpoints are not officially public/stable, so endpoint behavior can change.
- Use a Databricks SQL Warehouse with permissions to create schema/table and run MERGE.
- For production: move secrets to a secret manager and orchestrate with Airflow/Databricks Workflows.
