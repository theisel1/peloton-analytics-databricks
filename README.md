# Peloton Analytics on Databricks

End-to-end data and analytics project using my Peloton workout history.

The goals:

- Practice modern data engineering patterns (medallion architecture, Delta-style layers).
- Build a small but realistic analytics and ML workflow.
- Show how this can run **locally** and on **Databricks (Community Edition)**.

## Project Overview

**Data source**

Peloton workout history (rides, runs, etc.), exported as JSON/CSV and stored in `data_sample/` for demo purposes.

**Architecture**

- **Bronze**: raw Peloton exports, as close to source as possible.
- **Silver**: cleaned and normalized workouts (timestamps, metrics, units).
- **Gold**: analytics-friendly tables with daily/weekly aggregates, PR tracking, and training load metrics.

**Tech stack**

- Python
- (Optional) PySpark
- Pandas / SQL (e.g. DuckDB locally)
- Databricks Community Edition (for notebooks, Delta-style tables, and MLflow)

## Folder Structure

- `notebooks/` – notebooks for exploration, transformations, and analytics.
- `src/` – reusable Python modules (ingest, transform, feature engineering, modeling).
- `data_sample/` – anonymized sample Peloton data for the demo.
- `databricks/` – Databricks-specific exported notebooks / job configs.

## How to Run (local)

_TBD – will document once the first scripts are in place._

## How to Run (Databricks Community Edition)

_TBD – will add steps for importing notebooks and running on a small cluster._
