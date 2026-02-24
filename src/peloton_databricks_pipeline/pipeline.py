from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .config import Settings, get_settings
from .databricks_loader import DatabricksLoader
from .ml import train_and_generate_insights
from .peloton_api import PelotonClient, filter_workouts_since
from .transform import (
    aggregate_metrics,
    build_training_frame,
    performance_to_dataframe,
    workouts_to_dataframe,
)


RAW_WORKOUTS_PATH = Path("data/raw/workouts.json")
RAW_PERF_PATH = Path("data/raw/performance_metrics.json")
PROC_WORKOUTS_PATH = Path("data/processed/workouts.csv")
PROC_METRICS_PATH = Path("data/processed/metrics.csv")
PROC_TRAIN_PATH = Path("data/processed/training_frame.csv")


def _is_databricks_runtime() -> bool:
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION")) or os.getenv("DB_IS_DRIVER", "").lower() == "true"


def _should_use_spark_loader(settings: Settings, force_spark_loader: bool) -> bool:
    return force_spark_loader or settings.use_databricks_spark or _is_databricks_runtime()


def _build_loader(settings: Settings, force_spark_loader: bool = False):
    if _should_use_spark_loader(settings, force_spark_loader):
        try:
            from .databricks_spark_loader import DatabricksSparkLoader
        except Exception as exc:
            raise RuntimeError(
                "Spark loader requested but PySpark is unavailable. "
                "Run inside Databricks Runtime or install pyspark."
            ) from exc

        return DatabricksSparkLoader(
            catalog=settings.databricks_catalog,
            schema=settings.databricks_schema,
        )

    settings.require_databricks()
    assert settings.databricks_server_hostname is not None
    assert settings.databricks_http_path is not None
    assert settings.databricks_access_token is not None
    return DatabricksLoader(
        server_hostname=settings.databricks_server_hostname,
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_access_token,
        catalog=settings.databricks_catalog,
        schema=settings.databricks_schema,
    )


def extract_with_raw_payloads(
    write_local_staging: bool | None = None,
) -> tuple[list[dict], list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    settings.require_peloton()
    assert settings.peloton_username is not None
    assert settings.peloton_password is not None

    write_local = settings.write_local_staging if write_local_staging is None else write_local_staging

    client = PelotonClient(settings.peloton_username, settings.peloton_password)
    client.authenticate()

    workouts = client.get_workouts()
    workouts = filter_workouts_since(workouts, settings.peloton_since)
    if settings.peloton_max_workouts is not None and settings.peloton_max_workouts > 0:
        workouts = workouts[: settings.peloton_max_workouts]

    performance_frames = []
    raw_performance = []
    for workout in workouts:
        workout_id = workout["id"]
        performance = client.get_workout_performance(workout_id)
        performance_frames.append(performance_to_dataframe(workout_id, performance))
        raw_performance.append({"workout_id": workout_id, "performance": performance})

    workouts_df = workouts_to_dataframe(workouts)
    metrics_df = pd.concat(performance_frames, ignore_index=True) if performance_frames else pd.DataFrame()
    metrics_agg_df = aggregate_metrics(metrics_df)
    training_df = build_training_frame(workouts_df, metrics_agg_df)

    if write_local:
        RAW_WORKOUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROC_WORKOUTS_PATH.parent.mkdir(parents=True, exist_ok=True)

        RAW_WORKOUTS_PATH.write_text(json.dumps(workouts, indent=2), encoding="utf-8")
        RAW_PERF_PATH.write_text(json.dumps(raw_performance, indent=2), encoding="utf-8")

        workouts_df.to_csv(PROC_WORKOUTS_PATH, index=False)
        metrics_df.to_csv(PROC_METRICS_PATH, index=False)
        training_df.to_csv(PROC_TRAIN_PATH, index=False)

    return workouts, raw_performance, workouts_df, metrics_df, training_df


def run_extract(write_local_staging: bool | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, _, workouts_df, metrics_df, training_df = extract_with_raw_payloads(write_local_staging=write_local_staging)
    return workouts_df, metrics_df, training_df


def run_load(
    workouts_df: pd.DataFrame | None = None,
    metrics_df: pd.DataFrame | None = None,
    use_spark_loader: bool = False,
) -> None:
    settings = get_settings()
    loader = _build_loader(settings, force_spark_loader=use_spark_loader)

    if workouts_df is None or metrics_df is None:
        if not PROC_WORKOUTS_PATH.exists() or not PROC_METRICS_PATH.exists():
            raise ValueError(
                "No in-memory data was provided and local staged CSVs were not found. "
                "Run `run-all` for direct ingestion or enable `WRITE_LOCAL_STAGING=true`."
            )
        workouts_df = pd.read_csv(PROC_WORKOUTS_PATH)
        metrics_df = pd.read_csv(PROC_METRICS_PATH)

    loader.load(workouts_df=workouts_df, metrics_df=metrics_df)


def run_train(use_spark_loader: bool = False, model_base_path: str | None = None) -> dict[str, object]:
    settings = get_settings()
    loader = _build_loader(settings, force_spark_loader=use_spark_loader)
    training_df = loader.read_training_frame()

    if model_base_path is not None:
        base_path = Path(model_base_path)
    elif _should_use_spark_loader(settings, use_spark_loader):
        base_path = Path(settings.databricks_artifact_base_path)
    else:
        base_path = Path(".")

    model_dir = base_path / "models"
    report_path = base_path / "reports" / "insights.md"

    return train_and_generate_insights(
        training_df,
        model_dir=model_dir,
        report_path=report_path,
        enable_mlflow=settings.mlflow_enabled,
        mlflow_experiment_name=settings.mlflow_experiment_name,
        mlflow_run_name=settings.mlflow_run_name,
        mlflow_registered_model_name=settings.mlflow_registered_model_name,
    )


def run_all(
    use_spark_loader: bool = False,
    write_local_staging: bool | None = None,
    model_base_path: str | None = None,
) -> dict[str, object]:
    settings = get_settings()
    if _should_use_spark_loader(settings, use_spark_loader):
        return run_lakehouse(
            write_local_staging=write_local_staging,
            model_base_path=model_base_path,
        )

    workouts_df, metrics_df, _ = run_extract(write_local_staging=write_local_staging)
    run_load(workouts_df, metrics_df, use_spark_loader=use_spark_loader)
    return run_train(use_spark_loader=use_spark_loader, model_base_path=model_base_path)


def run_lakehouse(
    write_local_staging: bool | None = None,
    model_base_path: str | None = None,
) -> dict[str, object]:
    settings = get_settings()
    try:
        from .databricks_spark_loader import DatabricksSparkLoader
        from .lakehouse import ingest_to_lakehouse
    except Exception as exc:
        raise RuntimeError(
            "Lakehouse mode requires Databricks runtime with PySpark available."
        ) from exc

    workouts_raw, performance_raw, workouts_df, metrics_df, _ = extract_with_raw_payloads(
        write_local_staging=write_local_staging
    )

    loader = DatabricksSparkLoader(
        catalog=settings.databricks_catalog,
        schema=settings.databricks_schema,
    )
    lakehouse_result = ingest_to_lakehouse(
        loader=loader,
        workouts_raw=workouts_raw,
        performance_raw=performance_raw,
        workouts_df=workouts_df,
        metrics_df=metrics_df,
    )

    train_result = run_train(use_spark_loader=True, model_base_path=model_base_path)
    return {
        "bronze_workouts_rows": lakehouse_result.workouts_raw_rows,
        "bronze_metrics_rows": lakehouse_result.metrics_raw_rows,
        "silver_workouts_rows": lakehouse_result.workouts_rows,
        "silver_metrics_rows": lakehouse_result.metrics_rows,
        "rows_used_for_training": int(train_result["rows_used_for_training"]),
        "mae": float(train_result["mae"]),
        "r2": float(train_result["r2"]),
        "cluster_count": int(train_result["cluster_count"]),
        "artifact_model_dir": train_result.get("artifact_model_dir"),
        "artifact_report_path": train_result.get("artifact_report_path"),
        "mlflow_status": train_result.get("mlflow_status"),
        "mlflow_run_id": train_result.get("mlflow_run_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Peloton -> Databricks pipeline")
    parser.add_argument(
        "command",
        choices=["extract", "load", "train", "run-all", "run-lakehouse"],
        help="extract raw data, load, train, run-all, or run-lakehouse",
    )
    parser.add_argument(
        "--write-local-staging",
        action="store_true",
        help="Persist raw/processed files locally. By default, staging is skipped.",
    )
    parser.add_argument(
        "--use-spark-loader",
        action="store_true",
        help="Use Databricks Spark runtime for direct Delta writes (recommended in Databricks jobs).",
    )
    parser.add_argument(
        "--model-base-path",
        default=None,
        help="Base path for ML artifacts; in Databricks use /dbfs/...",
    )
    args = parser.parse_args()

    if args.command == "extract":
        workouts_df, metrics_df, training_df = run_extract(write_local_staging=args.write_local_staging)
        print(f"Extracted {len(workouts_df)} workouts and {len(metrics_df)} metric rows. Training rows: {len(training_df)}")
    elif args.command == "load":
        run_load(use_spark_loader=args.use_spark_loader)
        print("Loaded data to Databricks")
    elif args.command == "train":
        results = run_train(use_spark_loader=args.use_spark_loader, model_base_path=args.model_base_path)
        print(f"Training complete: {results}")
    elif args.command == "run-lakehouse":
        results = run_lakehouse(
            write_local_staging=args.write_local_staging,
            model_base_path=args.model_base_path,
        )
        print(f"Lakehouse pipeline complete: {results}")
    else:
        results = run_all(
            use_spark_loader=args.use_spark_loader,
            write_local_staging=args.write_local_staging,
            model_base_path=args.model_base_path,
        )
        print(f"Pipeline complete: {results}")


if __name__ == "__main__":
    main()
