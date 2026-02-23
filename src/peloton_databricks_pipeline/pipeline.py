from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import get_settings
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


def run_extract() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    settings.require_peloton()
    assert settings.peloton_username is not None
    assert settings.peloton_password is not None
    client = PelotonClient(settings.peloton_username, settings.peloton_password)
    client.authenticate()

    workouts = client.get_workouts()
    workouts = filter_workouts_since(workouts, settings.peloton_since)

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

    RAW_WORKOUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROC_WORKOUTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    RAW_WORKOUTS_PATH.write_text(json.dumps(workouts, indent=2), encoding="utf-8")
    RAW_PERF_PATH.write_text(json.dumps(raw_performance, indent=2), encoding="utf-8")

    workouts_df.to_csv(PROC_WORKOUTS_PATH, index=False)
    metrics_df.to_csv(PROC_METRICS_PATH, index=False)
    training_df.to_csv(PROC_TRAIN_PATH, index=False)

    return workouts_df, metrics_df, training_df


def run_load(workouts_df: pd.DataFrame | None = None, metrics_df: pd.DataFrame | None = None) -> None:
    settings = get_settings()
    settings.require_databricks()
    assert settings.databricks_server_hostname is not None
    assert settings.databricks_http_path is not None
    assert settings.databricks_access_token is not None
    loader = DatabricksLoader(
        server_hostname=settings.databricks_server_hostname,
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_access_token,
        catalog=settings.databricks_catalog,
        schema=settings.databricks_schema,
    )

    if workouts_df is None:
        workouts_df = pd.read_csv(PROC_WORKOUTS_PATH)
    if metrics_df is None:
        metrics_df = pd.read_csv(PROC_METRICS_PATH)

    loader.load(workouts_df=workouts_df, metrics_df=metrics_df)


def run_train() -> dict[str, float | int]:
    settings = get_settings()
    settings.require_databricks()
    assert settings.databricks_server_hostname is not None
    assert settings.databricks_http_path is not None
    assert settings.databricks_access_token is not None
    loader = DatabricksLoader(
        server_hostname=settings.databricks_server_hostname,
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_access_token,
        catalog=settings.databricks_catalog,
        schema=settings.databricks_schema,
    )

    training_df = loader.read_training_frame()
    return train_and_generate_insights(training_df)


def run_all() -> dict[str, float | int]:
    workouts_df, metrics_df, _ = run_extract()
    run_load(workouts_df, metrics_df)
    return run_train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Peloton -> Databricks pipeline")
    parser.add_argument(
        "command",
        choices=["extract", "load", "train", "run-all"],
        help="extract raw data, load to Databricks, train ML model, or run-all",
    )
    args = parser.parse_args()

    if args.command == "extract":
        workouts_df, metrics_df, training_df = run_extract()
        print(f"Extracted {len(workouts_df)} workouts and {len(metrics_df)} metric rows. Training rows: {len(training_df)}")
    elif args.command == "load":
        run_load()
        print("Loaded processed CSV data to Databricks")
    elif args.command == "train":
        results = run_train()
        print(f"Training complete: {results}")
    else:
        results = run_all()
        print(f"Pipeline complete: {results}")


if __name__ == "__main__":
    main()
