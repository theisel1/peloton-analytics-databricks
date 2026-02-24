from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def _ensure_artifact_paths(model_dir: Path, report_path: Path) -> tuple[Path, Path]:
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        return model_dir, report_path
    except OSError:
        fallback_root = Path("/tmp/peloton_analytics")
        fallback_model_dir = fallback_root / "models"
        fallback_report_path = fallback_root / "reports" / "insights.md"
        fallback_model_dir.mkdir(parents=True, exist_ok=True)
        fallback_report_path.parent.mkdir(parents=True, exist_ok=True)
        return fallback_model_dir, fallback_report_path


def _log_mlflow_artifacts(
    *,
    enable_mlflow: bool,
    mlflow_experiment_name: str | None,
    mlflow_run_name: str | None,
    mlflow_registered_model_name: str | None,
    regressor: RandomForestRegressor,
    feature_columns: list[str],
    importances: pd.Series,
    model_path: Path,
    cluster_summary_path: Path,
    report_path: Path,
    rows_used_for_training: int,
    mae: float,
    r2: float,
    cluster_count: int,
) -> tuple[str, str | None]:
    if not enable_mlflow:
        return "disabled", None

    try:
        import mlflow
        import mlflow.sklearn
    except Exception:
        return "unavailable", None

    try:
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)

        start_kwargs: dict[str, object] = {}
        if mlflow_run_name:
            start_kwargs["run_name"] = mlflow_run_name
        if mlflow.active_run() is not None:
            start_kwargs["nested"] = True

        with mlflow.start_run(**start_kwargs) as run:
            mlflow.log_params(
                {
                    "model_type": "RandomForestRegressor",
                    "n_estimators": regressor.n_estimators,
                    "train_rows": rows_used_for_training,
                    "feature_count": len(feature_columns),
                    "cluster_count": cluster_count,
                }
            )
            mlflow.log_metrics(
                {
                    "mae": mae,
                    "r2": r2,
                }
            )
            for feature_name, importance in importances.items():
                mlflow.log_metric(f"importance_{feature_name}", float(importance))

            if mlflow_registered_model_name:
                mlflow.sklearn.log_model(
                    sk_model=regressor,
                    artifact_path="model",
                    registered_model_name=mlflow_registered_model_name,
                )
            else:
                mlflow.sklearn.log_model(sk_model=regressor, artifact_path="model")

            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(cluster_summary_path))
            mlflow.log_artifact(str(report_path))
            return "logged", run.info.run_id
    except Exception as exc:
        print(f"MLflow logging skipped due to error: {exc}")
        return "error", None


def train_and_generate_insights(
    training_df: pd.DataFrame,
    model_dir: Path = Path("models"),
    report_path: Path = Path("reports/insights.md"),
    enable_mlflow: bool = True,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_registered_model_name: str | None = None,
) -> dict[str, float | int | str | None]:
    if training_df.empty:
        raise ValueError("Training data is empty. Run extraction and loading first.")

    feature_columns = [
        "ride_duration",
        "distance",
        "calories",
        "avg_cadence",
        "avg_heart_rate",
        "avg_resistance",
        "avg_speed",
    ]

    target_column = "total_work"
    numeric_columns = list({target_column, *feature_columns, "avg_output"})
    for column in numeric_columns:
        if column in training_df.columns:
            training_df[column] = pd.to_numeric(training_df[column], errors="coerce")

    usable = training_df[[target_column, *feature_columns]].copy()
    usable = usable.dropna(subset=[target_column])
    if len(usable) < 8:
        raise ValueError("Need at least 8 workouts with target values for reliable ML training.")

    X = usable[feature_columns].copy()
    y = usable[target_column]
    for column in feature_columns:
        if X[column].notna().sum() == 0:
            X.loc[:, column] = 0.0

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)

    regressor = RandomForestRegressor(n_estimators=300, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    kmeans_features = ["ride_duration", "avg_heart_rate", "avg_output", "avg_resistance"]
    for column in kmeans_features:
        if column not in training_df.columns:
            training_df[column] = np.nan
    cluster_frame = training_df[kmeans_features].apply(pd.to_numeric, errors="coerce")
    cluster_frame = cluster_frame.fillna(cluster_frame.median(numeric_only=True))
    cluster_frame = cluster_frame.fillna(0)

    n_clusters = min(3, len(cluster_frame))
    if n_clusters < 2:
        raise ValueError("Need at least 2 workouts to run clustering.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    clusters = kmeans.fit_predict(cluster_frame)

    enriched = training_df.copy()
    enriched["cluster"] = clusters

    cluster_summary = (
        enriched.groupby("cluster")
        .agg(
            workouts=("workout_id", "count"),
            avg_duration=("ride_duration", "mean"),
            avg_calories=("calories", "mean"),
            avg_heart_rate=("avg_heart_rate", "mean"),
            avg_output=("avg_output", "mean"),
        )
        .round(2)
    )

    importances = pd.Series(regressor.feature_importances_, index=feature_columns).sort_values(ascending=False)

    model_dir, report_path = _ensure_artifact_paths(model_dir=model_dir, report_path=report_path)

    model_path = model_dir / "peloton_work_model.joblib"
    cluster_summary_path = model_dir / "cluster_summary.csv"

    joblib.dump({"model": regressor, "imputer": imputer, "features": feature_columns}, model_path)
    cluster_summary.to_csv(cluster_summary_path)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Peloton ML Insights\n\n")
        f.write("## Model Quality\n")
        f.write(f"- MAE (total_work prediction): {mae:.2f}\n")
        f.write(f"- R^2: {r2:.3f}\n\n")
        f.write("## Top Predictors of Workout Output\n")
        for feature, importance in importances.items():
            f.write(f"- {feature}: {importance:.3f}\n")
        f.write(f"\n## Workout Segments (KMeans, k={n_clusters})\n")
        f.write(cluster_summary.to_string())
        f.write("\n")

    mlflow_status, mlflow_run_id = _log_mlflow_artifacts(
        enable_mlflow=enable_mlflow,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        mlflow_registered_model_name=mlflow_registered_model_name,
        regressor=regressor,
        feature_columns=feature_columns,
        importances=importances,
        model_path=model_path,
        cluster_summary_path=cluster_summary_path,
        report_path=report_path,
        rows_used_for_training=int(len(usable)),
        mae=mae,
        r2=r2,
        cluster_count=int(n_clusters),
    )

    return {
        "rows_used_for_training": int(len(usable)),
        "mae": mae,
        "r2": r2,
        "cluster_count": int(n_clusters),
        "artifact_model_dir": str(model_dir),
        "artifact_report_path": str(report_path),
        "mlflow_status": mlflow_status,
        "mlflow_run_id": mlflow_run_id,
    }
