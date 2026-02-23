from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_generate_insights(
    training_df: pd.DataFrame,
    model_dir: Path = Path("models"),
    report_path: Path = Path("reports/insights.md"),
) -> dict[str, float | int]:
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

    usable = training_df[[target_column, *feature_columns]].copy()
    usable[target_column] = pd.to_numeric(usable[target_column], errors="coerce")
    usable = usable.dropna(subset=[target_column])
    if len(usable) < 8:
        raise ValueError("Need at least 8 workouts with target values for reliable ML training.")

    X = usable[feature_columns]
    y = usable[target_column]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)

    regressor = RandomForestRegressor(n_estimators=300, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    kmeans_features = ["ride_duration", "avg_heart_rate", "avg_output", "avg_resistance"]
    cluster_frame = training_df[kmeans_features].apply(pd.to_numeric, errors="coerce")
    cluster_frame = cluster_frame.fillna(cluster_frame.median(numeric_only=True))

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

    model_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": regressor, "imputer": imputer, "features": feature_columns}, model_dir / "peloton_work_model.joblib")
    cluster_summary.to_csv(model_dir / "cluster_summary.csv")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Peloton ML Insights\n\n")
        f.write("## Model Quality\n")
        f.write(f"- MAE (total_work prediction): {mae:.2f}\n")
        f.write(f"- R^2: {r2:.3f}\n\n")
        f.write("## Top Predictors of Workout Output\n")
        for feature, importance in importances.items():
            f.write(f"- {feature}: {importance:.3f}\n")
        f.write(f"\n## Workout Segments (KMeans, k={n_clusters})\n")
        f.write(cluster_summary.to_markdown())
        f.write("\n")

    return {
        "rows_used_for_training": int(len(usable)),
        "mae": mae,
        "r2": r2,
        "cluster_count": int(n_clusters),
    }
