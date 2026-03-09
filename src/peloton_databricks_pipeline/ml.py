from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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


def _split_training_frame(
    X: pd.DataFrame,
    y: pd.Series,
    timestamp_series: pd.Series | None,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if timestamp_series is not None:
        ts = pd.to_datetime(timestamp_series, errors="coerce")
        if len(ts) >= 8 and bool(ts.notna().all()):
            ordered_idx = ts.sort_values().index.tolist()
            split_at = int(len(ordered_idx) * (1 - test_size))
            split_at = max(2, min(split_at, len(ordered_idx) - 2))
            train_idx = ordered_idx[:split_at]
            test_idx = ordered_idx[split_at:]
            return (
                X.loc[train_idx],
                X.loc[test_idx],
                y.loc[train_idx],
                y.loc[test_idx],
                "time_ordered",
            )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, "random"


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Backward compatibility for older sklearn versions.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(feature_columns: list[str], discipline_column: str) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                [discipline_column],
            ),
        ],
        remainder="drop",
    )


def _sanitize_mlflow_key(raw_key: str) -> str:
    allowed = []
    for char in raw_key:
        if char.isalnum() or char in {"_", ".", "/", "-", " "}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip().replace(" ", "_")


def _bucket_disciplines(
    discipline_series: pd.Series,
    kept_disciplines: set[str],
) -> pd.Series:
    normalized = discipline_series.fillna("unknown").astype(str).str.strip().str.lower()
    return normalized.where(normalized.isin(kept_disciplines), "other")


def _build_per_discipline_metrics(
    disciplines: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, dict[str, float]]:
    rows = pd.DataFrame({"discipline": disciplines, "actual": y_true, "pred": y_pred})
    rows = rows.dropna(subset=["actual", "pred"])
    if rows.empty:
        return {}

    per_discipline: dict[str, dict[str, float]] = {}
    for discipline, group in rows.groupby("discipline"):
        discipline_mae = float(mean_absolute_error(group["actual"], group["pred"]))
        if len(group) >= 2 and group["actual"].nunique() > 1:
            discipline_r2 = float(r2_score(group["actual"], group["pred"]))
        else:
            discipline_r2 = float("nan")
        per_discipline[str(discipline)] = {
            "test_rows": float(len(group)),
            "mae": discipline_mae,
            "r2": discipline_r2,
        }

    return per_discipline


def _assemble_two_stage_predictions(
    *,
    positive_probabilities: pd.Series,
    stage2_predictions: pd.Series,
    threshold: float,
) -> pd.Series:
    predicted_positive = positive_probabilities >= threshold
    combined = pd.Series(0.0, index=positive_probabilities.index, dtype=float)
    if bool(predicted_positive.any()):
        combined.loc[predicted_positive] = np.maximum(stage2_predictions.loc[predicted_positive], 0.0)
    return combined


def _tune_stage1_threshold(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    discipline_column: str,
    split_type: str,
) -> tuple[float, float | None, float | None]:
    n_rows = len(X_train)
    if n_rows < 40:
        return 0.5, None, None

    val_rows = max(8, int(n_rows * 0.2))
    val_rows = min(val_rows, n_rows - 8)
    if val_rows < 8:
        return 0.5, None, None

    if split_type == "time_ordered":
        ordered_idx = list(X_train.index)
    else:
        ordered_idx = list(X_train.sample(frac=1.0, random_state=42).index)

    train_idx = ordered_idx[:-val_rows]
    val_idx = ordered_idx[-val_rows:]
    X_fit = X_train.loc[train_idx]
    y_fit = y_train.loc[train_idx]
    X_val = X_train.loc[val_idx]
    y_val = y_train.loc[val_idx]

    y_fit_positive = (y_fit > 0).astype(int)
    positive_rows_fit = int((y_fit > 0).sum())
    if positive_rows_fit < 8:
        return 0.5, None, None

    stage1_tune = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(feature_columns, discipline_column)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    stage1_tune.fit(X_fit, y_fit_positive)

    stage2_tune = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(feature_columns, discipline_column)),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )
    stage2_tune.fit(X_fit.loc[y_fit > 0], y_fit.loc[y_fit > 0])

    positive_probabilities = pd.Series(stage1_tune.predict_proba(X_val)[:, 1], index=X_val.index)
    stage2_predictions = pd.Series(np.maximum(stage2_tune.predict(X_val), 0.0), index=X_val.index)

    # Conservative threshold search to control false positives on zero-heavy holdouts.
    thresholds = np.linspace(0.5, 0.95, 10)
    best_threshold = 0.5
    best_mae = float("inf")
    best_r2 = float("-inf")

    for threshold in thresholds:
        combined = _assemble_two_stage_predictions(
            positive_probabilities=positive_probabilities,
            stage2_predictions=stage2_predictions,
            threshold=float(threshold),
        )
        mae = float(mean_absolute_error(y_val, combined))
        if y_val.nunique() > 1:
            r2 = float(r2_score(y_val, combined))
        else:
            r2 = float("-inf")

        if mae < best_mae - 1e-9 or (
            abs(mae - best_mae) <= 1e-9 and (r2 > best_r2 or (abs(r2 - best_r2) <= 1e-9 and threshold > best_threshold))
        ):
            best_threshold = float(threshold)
            best_mae = mae
            best_r2 = r2

    if not np.isfinite(best_mae):
        return 0.5, None, None

    return best_threshold, best_mae, (None if not np.isfinite(best_r2) else best_r2)


def _log_mlflow_artifacts(
    *,
    enable_mlflow: bool,
    mlflow_experiment_name: str | None,
    mlflow_run_name: str | None,
    mlflow_registered_model_name: str | None,
    stage1_classifier_pipeline: Pipeline,
    stage2_regressor_pipeline: Pipeline,
    classifier_n_estimators: int,
    regressor_n_estimators: int,
    classifier_importances: pd.Series,
    regressor_importances: pd.Series,
    model_path: Path,
    cluster_summary_path: Path,
    report_path: Path,
    rows_used_for_training: int,
    stage2_train_rows: int,
    zero_target_rows: int,
    mae: float,
    r2: float,
    baseline_mae: float,
    baseline_r2: float,
    mae_improvement_vs_baseline: float,
    r2_improvement_vs_baseline: float,
    stage1_accuracy: float,
    stage1_precision: float,
    stage1_recall: float,
    stage1_f1: float,
    stage1_positive_rate_train: float,
    stage1_positive_rate_test: float,
    stage1_pred_positive_rate_test: float,
    stage1_threshold: float,
    threshold_tuning_mae: float | None,
    threshold_tuning_r2: float | None,
    split_type: str,
    cluster_count: int,
    per_discipline_metrics: dict[str, dict[str, float]],
    discipline_bucket_min_rows: int,
    discipline_kept_count: int,
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
                    "model_type": "two_stage_random_forest",
                    "classifier_n_estimators": classifier_n_estimators,
                    "regressor_n_estimators": regressor_n_estimators,
                    "train_rows": rows_used_for_training,
                    "stage2_train_rows": stage2_train_rows,
                    "zero_target_rows": zero_target_rows,
                    "feature_count_classifier": len(classifier_importances),
                    "feature_count_regressor": len(regressor_importances),
                    "cluster_count": cluster_count,
                    "split_type": split_type,
                    "discipline_bucket_min_rows": discipline_bucket_min_rows,
                    "discipline_kept_count": discipline_kept_count,
                    "target_strategy": "stage1_zero_vs_positive_then_stage2_regression",
                    "stage1_threshold": stage1_threshold,
                }
            )
            metrics_payload: dict[str, float] = {
                "mae": mae,
                "r2": r2,
                "baseline_mae": baseline_mae,
                "baseline_r2": baseline_r2,
                "mae_improvement_vs_baseline": mae_improvement_vs_baseline,
                "r2_improvement_vs_baseline": r2_improvement_vs_baseline,
                "stage1_accuracy": stage1_accuracy,
                "stage1_precision": stage1_precision,
                "stage1_recall": stage1_recall,
                "stage1_f1": stage1_f1,
                "stage1_positive_rate_train": stage1_positive_rate_train,
                "stage1_positive_rate_test": stage1_positive_rate_test,
                "stage1_pred_positive_rate_test": stage1_pred_positive_rate_test,
            }
            if threshold_tuning_mae is not None:
                metrics_payload["stage1_threshold_tuning_val_mae"] = threshold_tuning_mae
            if threshold_tuning_r2 is not None:
                metrics_payload["stage1_threshold_tuning_val_r2"] = threshold_tuning_r2
            mlflow.log_metrics(metrics_payload)

            for feature_name, importance in classifier_importances.items():
                safe_feature_name = _sanitize_mlflow_key(str(feature_name))
                mlflow.log_metric(f"classification_importance_{safe_feature_name}", float(importance))

            for feature_name, importance in regressor_importances.items():
                safe_feature_name = _sanitize_mlflow_key(str(feature_name))
                mlflow.log_metric(f"regression_importance_{safe_feature_name}", float(importance))

            for discipline, metrics in per_discipline_metrics.items():
                safe_discipline = _sanitize_mlflow_key(str(discipline))
                mlflow.log_metric(f"discipline_{safe_discipline}_test_rows", float(metrics["test_rows"]))
                mlflow.log_metric(f"discipline_{safe_discipline}_mae", float(metrics["mae"]))
                if np.isfinite(float(metrics["r2"])):
                    mlflow.log_metric(f"discipline_{safe_discipline}_r2", float(metrics["r2"]))

            # Keep stage-2 regressor as the canonical sklearn model artifact.
            if mlflow_registered_model_name:
                mlflow.sklearn.log_model(
                    sk_model=stage2_regressor_pipeline,
                    artifact_path="model",
                    registered_model_name=mlflow_registered_model_name,
                )
            else:
                mlflow.sklearn.log_model(sk_model=stage2_regressor_pipeline, artifact_path="model")

            # Log stage-1 classifier separately for reproducibility.
            mlflow.sklearn.log_model(sk_model=stage1_classifier_pipeline, artifact_path="stage1_classifier_model")

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
    discipline_column = "fitness_discipline"
    min_discipline_rows = 10

    target_column = "total_work"
    numeric_columns = list({target_column, *feature_columns, "avg_output"})
    for column in numeric_columns:
        if column in training_df.columns:
            training_df[column] = pd.to_numeric(training_df[column], errors="coerce")

    if discipline_column not in training_df.columns:
        training_df[discipline_column] = "unknown"

    usable = training_df[[target_column, *feature_columns, discipline_column]].copy()
    usable = usable.dropna(subset=[target_column])
    if len(usable) < 8:
        raise ValueError("Need at least 8 workouts with target values for reliable ML training.")

    X = usable[[*feature_columns, discipline_column]].copy()
    y = usable[target_column]
    timestamp_series = training_df.loc[usable.index, "created_at"] if "created_at" in training_df.columns else None

    X_train_raw, X_test_raw, y_train, y_test, split_type = _split_training_frame(
        X=X,
        y=y,
        timestamp_series=timestamp_series,
        test_size=0.25,
    )

    train_disciplines = X_train_raw[discipline_column].fillna("unknown").astype(str).str.strip().str.lower()
    discipline_counts = train_disciplines.value_counts()
    kept_disciplines = set(discipline_counts[discipline_counts >= min_discipline_rows].index.tolist())
    kept_disciplines.add("unknown")

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train[discipline_column] = _bucket_disciplines(X_train[discipline_column], kept_disciplines)
    X_test[discipline_column] = _bucket_disciplines(X_test[discipline_column], kept_disciplines)

    # Stage 1: classify whether a workout has non-zero total_work.
    y_train_positive = (y_train > 0).astype(int)
    y_test_positive = (y_test > 0).astype(int)

    stage1_threshold, threshold_tuning_mae, threshold_tuning_r2 = _tune_stage1_threshold(
        X_train=X_train,
        y_train=y_train,
        feature_columns=feature_columns,
        discipline_column=discipline_column,
        split_type=split_type,
    )

    stage1_classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )
    stage1_classifier_pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(feature_columns, discipline_column)),
            ("classifier", stage1_classifier),
        ]
    )
    stage1_classifier_pipeline.fit(X_train, y_train_positive)
    y_test_positive_proba = pd.Series(stage1_classifier_pipeline.predict_proba(X_test)[:, 1], index=y_test.index)
    y_test_positive_pred = (y_test_positive_proba >= stage1_threshold).astype(int)

    stage1_accuracy = float(accuracy_score(y_test_positive, y_test_positive_pred))
    stage1_precision = float(precision_score(y_test_positive, y_test_positive_pred, zero_division=0))
    stage1_recall = float(recall_score(y_test_positive, y_test_positive_pred, zero_division=0))
    stage1_f1 = float(f1_score(y_test_positive, y_test_positive_pred, zero_division=0))
    stage1_positive_rate_train = float(y_train_positive.mean())
    stage1_positive_rate_test = float(y_test_positive.mean())
    stage1_pred_positive_rate_test = float(y_test_positive_pred.mean())

    # Stage 2: regress total_work only on positive-target training rows.
    stage2_train_mask = y_train > 0
    stage2_train_rows = int(stage2_train_mask.sum())
    if stage2_train_rows < 8:
        raise ValueError(
            "Need at least 8 positive-target workouts in the train split for stage-2 regression training."
        )

    stage2_regressor = RandomForestRegressor(n_estimators=300, random_state=42)
    stage2_regressor_pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(feature_columns, discipline_column)),
            ("regressor", stage2_regressor),
        ]
    )
    stage2_regressor_pipeline.fit(X_train.loc[stage2_train_mask], y_train.loc[stage2_train_mask])

    stage2_predictions = pd.Series(np.maximum(stage2_regressor_pipeline.predict(X_test), 0.0), index=y_test.index)
    y_pred_series = _assemble_two_stage_predictions(
        positive_probabilities=y_test_positive_proba,
        stage2_predictions=stage2_predictions,
        threshold=stage1_threshold,
    )

    mae = float(mean_absolute_error(y_test, y_pred_series))
    r2 = float(r2_score(y_test, y_pred_series))

    baseline_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_r2 = float(r2_score(y_test, baseline_pred))
    mae_improvement_vs_baseline = baseline_mae - mae
    r2_improvement_vs_baseline = r2 - baseline_r2

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

    classifier_feature_columns = (
        stage1_classifier_pipeline.named_steps["preprocessor"].get_feature_names_out().astype(str).tolist()
    )
    regressor_feature_columns = (
        stage2_regressor_pipeline.named_steps["preprocessor"].get_feature_names_out().astype(str).tolist()
    )

    classifier_importances = pd.Series(
        stage1_classifier_pipeline.named_steps["classifier"].feature_importances_,
        index=classifier_feature_columns,
    ).sort_values(ascending=False)
    regressor_importances = pd.Series(
        stage2_regressor_pipeline.named_steps["regressor"].feature_importances_,
        index=regressor_feature_columns,
    ).sort_values(ascending=False)

    per_discipline_metrics = _build_per_discipline_metrics(
        disciplines=X_test[discipline_column],
        y_true=y_test,
        y_pred=y_pred_series,
    )

    model_dir, report_path = _ensure_artifact_paths(model_dir=model_dir, report_path=report_path)

    model_path = model_dir / "peloton_work_model.joblib"
    cluster_summary_path = model_dir / "cluster_summary.csv"

    joblib.dump(
        {
            "stage1_classifier": stage1_classifier_pipeline,
            "stage2_regressor": stage2_regressor_pipeline,
            "stage1_threshold": stage1_threshold,
            "features_numeric": feature_columns,
            "feature_discipline": discipline_column,
            "discipline_bucket_min_rows": min_discipline_rows,
            "disciplines_kept": sorted(kept_disciplines),
            "prediction_strategy": "if stage1 predicts 0 then output 0 else stage2 prediction",
        },
        model_path,
    )
    cluster_summary.to_csv(cluster_summary_path)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Peloton ML Insights\n\n")
        f.write("## Model Quality\n")
        f.write(f"- Split strategy: {split_type}\n")
        f.write("- Target strategy: stage-1 zero/non-zero classification, then stage-2 regression\n")
        f.write(f"- MAE (total_work prediction): {mae:.2f}\n")
        f.write(f"- R^2: {r2:.3f}\n\n")

        f.write("## Stage-1 Classification (total_work > 0)\n")
        f.write(f"- Probability threshold: {stage1_threshold:.3f}\n")
        f.write(f"- Accuracy: {stage1_accuracy:.3f}\n")
        f.write(f"- Precision: {stage1_precision:.3f}\n")
        f.write(f"- Recall: {stage1_recall:.3f}\n")
        f.write(f"- F1: {stage1_f1:.3f}\n")
        f.write(f"- Positive rate (train): {stage1_positive_rate_train:.3f}\n")
        f.write(f"- Positive rate (test): {stage1_positive_rate_test:.3f}\n")
        f.write(f"- Predicted positive rate (test): {stage1_pred_positive_rate_test:.3f}\n\n")
        if threshold_tuning_mae is not None:
            f.write(f"- Threshold tuning validation MAE: {threshold_tuning_mae:.2f}\n")
        if threshold_tuning_r2 is not None:
            f.write(f"- Threshold tuning validation R^2: {threshold_tuning_r2:.3f}\n")
        if threshold_tuning_mae is not None or threshold_tuning_r2 is not None:
            f.write("\n")

        f.write("## Baseline Comparison\n")
        f.write(f"- Baseline MAE (mean predictor): {baseline_mae:.2f}\n")
        f.write(f"- Baseline R^2: {baseline_r2:.3f}\n")
        f.write(f"- MAE improvement vs baseline: {mae_improvement_vs_baseline:.2f}\n")
        f.write(f"- R^2 improvement vs baseline: {r2_improvement_vs_baseline:.3f}\n\n")

        f.write("## Training Composition\n")
        f.write(f"- Rows used (non-null target): {len(usable)}\n")
        f.write(f"- Zero-target rows: {int((y == 0).sum())}\n")
        f.write(f"- Stage-2 train rows (target > 0): {stage2_train_rows}\n\n")

        f.write("## Discipline Encoding\n")
        f.write(f"- Discipline feature: {discipline_column}\n")
        f.write(f"- Rare discipline threshold (train rows): {min_discipline_rows}\n")
        f.write(f"- Distinct disciplines kept: {len(kept_disciplines)}\n\n")

        f.write("## Top Predictors (Stage-1 Classification)\n")
        for feature, importance in classifier_importances.head(12).items():
            f.write(f"- {feature}: {importance:.3f}\n")

        f.write("\n## Top Predictors (Stage-2 Regression)\n")
        for feature, importance in regressor_importances.head(12).items():
            f.write(f"- {feature}: {importance:.3f}\n")

        f.write("\n## Per-Discipline Holdout Metrics\n")
        if per_discipline_metrics:
            ordered_disciplines = sorted(
                per_discipline_metrics.items(),
                key=lambda item: item[1]["test_rows"],
                reverse=True,
            )
            for discipline, metrics in ordered_disciplines:
                r2_display = "n/a" if not np.isfinite(metrics["r2"]) else f"{metrics['r2']:.3f}"
                f.write(
                    f"- {discipline}: rows={int(metrics['test_rows'])}, "
                    f"MAE={metrics['mae']:.2f}, R^2={r2_display}\n"
                )
        else:
            f.write("- No per-discipline metrics available for this run.\n")

        f.write(f"\n## Workout Segments (KMeans, k={n_clusters})\n")
        f.write(cluster_summary.to_string())
        f.write("\n")

    mlflow_status, mlflow_run_id = _log_mlflow_artifacts(
        enable_mlflow=enable_mlflow,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        mlflow_registered_model_name=mlflow_registered_model_name,
        stage1_classifier_pipeline=stage1_classifier_pipeline,
        stage2_regressor_pipeline=stage2_regressor_pipeline,
        classifier_n_estimators=stage1_classifier.n_estimators,
        regressor_n_estimators=stage2_regressor.n_estimators,
        classifier_importances=classifier_importances,
        regressor_importances=regressor_importances,
        model_path=model_path,
        cluster_summary_path=cluster_summary_path,
        report_path=report_path,
        rows_used_for_training=int(len(usable)),
        stage2_train_rows=stage2_train_rows,
        zero_target_rows=int((y == 0).sum()),
        mae=mae,
        r2=r2,
        baseline_mae=baseline_mae,
        baseline_r2=baseline_r2,
        mae_improvement_vs_baseline=mae_improvement_vs_baseline,
        r2_improvement_vs_baseline=r2_improvement_vs_baseline,
        stage1_accuracy=stage1_accuracy,
        stage1_precision=stage1_precision,
        stage1_recall=stage1_recall,
        stage1_f1=stage1_f1,
        stage1_positive_rate_train=stage1_positive_rate_train,
        stage1_positive_rate_test=stage1_positive_rate_test,
        stage1_pred_positive_rate_test=stage1_pred_positive_rate_test,
        stage1_threshold=stage1_threshold,
        threshold_tuning_mae=threshold_tuning_mae,
        threshold_tuning_r2=threshold_tuning_r2,
        split_type=split_type,
        cluster_count=int(n_clusters),
        per_discipline_metrics=per_discipline_metrics,
        discipline_bucket_min_rows=min_discipline_rows,
        discipline_kept_count=len(kept_disciplines),
    )

    return {
        "rows_used_for_training": int(len(usable)),
        "stage2_train_rows": stage2_train_rows,
        "zero_target_rows": int((y == 0).sum()),
        "mae": mae,
        "r2": r2,
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2,
        "mae_improvement_vs_baseline": mae_improvement_vs_baseline,
        "r2_improvement_vs_baseline": r2_improvement_vs_baseline,
        "stage1_accuracy": stage1_accuracy,
        "stage1_precision": stage1_precision,
        "stage1_recall": stage1_recall,
        "stage1_f1": stage1_f1,
        "stage1_threshold": stage1_threshold,
        "stage1_threshold_tuning_val_mae": threshold_tuning_mae,
        "stage1_threshold_tuning_val_r2": threshold_tuning_r2,
        "split_type": split_type,
        "cluster_count": int(n_clusters),
        "artifact_model_dir": str(model_dir),
        "artifact_report_path": str(report_path),
        "mlflow_status": mlflow_status,
        "mlflow_run_id": mlflow_run_id,
    }
