from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    peloton_username: str | None = Field(default=None, alias="PELOTON_USERNAME")
    peloton_password: str | None = Field(default=None, alias="PELOTON_PASSWORD")
    peloton_since: str | None = Field(default=None, alias="PELOTON_SINCE")
    peloton_max_workouts: int | None = Field(default=None, alias="PELOTON_MAX_WORKOUTS")
    write_local_staging: bool = Field(default=False, alias="WRITE_LOCAL_STAGING")

    databricks_server_hostname: str | None = Field(default=None, alias="DATABRICKS_SERVER_HOSTNAME")
    databricks_http_path: str | None = Field(default=None, alias="DATABRICKS_HTTP_PATH")
    databricks_access_token: str | None = Field(default=None, alias="DATABRICKS_ACCESS_TOKEN")
    databricks_catalog: str = Field(default="main", alias="DATABRICKS_CATALOG")
    databricks_schema: str = Field(default="fitness", alias="DATABRICKS_SCHEMA")
    use_databricks_spark: bool = Field(default=False, alias="USE_DATABRICKS_SPARK")
    databricks_artifact_base_path: str = Field(
        default="/dbfs/FileStore/peloton_analytics",
        alias="DATABRICKS_ARTIFACT_BASE_PATH",
    )
    mlflow_enabled: bool = Field(default=True, alias="MLFLOW_ENABLED")
    mlflow_experiment_name: str | None = Field(default=None, alias="MLFLOW_EXPERIMENT_NAME")
    mlflow_run_name: str | None = Field(default=None, alias="MLFLOW_RUN_NAME")
    mlflow_registered_model_name: str | None = Field(default=None, alias="MLFLOW_REGISTERED_MODEL_NAME")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def require_peloton(self) -> None:
        if not self.peloton_username or not self.peloton_password:
            raise ValueError("Missing Peloton credentials: set PELOTON_USERNAME and PELOTON_PASSWORD.")

    def require_databricks(self) -> None:
        if not self.databricks_server_hostname or not self.databricks_http_path or not self.databricks_access_token:
            raise ValueError(
                "Missing Databricks SQL credentials: set DATABRICKS_SERVER_HOSTNAME, "
                "DATABRICKS_HTTP_PATH, and DATABRICKS_ACCESS_TOKEN."
            )


def get_settings() -> Settings:
    return Settings()
