from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    peloton_username: str | None = Field(default=None, alias="PELOTON_USERNAME")
    peloton_password: str | None = Field(default=None, alias="PELOTON_PASSWORD")
    peloton_since: str | None = Field(default=None, alias="PELOTON_SINCE")

    databricks_server_hostname: str | None = Field(default=None, alias="DATABRICKS_SERVER_HOSTNAME")
    databricks_http_path: str | None = Field(default=None, alias="DATABRICKS_HTTP_PATH")
    databricks_access_token: str | None = Field(default=None, alias="DATABRICKS_ACCESS_TOKEN")
    databricks_catalog: str = Field(default="main", alias="DATABRICKS_CATALOG")
    databricks_schema: str = Field(default="fitness", alias="DATABRICKS_SCHEMA")

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
