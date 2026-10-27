"""Shared project paths and Spark session configuration."""

import os
from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(
    override: str | Path | None,
    *,
    default: str,
    environment_variable: str,
) -> Path:
    """Resolve a CLI override or environment value relative to the project root."""
    configured_path = override or os.getenv(environment_variable) or default
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def create_spark_session(app_name: str) -> SparkSession:
    """Create the local Spark session using the project's existing memory settings."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.permissions.enabled", "false")
        .config("spark.hadoop.fs.permissions.umask-mode", "022")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
