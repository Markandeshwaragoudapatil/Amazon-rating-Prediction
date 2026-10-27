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
    builder = SparkSession.builder.appName(app_name)
    # Keep the settings that were used on the original Windows machine
    # configurable. The Hadoop permissions options are retained for compatibility.
    builder = (
        builder.config("spark.hadoop.fs.permissions.enabled", "false")
        .config("spark.hadoop.fs.permissions.umask-mode", "022")
    )
    executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "2g")
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    return (
        builder.config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .getOrCreate()
    )
