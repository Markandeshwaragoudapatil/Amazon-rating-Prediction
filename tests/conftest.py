"""Shared pytest fixtures."""

import pytest
from collections.abc import Iterator

from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Provide one small local Spark session for focused transformation tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("AmazonRatingPredictionTests")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
