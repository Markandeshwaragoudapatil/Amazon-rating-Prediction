"""Tests for path resolution and the prediction interface."""

from pathlib import Path

import pytest
from pyspark.sql import DataFrame, functions as F
from pyspark.sql import SparkSession

from src.predict import load_model, predict_reviews
from src.utils import PROJECT_ROOT, resolve_project_path


def test_default_project_path_is_relative_to_project_root() -> None:
    path = resolve_project_path(
        None,
        default="models/rating_gbt_model",
        environment_variable="AMAZON_RATING_MODEL_DIR",
    )

    assert path == PROJECT_ROOT / "models" / "rating_gbt_model"


def test_path_override_supports_relative_absolute_and_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = resolve_project_path(
        "custom/reviews.csv",
        default="data/amazon_reviews.csv",
        environment_variable="AMAZON_RATING_DATA",
    )
    assert relative == PROJECT_ROOT / "custom" / "reviews.csv"

    absolute_path = tmp_path / "reviews.csv"
    assert resolve_project_path(
        absolute_path,
        default="data/amazon_reviews.csv",
        environment_variable="AMAZON_RATING_DATA",
    ) == absolute_path

    monkeypatch.setenv("AMAZON_RATING_DATA", str(absolute_path))
    assert resolve_project_path(
        None,
        default="data/amazon_reviews.csv",
        environment_variable="AMAZON_RATING_DATA",
    ) == absolute_path


def test_missing_model_has_actionable_error(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError, match="Train a model"):
        load_model(missing_model)


def test_prediction_validates_inputs_and_returns_expected_columns(
    spark: SparkSession,
) -> None:
    source = spark.createDataFrame(
        [("A useful review", 1, 2)],
        ["Text", "HelpfulnessNumerator", "HelpfulnessDenominator"],
    )

    class FakeModel:
        def transform(self, dataframe: DataFrame) -> DataFrame:
            return dataframe.withColumn("prediction", F.lit(4.25))

    predictions = predict_reviews(FakeModel(), source)
    assert predictions.columns == ["Text", "prediction"]
    assert predictions.first().prediction == pytest.approx(4.25)

    invalid_source = spark.createDataFrame([("Missing helpfulness",)], ["Text"])
    with pytest.raises(ValueError, match="HelpfulnessNumerator"):
        predict_reviews(FakeModel(), invalid_source)


def test_prediction_rejects_models_without_prediction_column(
    spark: SparkSession,
) -> None:
    source = spark.createDataFrame(
        [("A useful review", 1, 2)],
        ["Text", "HelpfulnessNumerator", "HelpfulnessDenominator"],
    )

    class InvalidModel:
        def transform(self, dataframe: DataFrame) -> DataFrame:
            return dataframe

    with pytest.raises(ValueError, match="prediction"):
        predict_reviews(InvalidModel(), source)


def test_spark_memory_settings_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.utils import create_spark_session

    # Exercise SparkSession.builder configuration without starting a JVM.
    class FakeSession:
        def __init__(self):
            self.settings = {}

        def appName(self, value):
            return self

        def config(self, key, value):
            self.settings[key] = value
            return self

        def getOrCreate(self):
            return self.settings

    fake = FakeSession()
    monkeypatch.setattr("src.utils.SparkSession.builder", fake)
    monkeypatch.setenv("SPARK_EXECUTOR_MEMORY", "1g")
    monkeypatch.setenv("SPARK_DRIVER_MEMORY", "2g")

    settings = create_spark_session("test-config")
    assert settings["spark.executor.memory"] == "1g"
    assert settings["spark.driver.memory"] == "2g"
