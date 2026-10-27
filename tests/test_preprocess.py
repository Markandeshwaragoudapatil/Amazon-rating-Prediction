"""Focused tests for Spark review cleaning and feature preparation."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from src.preprocess import basic_clean, prepare_prediction_input


@pytest.fixture
def review_schema() -> StructType:
    return StructType(
        [
            StructField("Text", StringType(), True),
            StructField("HelpfulnessNumerator", StringType(), True),
            StructField("HelpfulnessDenominator", StringType(), True),
            StructField("Score", StringType(), True),
        ]
    )


def test_cleaning_converts_invalid_numbers_to_zero_and_creates_label(
    spark: SparkSession, review_schema: StructType
) -> None:
    source = spark.createDataFrame(
        [
            ("Helpful review", "2", "4", "5"),
            ("Invalid values", "not-a-number", "?", "invalid-score"),
        ],
        review_schema,
    )

    rows = {
        row.Text: row
        for row in basic_clean(source).select(
            "Text",
            "HelpfulnessNumerator",
            "HelpfulnessDenominator",
            "Score",
            "helpfulness_ratio",
            "label",
        ).collect()
    }

    assert rows["Helpful review"].HelpfulnessNumerator == 2
    assert rows["Helpful review"].HelpfulnessDenominator == 4
    assert rows["Helpful review"].helpfulness_ratio == pytest.approx(0.5)
    assert rows["Helpful review"].label == pytest.approx(5.0)
    assert rows["Invalid values"].HelpfulnessNumerator == 0
    assert rows["Invalid values"].HelpfulnessDenominator == 0
    assert rows["Invalid values"].helpfulness_ratio == pytest.approx(0.0)
    assert rows["Invalid values"].Score == pytest.approx(0.0)
    assert rows["Invalid values"].label == pytest.approx(0.0)


def test_cleaning_removes_null_and_empty_text(
    spark: SparkSession, review_schema: StructType
) -> None:
    source = spark.createDataFrame(
        [
            ("Keep this review", "0", "0", "3"),
            (None, "0", "0", "4"),
            ("", "0", "0", "2"),
        ],
        review_schema,
    )

    cleaned_text = [row.Text for row in basic_clean(source).select("Text").collect()]
    assert cleaned_text == ["Keep this review"]


def test_zero_helpfulness_denominator_produces_zero_ratio(
    spark: SparkSession, review_schema: StructType
) -> None:
    source = spark.createDataFrame(
        [("No votes", "3", "0", "4")],
        review_schema,
    )

    row = basic_clean(source).select("helpfulness_ratio").first()
    assert row.helpfulness_ratio == pytest.approx(0.0)


def test_prediction_preparation_uses_same_ratio_and_text_filtering(
    spark: SparkSession, review_schema: StructType
) -> None:
    source = spark.createDataFrame(
        [
            ("Half found helpful", "1", "2", "ignored"),
            ("No votes", "2", "0", "ignored"),
            ("", "1", "2", "ignored"),
        ],
        review_schema,
    ).drop("Score")

    rows = {
        row.Text: row.helpfulness_ratio
        for row in prepare_prediction_input(source)
        .select("Text", "helpfulness_ratio")
        .collect()
    }
    assert rows == {
        "Half found helpful": pytest.approx(0.5),
        "No votes": pytest.approx(0.0),
    }


def test_training_and_prediction_report_missing_columns(spark: SparkSession) -> None:
    source = spark.createDataFrame([("Text only",)], ["Text"])

    with pytest.raises(ValueError, match="HelpfulnessNumerator"):
        basic_clean(source)
    with pytest.raises(ValueError, match="HelpfulnessNumerator"):
        prepare_prediction_input(source)
