"""Load a saved Spark model and predict ratings for reviews."""

import argparse
import os
from pathlib import Path
from typing import Protocol

from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import DataFrame, SparkSession

from .preprocess import prepare_prediction_input
from .utils import create_spark_session, resolve_project_path


class PredictionModel(Protocol):
    """Structural interface required by the prediction helper."""

    def transform(self, dataset: DataFrame) -> DataFrame: ...


SAMPLE_REVIEWS = [
    {
        "Text": "Amazing product, really loved it!",
        "HelpfulnessNumerator": 5,
        "HelpfulnessDenominator": 6,
    },
    {
        "Text": "Worst purchase ever, total waste.",
        "HelpfulnessNumerator": 0,
        "HelpfulnessDenominator": 1,
    },
    {
        "Text": "It was okay, nothing special.",
        "HelpfulnessNumerator": 2,
        "HelpfulnessDenominator": 3,
    },
]


def load_model(model_path: str | Path) -> CrossValidatorModel:
    """Load the CrossValidatorModel written by the training pipeline."""
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Spark model directory not found: {path}. "
            "Train a model with `python -m src.main` or set --model-dir."
        )
    return CrossValidatorModel.load(str(path))


def predict_reviews(model: PredictionModel, df: DataFrame) -> DataFrame:
    """Validate and transform review inputs, returning only requested output columns."""
    prepared_df = prepare_prediction_input(df)
    predictions = model.transform(prepared_df)
    required_output_columns = {"Text", "prediction"}
    missing_output_columns = required_output_columns.difference(predictions.columns)
    if missing_output_columns:
        missing = ", ".join(sorted(missing_output_columns))
        raise ValueError(f"Model predictions are missing expected output columns: {missing}.")
    return predictions.select("Text", "prediction")


def build_parser() -> argparse.ArgumentParser:
    """Build the prediction command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        help="Directory containing the saved Spark CrossValidatorModel.",
    )
    parser.add_argument(
        "--input",
        help="Optional input CSV with Text and helpfulness columns; defaults to sample reviews.",
    )
    parser.add_argument(
        "--output",
        help="Optional output directory for Spark CSV predictions.",
    )
    return parser


def _read_reviews(spark: SparkSession, input_path: Path | None) -> DataFrame:
    if input_path is None:
        return spark.createDataFrame(SAMPLE_REVIEWS)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Prediction input CSV not found: {input_path}. "
            "Provide a valid --input path."
        )
    return spark.read.csv(str(input_path), header=True, inferSchema=True)


def main(argv: list[str] | None = None) -> None:
    """Load the saved model, predict ratings, and optionally write a CSV directory."""
    args = build_parser().parse_args(argv)
    model_path = resolve_project_path(
        args.model_dir,
        default="models/rating_gbt_model",
        environment_variable="AMAZON_RATING_MODEL_DIR",
    )
    input_override = args.input or os.getenv("AMAZON_RATING_INPUT")
    input_path = (
        resolve_project_path(
            input_override,
            default="",
            environment_variable="AMAZON_RATING_INPUT",
        )
        if input_override
        else None
    )
    output_override = args.output or os.getenv("AMAZON_RATING_OUTPUT")
    output_path = (
        resolve_project_path(
            output_override,
            default="predictions",
            environment_variable="AMAZON_RATING_OUTPUT",
        )
        if output_override
        else None
    )

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Spark model directory not found: {model_path}. "
            "Train a model with `python -m src.main` or set --model-dir."
        )
    if input_path is not None and not input_path.is_file():
        raise FileNotFoundError(
            f"Prediction input CSV not found: {input_path}. "
            "Provide a valid --input path."
        )

    spark = create_spark_session("Predict Amazon Ratings")
    try:
        model = load_model(model_path)
        reviews = _read_reviews(spark, input_path)
        predictions = predict_reviews(model, reviews)
        if output_path is None:
            predictions.show(truncate=False)
        else:
            predictions.write.mode("overwrite").option("header", True).csv(
                str(output_path)
            )
            print(f"Predictions written to: {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
