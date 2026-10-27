"""Train and evaluate the Amazon review rating model."""

import argparse
import os
from pathlib import Path

from .model import train_and_evaluate
from .preprocess import basic_clean
from .utils import PROJECT_ROOT, create_spark_session, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Training CSV (default: data/amazon_reviews.csv).")
    parser.add_argument(
        "--model-dir",
        help="Directory for the saved Spark CrossValidatorModel.",
    )
    row_limit = parser.add_mutually_exclusive_group()
    row_limit.add_argument(
        "--max-rows",
        type=int,
        help="Maximum number of source rows to use; 0 means the full dataset.",
    )
    row_limit.add_argument(
        "--full-dataset",
        action="store_true",
        help="Use every row instead of the default 60,000-row limit.",
    )
    return parser


def _row_limit(value: int | None, full_dataset: bool) -> int:
    """Resolve the CLI/environment row limit and reject negative values."""
    if full_dataset:
        return 0
    configured_value = value
    if configured_value is None:
        try:
            configured_value = int(os.getenv("AMAZON_RATING_MAX_ROWS", "60000"))
        except ValueError as error:
            raise ValueError(
                "AMAZON_RATING_MAX_ROWS must be a non-negative integer."
            ) from error
    if configured_value < 0:
        raise ValueError("The row limit must be non-negative; use 0 for the full dataset.")
    return configured_value


def run_training(data_path: Path, model_path: Path, max_rows: int) -> None:
    """Run training and always stop the Spark session."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {data_path}. "
            "Provide --data or set AMAZON_RATING_DATA."
        )

    spark = create_spark_session("Amazon Rating Prediction")
    try:
        print(f"Project root  : {PROJECT_ROOT}")
        print(f"Data path     : {data_path}")
        print(f"Model save to : {model_path}")
        print("\n=== Loading data ===")
        df = spark.read.csv(str(data_path), header=True, inferSchema=True)
        source_row_count = df.count()
        print(f"Loaded {source_row_count} rows.")
        if max_rows:
            df = df.limit(max_rows)
            print(f"Using at most {max_rows} source rows.")
        else:
            print("Using the full dataset.")

        print("\n=== Cleaning data ===")
        df = basic_clean(df)
        print(f"After cleaning: {df.count()} rows.")

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        print(f"\nTrain size: {train_df.count()}  |  Test size: {test_df.count()}")
        train_and_evaluate(train_df, test_df, model_path)
        print(f"\nTraining complete. Model saved at: {model_path}")
    finally:
        spark.stop()


def main(argv: list[str] | None = None) -> None:
    """Parse options and start model training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        max_rows = _row_limit(args.max_rows, args.full_dataset)
    except ValueError as error:
        parser.error(str(error))

    data_path = resolve_project_path(
        args.data,
        default="data/amazon_reviews.csv",
        environment_variable="AMAZON_RATING_DATA",
    )
    model_path = resolve_project_path(
        args.model_dir,
        default="models/rating_gbt_model",
        environment_variable="AMAZON_RATING_MODEL_DIR",
    )
    run_training(data_path, model_path, max_rows)


if __name__ == "__main__":
    main()
