from pyspark.sql import DataFrame
from pyspark.sql import functions as F


TRAINING_COLUMNS = (
    "Text",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
    "Score",
)
PREDICTION_COLUMNS = (
    "Text",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
)


def validate_required_columns(
    df: DataFrame, required_columns: tuple[str, ...], *, context: str
) -> None:
    """Raise an actionable error when a dataframe lacks required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{context} requires these columns: {missing}.")


def _clean_nonnegative_integer(df: DataFrame, column: str) -> DataFrame:
    """Keep digit-only values and map other values to zero, matching prior behavior."""
    value_as_text = F.col(column).cast("string")
    return df.withColumn(
        column,
        F.when(value_as_text.rlike(r"^[0-9]+$"), value_as_text)
        .otherwise(F.lit("0"))
        .cast("int"),
    )


def prepare_review_data(df: DataFrame, *, include_label: bool) -> DataFrame:
    """Clean common training/inference inputs and optionally create the label."""
    required_columns = TRAINING_COLUMNS if include_label else PREDICTION_COLUMNS
    context = "Training data" if include_label else "Prediction data"
    validate_required_columns(df, required_columns, context=context)

    df = _clean_nonnegative_integer(df, "HelpfulnessNumerator")
    df = _clean_nonnegative_integer(df, "HelpfulnessDenominator")
    df = df.filter(F.col("Text").isNotNull() & (F.length("Text") > 0))
    df = df.withColumn(
        "helpfulness_ratio",
        F.when(
            F.col("HelpfulnessDenominator") > 0,
            F.col("HelpfulnessNumerator") / F.col("HelpfulnessDenominator"),
        ).otherwise(F.lit(0.0)),
    )

    if include_label:
        df = _clean_nonnegative_integer(df, "Score")
        df = df.withColumn("Score", F.col("Score").cast("float"))
        df = df.withColumn("label", F.col("Score"))
    return df


def basic_clean(df: DataFrame) -> DataFrame:
    """Clean training rows, preserving the original invalid-score-to-zero policy."""
    return prepare_review_data(df, include_label=True)


def prepare_prediction_input(df: DataFrame) -> DataFrame:
    """Prepare inference inputs using the same numeric and ratio logic as training."""
    return prepare_review_data(df, include_label=False)
