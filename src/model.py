"""Spark ML pipeline construction, training, persistence, and evaluation."""

import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    VectorAssembler,
)

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder


def train_and_evaluate(
    train_df: DataFrame, test_df: DataFrame, model_path: str | Path
) -> tuple[CrossValidatorModel, tuple[float, float, float]]:
    """Fit the existing cross-validated pipeline, save it, and report test metrics."""
    print("\n=== Building pipeline components ===")

    # -----------------------------
    # TEXT PROCESSING STAGES
    # -----------------------------
    tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="tf_features",
        numFeatures=2**12  # 4096 dimensions
    )

    idf = IDF(inputCol="tf_features", outputCol="text_features")

    assembler = VectorAssembler(
        inputCols=["text_features", "helpfulness_ratio"],
        outputCol="features",
    )

    # -----------------------------
    # GBT REGRESSOR
    # -----------------------------
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="label",
        maxDepth=4,
        maxIter=20,
        stepSize=0.1,
        seed=42,
    )

    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, assembler, gbt])

    # -----------------------------
    # PARAM GRID FOR GBT ONLY
    # -----------------------------
    print("\n=== Setting up hyperparameter grid ===")

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 4, 5])
        .addGrid(gbt.maxIter, [10, 20])
        .build()
    )

    evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )

    # -----------------------------
    # CROSS VALIDATION SETUP
    # -----------------------------
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=2,
        parallelism=1
    )

    # -----------------------------
    # TRAINING WITH TIMER
    # -----------------------------
    print("\n=== Training model... ===")
    start = time.time()

    model = cv.fit(train_df)

    end = time.time()
    print(f"\nTraining completed in {end - start:.2f} seconds.")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    model_directory = Path(model_path)
    model_directory.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to: {model_directory}")
    model.write().overwrite().save(str(model_directory))

    # -----------------------------
    # EVALUATION
    # -----------------------------
    print("\n=== Evaluating on Test Set ===")
    preds = model.transform(test_df)

    rmse = evaluator.evaluate(preds)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(preds)
    r2 = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2"
    ).evaluate(preds)

    print("\n=== Model Performance ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    return model, (rmse, mae, r2)
