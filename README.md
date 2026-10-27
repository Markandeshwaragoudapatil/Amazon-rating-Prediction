# Amazon Rating Prediction

An educational PySpark project that predicts a numeric Amazon review rating from the review text and the review's helpfulness ratio. The project keeps the existing Spark ML workflow and exposes separate training and prediction commands.

## How it works

The training program reads a CSV, cleans the expected fields, removes rows with null or empty review text, and derives the label from `Score`. It then makes the existing seeded 80/20 split and fits a Spark ML pipeline:

1. `Tokenizer`
2. `StopWordsRemover`
3. `HashingTF` with 4,096 features
4. `IDF`
5. `VectorAssembler` combining text features and helpfulness ratio
6. `GBTRegressor`

`CrossValidator` evaluates the existing grid of GBT depth `[3, 4, 5]` and iteration count `[10, 20]` over two folds. Evaluation on the held-out split reports RMSE, MAE, and R². These scores are not published here because no verified evaluation result is available.

Prediction loads the Spark `CrossValidatorModel` produced by training; it does not train a model automatically. Training and inference share the same helpfulness-ratio preparation.

## Technologies

- Python
- PySpark 4.0.1 and Spark ML
- pytest

## Data expectations and preprocessing

Training CSVs must have these headers:

| Column | Use |
| --- | --- |
| `Text` | Review text used by the tokenizer |
| `HelpfulnessNumerator` | Count of helpful votes |
| `HelpfulnessDenominator` | Total helpfulness votes |
| `Score` | Numeric rating used as the label |

The input is read with Spark CSV header and schema inference enabled. Helpfulness numerator, denominator, and score values are accepted only when they are digit-only nonnegative integers. **The existing behavior maps any other value (including missing, negative, decimal-form, or non-numeric values) to zero.** Consequently, an invalid `Score` becomes a zero-valued training label rather than being dropped. This policy is retained for compatibility, but should be reviewed before using the model for consequential decisions.

Rows with null or zero-length `Text` are removed. A zero or invalid helpfulness denominator produces a helpfulness ratio of `0.0`; otherwise the ratio is numerator divided by denominator. Prediction CSVs need `Text`, `HelpfulnessNumerator`, and `HelpfulnessDenominator`; they do not need `Score`.

### Dataset source, attribution, and license

The supplied CSV is approximately 300.9 MB and contains fields such as product/user identifiers and profile names. Its original source, attribution terms, and redistribution license are not established by the files currently in this repository. Do not publish the dataset until its origin, license, privacy implications, and redistribution rights have been verified. This repository ignores local data by default; arrange an authorized download or provide a local copy at `data/amazon_reviews.csv`.

## Prerequisites and environment

- Windows 11 with PowerShell
- Python 3.11 or 3.12 recommended for a conservative PySpark setup. PySpark 4.0.1 requires Python 3.9 or newer; this project has not been validated on Python 3.14.
- A compatible Java Development Kit. Use Java 17 or 21 for Spark 4.0.x; the Java 26 installation on the original development machine has not been validated.
- Memory and disk space appropriate for Spark. Training uses the existing 2 GB executor and 4 GB driver memory settings.

Check the Spark 4.0.1 documentation and your local Java/Python installation if using versions outside these recommendations. No WSL requirement is intended.

## Windows 11 setup (PowerShell)

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, allow it for the current PowerShell process only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

Place an authorized training CSV at `.\data\amazon_reviews.csv`, or pass another path using the command-line options below. Use `python -m ...` from the project root so imports resolve consistently.

## Training

Train using the default dataset, model directory, and 60,000-row cap:

```powershell
python -m src.main
```

Override paths and the row cap:

```powershell
python -m src.main --data .\data\amazon_reviews.csv --model-dir .\models\rating_gbt_model --max-rows 60000
```

To train using the full dataset, either use:

```powershell
python -m src.main --full-dataset
```

or set `--max-rows 0`. Negative row limits are rejected. The default 60,000-row cap can also be changed with `AMAZON_RATING_MAX_ROWS`.

Path environment overrides are `AMAZON_RATING_DATA` and `AMAZON_RATING_MODEL_DIR`. Command-line arguments take precedence. Relative paths resolve from the project root; absolute paths are accepted.

## Prediction

Predict the bundled sample reviews using the model in the default model directory:

```powershell
python -m src.predict
```

Predict rows in a CSV and write results to a Spark output directory:

```powershell
python -m src.predict --model-dir .\models\rating_gbt_model --input .\data\reviews_to_score.csv --output .\predictions
```

The input CSV must include `Text`, `HelpfulnessNumerator`, and `HelpfulnessDenominator`. The output is a Spark CSV directory containing `Text` and `prediction` columns (including Spark part files). `AMAZON_RATING_MODEL_DIR`, `AMAZON_RATING_INPUT`, and `AMAZON_RATING_OUTPUT` can be used as path defaults. A missing model produces an error with the training command; prediction never starts training implicitly.

The legacy script `python .\test_spark.py` remains as an alias for the sample prediction command, not as the automated test suite.

## Evaluation metrics

- **RMSE** (root mean squared error) penalizes larger errors more heavily; lower is better.
- **MAE** (mean absolute error) is the average absolute rating error; lower is better.
- **R²** compares the model with a constant-mean baseline; values closer to 1 generally indicate stronger fit, while negative values can indicate worse performance than that baseline.

No accuracy, metric value, training duration, or performance claim is asserted without a reproducible measured result.

## Dataset and model storage

The approximately 300.9 MB CSV exceeds GitHub's 100 MB per-file limit and is ignored by `.gitignore`. Do not work around the limit by committing it; verify its license and use an authorized download or external dataset-hosting process instead.

The generated Spark model directory is about 106 KB in the supplied workspace and is ignored as a local build artifact. The empty `models/rating_model.pkl` file is not loaded or written by the application and is not interchangeable with the Spark `CrossValidatorModel`. It is retained on disk but ignored; it can be removed later only after confirming it has no separate value. Prefer retraining for reproducibility. If a trained model must be distributed, package and version it as a release/download artifact or evaluate Git LFS rather than committing generated Spark files blindly.

## Tests

Run the lightweight test suite:

```powershell
python -m pytest
```

Tests use small local Spark dataframes and a stub prediction model. They do not load the complete dataset or run model training/cross-validation. A compatible Java runtime is still required by the Spark integration tests.

## Limitations and future work

- Verify the exact dataset source/license and exclude or protect personal/user-identifying fields before redistribution.
- Invalid target scores currently become label `0`; consider an explicit label validation/drop policy as a separately reviewed behavior change.
- CSV schema inference can be costly and can infer inconsistent types; a declared schema could improve repeatability.
- The default first-60,000-row limit is for a manageable training run and may not represent the full dataset.
- Model evaluation is a single seeded train/test split with the existing small cross-validation grid.
- Future improvements could include reproducible experiment metadata, a model/data versioning strategy, stronger input schema checks, and documented measured baseline metrics.
