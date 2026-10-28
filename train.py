
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/"
    "mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv"
)

OUTPUT_FILE = BASE_DIR / "model_xgb.bin"

SPLIT_SEED = 11
MODEL_SEED = 24

N_ESTIMATORS = 75
MAX_DEPTH = 3
LEARNING_RATE = 0.1


# --------------------------------------------------
# Data preparation
# --------------------------------------------------
def load_and_clean(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = df.columns.str.lower()

    df["status"] = df["status"].map({
        1: "ok",
        2: "default",
        0: "unk",
    })

    df["home"] = df["home"].map({
        1: "rent",
        2: "owner",
        3: "private",
        4: "ignore",
        5: "parents",
        6: "other",
        0: "unk",
    })

    df["marital"] = df["marital"].map({
        1: "single",
        2: "married",
        3: "widow",
        4: "separated",
        5: "divorced",
        0: "unk",
    })

    df["records"] = df["records"].map({
        1: "no",
        2: "yes",
        0: "unk",
    })

    df["job"] = df["job"].map({
        1: "fixed",
        2: "partime",
        3: "freelance",
        4: "others",
        0: "unk",
    })

    for column in ["income", "assets", "debt"]:
        df[column] = df[column].replace(99999999.0, np.nan)

    df = df[df["status"] != "unk"].reset_index(drop=True)

    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


# --------------------------------------------------
# Model pipeline
# --------------------------------------------------
def make_pipe_xgb():
    return make_pipeline(
        DictVectorizer(sparse=False),
        SimpleImputer(strategy="median"),
        xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            eval_metric="auc",
            random_state=MODEL_SEED,
            n_jobs=-1,
        ),
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate_model(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n{name}")
    print("-" * 45)
    print(f"Threshold:  {threshold:.2f}")
    print(f"ROC-AUC:    {roc_auc_score(y_true, probabilities):.4f}")
    print(f"PR-AUC:     {average_precision_score(y_true, probabilities):.4f}")
    print(f"Precision:  {precision_score(y_true, predictions, zero_division=0):.4f}")
    print(f"Recall:     {recall_score(y_true, predictions, zero_division=0):.4f}")
    print(f"F1:         {f1_score(y_true, predictions, zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))


def main():
    print("Loading and cleaning data...")

    df = load_and_clean()

    X = df.drop(columns=["status"])
    y = (df["status"] == "default").astype(int)

    # First split: 80% development data, 20% final test data.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=y,
    )

    # Split the 80% development data into 60% train and 20% validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=SPLIT_SEED,
        stratify=y_dev,
    )

    print(
        f"Rows — train: {len(X_train)}, "
        f"validation: {len(X_val)}, test: {len(X_test)}"
    )

    # Fit the pipeline using training data only.
    print("\nTraining XGBoost pipeline...")

    model = make_pipe_xgb()
    model.fit(to_records(X_train), y_train)

    # Select the threshold using validation data only.
    print("\nSelecting threshold on validation data...")

    val_probabilities = model.predict_proba(to_records(X_val))[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)

    scores = [
        f1_score(
            y_val,
            (val_probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(scores))
    best_threshold = float(thresholds[best_index])

    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation F1:      {scores[best_index]:.4f}")

    evaluate_model(
        "VALIDATION RESULTS",
        y_val,
        val_probabilities,
        best_threshold,
    )

    # Evaluate the selected threshold once on the untouched test set.
    test_probabilities = model.predict_proba(to_records(X_test))[:, 1]

    evaluate_model(
        "FINAL TEST RESULTS",
        y_test,
        test_probabilities,
        best_threshold,
    )

    # Save the pipeline, selected threshold, and metadata together.
    artifact = {
        "model": model,
        "threshold": best_threshold,
        "metadata": {
            "model_name": "CreditLens XGBoost",
            "version": "2.0",
            "target": "default",
            "threshold_metric": "validation_f1",
            "split_seed": SPLIT_SEED,
            "model_seed": MODEL_SEED,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
        },
    }

    with OUTPUT_FILE.open("wb") as f_out:
        pickle.dump(artifact, f_out)

    print(f"\nSaved model artifact to: {OUTPUT_FILE}")
    print(f"Saved threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()
