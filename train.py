import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/"
    "mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv"
)

OUTPUT_FILE = "model_xgb.bin"
SPLIT_SEED = 11  
MODEL_SEED = 24

# Best hyperparameters from the notebook
N_ESTIMATORS = 75
MAX_DEPTH = 3
LEARNING_RATE = 0.1


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_and_clean(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = df.columns.str.lower()

    df["status"] = df["status"].map({1: "ok", 2: "default", 0: "unk"})

    df["home"] = df["home"].map(
        {1: "rent", 2: "owner", 3: "private", 4: "ignore",
         5: "parents", 6: "other", 0: "unk"}
    )

    df["marital"] = df["marital"].map(
        {1: "single", 2: "married", 3: "widow",
         4: "separated", 5: "divorced", 0: "unk"}
    )

    df["records"] = df["records"].map({1: "no", 2: "yes", 0: "unk"})

    df["job"] = df["job"].map(
        {1: "fixed", 2: "partime", 3: "freelance", 4: "others", 0: "unk"}
    )

    # 99999999 is a placeholder for "missing"
    for c in ["income", "assets", "debt"]:
        df[c] = df[c].replace(99999999.0, np.nan)

    # Drop rows where the target is unknown
    df = df[df["status"] != "unk"].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=SPLIT_SEED
    )

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = (df_full_train["status"] == "default").astype(int).values
    y_test = (df_test["status"] == "default").astype(int).values

    X_full_train = df_full_train.drop(columns="status")
    X_test = df_test.drop(columns="status")

    return X_full_train, X_test, y_full_train, y_test


def to_records(df: pd.DataFrame) -> list[dict]:

    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_pipe_xgb(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE):
   
    return make_pipeline(
        DictVectorizer(sparse=False),
        SimpleImputer(strategy="median"),
        xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="auc",
            random_state=MODEL_SEED,
            n_jobs=-1
        )
    )


def main():
    
    print("Loading and cleaning data...")
    df = load_and_clean()

    X_full_train_df, X_test_df, y_full_train, y_test = split_data(df)
    X_full_train = to_records(X_full_train_df)
    X_test = to_records(X_test_df)

    print("Training final XGBoost pipeline...")
    model = make_pipe_xgb()
    model.fit(X_full_train, y_full_train)

    train_auc = roc_auc_score(y_full_train, model.predict_proba(X_full_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"train_auc: {train_auc:.4f}")
    print(f"test_auc:  {test_auc:.4f}")

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"Model saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()