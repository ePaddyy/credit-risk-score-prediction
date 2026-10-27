# Risk Score Prediction

A machine learning service that predicts the probability that a credit applicant will **default** on a loan. The final model is an XGBoost classifier, trained on the CreditScoring dataset, served with Flask and Gunicorn, and packaged in Docker.

## Problem

Lenders need to decide whether to approve a loan. Given an applicant's profile (job, income, assets, debt, loan amount, and so on), the model returns the probability of default, so a lender can rank applicants by risk instead of treating them all the same.

## Data

- **Source:** the CreditScoring dataset from the [mlbookcamp-code repository](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-06-trees).
- **Target:** `status` (`default` = 1, `ok` = 0). Rows with unknown status were removed.
- **Features:** `seniority`, `home`, `time`, `age`, `marital`, `records`, `job`, `expenses`, `income`, `assets`, `debt`, `amount`, `price`.
- **Cleaning:** numeric codes were mapped to readable categories, and the placeholder value `99999999` in `income`, `assets` and `debt` was treated as missing.

## Approach

1. Split the data 60% train / 20% validation / 20% test.
2. Build every model as a scikit-learn pipeline: `DictVectorizer` → `SimpleImputer(median)` → model (plus `StandardScaler` for logistic regression).
3. Tune hyperparameters with 5-fold stratified cross-validation, scoring on ROC AUC.
4. Retrain the best configuration of each model on train + validation, and compare them once on the held-out test set.

## Model comparison

| Model | Test AUC |
|---|---|
| Logistic Regression | 0.82582 |
| Decision Tree | 0.760144 |
| Random Forest | 0.82256 |
| **XGBoost (selected)** | 0.833188 |

Selected XGBoost hyperparameters: `n_estimators=75`, `max_depth=3`, `learning_rate=0.1`.

## Project structure

```
risk-score-prediction/
├── data/                  # optional local copy of the dataset
├── notebooks/             # exploration and model selection
├── train.py               # cleans data, trains the final model, saves model_xgb.bin
├── predict.py             # Flask app (/ping and /predict)
├── test.py                # sends a sample applicant to the running service
├── applicant.py           # sample applicant used by test.py
├── test_model.py          # checks the pickled model directly
├── model_xgb.bin          # trained pipeline
├── Dockerfile
├── .dockerignore
├── .gitignore
├── Pipfile
├── Pipfile.lock
└── README.md
```

## Run it

### Locally

```bash
pipenv install
pipenv shell

python train.py                              # creates model_xgb.bin
gunicorn --bind 0.0.0.0:9696 predict:app     # start the service
```

In a second terminal:

```bash
python ping.py     # prints PONG
python test.py     # sends a sample applicant
```

### With Docker

```bash
docker build -t risk-score .
docker run -it --rm -p 9696:9696 risk-score
```

Then run `python ping.py` and `python test.py` from another terminal.

## API

**`GET /ping`** returns `PONG`.

**`POST /predict`** takes one applicant as JSON:

```json
{
  "seniority": 3,
  "home": "owner",
  "time": 36,
  "age": 30,
  "marital": "married",
  "records": "no",
  "job": "fixed",
  "expenses": 75,
  "income": 120.0,
  "assets": 3000.0,
  "debt": 0.0,
  "amount": 800,
  "price": 1000
}
```

and returns:

```json
{
  "default_probability": 0.0,
  "default": false
}
```

(The values above are illustrative. Run `python test.py` to see a real response.) A `null` for `income`, `assets` or `debt` is filled in with the training median.

## Notes and limitations

- The decision threshold is `0.5`. In a real lending setting it should be tuned to the relative cost of approving a defaulter versus rejecting a good applicant.
- Categorical values must match the training categories (for example `home`: `rent`, `owner`, `private`, `ignore`, `parents`, `other`, `unk`). Unseen categories are silently ignored by the vectorizer.
- The pickle depends on library versions, so use the versions pinned in `Pipfile.lock`.
- This is a learning project on a small public dataset, not a production credit-scoring system.

## Tech stack

Python 3.11, pandas, scikit-learn, XGBoost, Flask, Gunicorn, Pipenv, Docker.