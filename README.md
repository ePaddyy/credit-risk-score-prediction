# CreditLens — Credit Risk Prediction

An end-to-end machine learning project that estimates the probability of credit default using XGBoost. CreditLens combines data preprocessing, model training and evaluation, decision-threshold selection, model explainability, and a Flask REST API.

## Overview

CreditLens is designed to demonstrate how a trained machine learning model can be integrated into an API for credit-risk prediction.

The project covers the following stages:

- Data cleaning and preprocessing
- XGBoost model training
- Model evaluation using classification metrics
- Decision-threshold selection using validation data
- Model persistence for reuse during inference
- SHAP-based model explainability
- REST API development with Flask
- Automated API testing with pytest
- Containerization with Docker

**Important:** This project is for educational and demonstration purposes. Its predictions should not be used as the sole basis for real lending decisions.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- Flask
- pytest
- Docker

## Project Structure

```text
credit-risk-score-prediction/
├── train.py           # Data preparation, training and evaluation
├── predict.py         # Flask REST API
├── explain.py         # SHAP model explainability, if included
├── test_api.py        # Automated API tests
├── applicant.py       # Applicant-related functionality, if included
├── test.py            # Additional tests or scripts, if included
├── model_xgb.bin      # Saved model artifact
├── Dockerfile         # Docker image configuration
├── requirements.txt   # Python dependencies
├── .gitignore         # Files excluded from Git
├── README.md          # Project documentation
└── crsp.ipynb         # Exploratory analysis notebook
```

The structure above describes the main files; filenames should be adjusted to match the final repository.

## Getting Started

### Prerequisites

- Python and pip
- Git
- Docker, if running the containerized application

### 1. Clone the repository

```bash
git clone https://github.com/ePaddyy/credit-risk-score-prediction.git
cd credit-risk-score-prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

The training script prepares the dataset, trains the model, evaluates its performance, selects a decision threshold using validation data, and saves the model artifact.

The saved artifact must be available at the location expected by the API.

## Running the API

Start the Flask application:

```bash
python predict.py
```

The API listens on port `9696`.

### Health check

**Endpoint:** `GET /ping`

```bash
curl http://localhost:9696/ping
```

Expected response:

```json
{
  "status": "ok"
}
```

### Make a prediction

**Endpoint:** `POST /predict`

**Content-Type:** `application/json`

Example request:

```json
{
  "seniority": 3,
  "home": "owner",
  "time": 36,
  "age": 69,
  "marital": "married",
  "records": "no",
  "job": "fixed",
  "expenses": 75,
  "income": 120.0,
  "assets": 3000.0,
  "debt": 0.0,
  "amount": 1000000,
  "price": 1000
}
```

Example response structure:

```json
{
  "default_probability": 0.4,
  "threshold": 0.34,
  "default": true
}
```

The probability above is illustrative, not a measured prediction for the example applicant.

The API returns:

- `default_probability`: the model's estimated probability of default.
- `threshold`: the decision cutoff selected during model development.
- `default`: whether the probability meets or exceeds that cutoff.

The example threshold is `0.34`. A probability greater than or equal to the threshold produces `true`; otherwise, it produces `false`.

The selected threshold is an operating choice based on validation performance, not a universal credit-risk standard.

## Model Evaluation

The training script evaluates the model using metrics such as:

- ROC-AUC
- Precision
- Recall
- F1-score
- Average precision (PR-AUC)
- Confusion matrix

The decision threshold is selected using validation data. The held-out test set is then used to assess the model's performance at that threshold.

Add the final test-set metrics from your latest training run here before publishing the project.

## Model Explainability

SHAP (SHapley Additive exPlanations) can be used to examine how input features influence the model's predictions.

The explainability script can generate global feature-importance and summary plots, depending on the implementation in the repository.

These explanations describe model behavior; they do not establish that a feature causes credit default.

## Automated Testing

Run the API tests with:

```bash
pytest -v
```

The tests cover API health checks, valid predictions, missing fields, invalid input, and handling of null numerical values.

## Docker

Build the Docker image from the directory containing the Dockerfile:

```bash
docker build -t risk-score:latest .
```

Run the container:

```bash
docker run -d --name creditlens-container -p 9696:9696 risk-score:latest
```

Check the container:

```bash
docker ps
docker logs creditlens-container
```

Test the health endpoint:

```bash
curl http://localhost:9696/ping
```

The Docker image must include the application dependencies and make the trained model artifact available at the path expected by the API.

To update the application after changing the model or source code, rebuild the image and recreate the container.

## Limitations and Responsible Use

- Model performance depends on the training data and evaluation methodology.
- A probability estimate is not a guarantee of a future outcome.
- Threshold selection involves trade-offs between false positives and false negatives.
- The model should be assessed for fairness, bias, calibration, and suitability before any real-world credit use.
- Applicant data should be handled securely and in accordance with applicable privacy requirements.

## Author

**Emmanuel Paddy Adams**

GitHub: [ePaddyy](https://github.com/ePaddyy)

## Acknowledgements

The project uses open-source Python machine learning libraries and a publicly available credit-scoring dataset for educational model development.
