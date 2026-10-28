import pickle

import numpy as np
from flask import Flask, jsonify, request

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model_xgb.bin"

with MODEL_FILE.open("rb") as f_in:
    artifact = pickle.load(f_in)

model = artifact["model"]
THRESHOLD = artifact["threshold"]

app = Flask("risk-score")

REQUIRED_FIELDS = {
    "seniority",
    "home",
    "time",
    "age",
    "marital",
    "records",
    "job",
    "expenses",
    "income",
    "assets",
    "debt",
    "amount",
    "price",
}

CATEGORICAL_FIELDS = {
    "home": {"rent", "owner", "private", "ignore", "parents", "other", "unk"},
    "marital": {"single", "married", "widow", "separated", "divorced", "unk"},
    "records": {"no", "yes", "unk"},
    "job": {"fixed", "partime", "freelance", "others", "unk"},
}

NUMERIC_FIELDS = REQUIRED_FIELDS - CATEGORICAL_FIELDS.keys()


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    applicant = request.get_json(silent=True)

    if not isinstance(applicant, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    missing = REQUIRED_FIELDS - applicant.keys()

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "fields": sorted(missing),
        }), 400

    errors = []

    for field, allowed_values in CATEGORICAL_FIELDS.items():
        value = applicant[field]

        if not isinstance(value, str) or value not in allowed_values:
            errors.append(
                f"{field} must be one of {sorted(allowed_values)}"
            )

    for field in NUMERIC_FIELDS:
        value = applicant[field]

        # Null is allowed to represent a missing numeric value.
        if value is None:
            applicant[field] = np.nan
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{field} must be a number or null")
            continue

        if not np.isfinite(value):
            errors.append(f"{field} must be finite or null")

    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    probability = float(model.predict_proba([applicant])[0, 1])

    return jsonify({
        "default_probability": probability,
        "threshold": THRESHOLD,
        "default": bool(probability >= THRESHOLD),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False)
