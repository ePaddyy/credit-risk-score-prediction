import pickle

import numpy as np
from flask import Flask, jsonify, request

model_file = "model_xgb.bin"

# The pickle holds the whole pipeline (DictVectorizer -> SimpleImputer -> XGBClassifier)
with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("risk-score")


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


@app.route("/predict", methods=["POST"])
def predict():
    applicant = request.get_json()

    # JSON null -> NaN, so the imputer can fill it in
    # applicant = {k: (np.nan if v is None else v) for k, v in applicant.items()}

    y_pred = model.predict_proba([applicant])[0, 1]
    default = y_pred >= 0.5

    result = {
        "default_probability": float(y_pred),
        "default": bool(default),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)