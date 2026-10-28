
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import shap

from train import load_and_clean, to_records


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model_xgb.bin"
OUTPUT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------
print("Loading trained model...")

with MODEL_FILE.open("rb") as f:
    model = pickle.load(f)


# --------------------------------------------------
# 3. Load and prepare data
# --------------------------------------------------
print("Loading dataset...")

df = load_and_clean()

X = df.drop(columns=["status"])

# Sample up to 500 applicants for faster explanation
X_sample = X.sample(
    n=min(500, len(X)),
    random_state=24
)

records = to_records(X_sample)


# --------------------------------------------------
# 4. Transform data using the fitted pipeline steps
# --------------------------------------------------
print("Transforming data...")

vectorizer = model.named_steps["dictvectorizer"]
imputer = model.named_steps["simpleimputer"]
classifier = model.named_steps["xgbclassifier"]

X_vectorized = vectorizer.transform(records)
X_transformed = imputer.transform(X_vectorized)

feature_names = vectorizer.get_feature_names_out()


# --------------------------------------------------
# 5. Calculate SHAP values
# --------------------------------------------------
print("Calculating SHAP values...")

explainer = shap.TreeExplainer(classifier)

shap_values = explainer(X_transformed)

# Add the original feature names to the explanation
shap_values.feature_names = list(feature_names)


# --------------------------------------------------
# 6. Global SHAP feature importance
# --------------------------------------------------
print("Generating global feature importance plot...")

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "creditlens_shap_global_importance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 7. SHAP summary plot
# --------------------------------------------------
print("Generating SHAP summary plot...")

shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "creditlens_shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 8. Finish
# --------------------------------------------------
print("\nSHAP analysis completed successfully!")

print(
    "Global importance:",
    OUTPUT_DIR / "creditlens_shap_global_importance.png"
)

print(
    "SHAP summary:",
    OUTPUT_DIR / "creditlens_shap_summary.png"
)
