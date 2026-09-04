import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# ============================================================
# Load training dataset
# ============================================================

data = pd.read_csv(
    "data/training_dataset.csv"
)


# We only train the information classifier
# on relevant sentences.

data = data[
    data["relevance"] == 1
].copy()


X = data["sentence"]
y = data["category"]


# ============================================================
# Create final model
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
    ),

    (
        "classifier",
        LinearSVC(
            C=1.0,
            random_state=42
        )
    )
])


# ============================================================
# Train on complete training dataset
# ============================================================

print("\nTraining FocusNotes information classifier...")

model.fit(
    X,
    y
)

print("Training completed.")


# ============================================================
# Save model
# ============================================================

joblib.dump(
    model,
    "models/information_classifier.pkl"
)

print(
    "\nModel saved to:"
    "\nmodels/information_classifier.pkl"
)