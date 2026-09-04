import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Load training dataset
# ============================================================

data = pd.read_csv(
    "data/training_dataset.csv"
)


# ============================================================
# 2. Create combined text
# ============================================================

# The relevance decision depends on BOTH:
#
#   Agenda
#   +
#   Sentence
#
# Therefore we combine them into one input.

data["text"] = (
    data["agenda"]
    + " [SEP] "
    + data["sentence"]
)


X = data["text"]

y = data["relevance"]


# ============================================================
# 3. Create TF-IDF + SVM pipeline
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
# 4. Train
# ============================================================

print("\n==========================================")
print("FOCUSNOTES RELEVANCE CLASSIFIER")
print("==========================================")

print(
    f"\nTraining examples: {len(X)}"
)

print("\nTraining model...")

model.fit(
    X,
    y
)

print("Training completed.")


# ============================================================
# 5. Save model
# ============================================================

joblib.dump(
    model,
    "models/relevance_classifier.pkl"
)

print(
    "\nModel saved to:"
    "\nmodels/relevance_classifier.pkl"
)