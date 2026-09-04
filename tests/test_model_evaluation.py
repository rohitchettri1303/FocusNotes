import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv(
    "data/conversation_dataset.csv"
)


# Only relevant sentences are used for
# information-type classification.
data = data[data["relevance"] == 1].copy()


X = data["sentence"]
y = data["category"]


# --------------------------------------------------
# 2. Create NLP pipeline
# --------------------------------------------------

pipeline = Pipeline([
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
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# --------------------------------------------------
# 3. Cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)


predictions = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv
)


# --------------------------------------------------
# 4. Evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n========================================")
print("TF-IDF + LOGISTIC REGRESSION")
print("========================================\n")

print(f"Accuracy :  {accuracy:.3f}")
print(f"Precision:  {precision:.3f}")
print(f"Recall   :  {recall:.3f}")
print(f"F1 Score :  {f1:.3f}")


# --------------------------------------------------
# 5. Detailed classification report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)


# --------------------------------------------------
# 6. Confusion matrix
# --------------------------------------------------

labels = sorted(y.unique())

matrix = confusion_matrix(
    y,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print("Labels:", labels)
print(matrix)