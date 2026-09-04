import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Load datasets
# ============================================================

train_df = pd.read_csv(
    "data/training_dataset.csv"
)

test_df = pd.read_csv(
    "data/test_dataset.csv"
)


# ============================================================
# 2. Use only relevant sentences
# ============================================================

train_df = train_df[
    train_df["relevance"] == 1
].copy()

test_df = test_df[
    test_df["relevance"] == 1
].copy()


X_train = train_df["sentence"]
y_train = train_df["category"]

X_test = test_df["sentence"]
y_test = test_df["category"]


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
print("FOCUSNOTES - TF-IDF + SVM")
print("==========================================")

print(f"\nTraining examples: {len(X_train)}")
print(f"Testing examples:  {len(X_test)}")

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 5. Predict
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 6. Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n------------------------------------------")
print("PERFORMANCE")
print("------------------------------------------")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")


# ============================================================
# 7. Classification report
# ============================================================

print("\n------------------------------------------")
print("CLASSIFICATION REPORT")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 8. Confusion matrix
# ============================================================

labels = sorted(
    y_test.unique()
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\n------------------------------------------")
print("CONFUSION MATRIX")
print("------------------------------------------")

print("Labels:")
print(labels)

print("\nMatrix:")
print(matrix)