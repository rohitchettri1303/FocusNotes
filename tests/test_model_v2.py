import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
# 1. Load training and testing data
# ============================================================

train_df = pd.read_csv(
    "data/training_dataset.csv"
)

test_df = pd.read_csv(
    "data/test_dataset.csv"
)


# ============================================================
# 2. Separate features and labels
# ============================================================

# We are evaluating INFORMATION TYPE classification.
# Therefore, only relevant examples are used.

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


print("\n==========================================")
print("FOCUSNOTES - MODEL V2 EVALUATION")
print("==========================================")

print(f"\nTraining examples: {len(X_train)}")
print(f"Testing examples:  {len(X_test)}")


# ============================================================
# 3. Create TF-IDF + Logistic Regression pipeline
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
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# ============================================================
# 4. Train
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 5. Predict unseen test data
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 6. Calculate metrics
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


# ============================================================
# 7. Display metrics
# ============================================================

print("\n------------------------------------------")
print("PERFORMANCE")
print("------------------------------------------")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")


# ============================================================
# 8. Classification report
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
# 9. Confusion matrix
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