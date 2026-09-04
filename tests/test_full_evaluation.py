import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Load test dataset
# ============================================================

data = pd.read_csv(
    "data/test_dataset.csv"
)

print("\n==============================================")
print("FOCUSNOTES - FULL PIPELINE EVALUATION")
print("==============================================")

print(
    f"\nTest examples: {len(data)}"
)


# ============================================================
# 2. Load saved models
# ============================================================

print("\nLoading models...")

relevance_model = joblib.load(
    "models/relevance_classifier.pkl"
)

information_model = joblib.load(
    "models/information_classifier.pkl"
)

print("Models loaded successfully.")


# ============================================================
# 3. Prepare relevance classifier input
# ============================================================

data["text"] = (
    data["agenda"]
    + " [SEP] "
    + data["sentence"]
)


# ============================================================
# 4. Stage 1 - Relevance prediction
# ============================================================

print("\nRunning relevance classification...")

relevance_predictions = relevance_model.predict(
    data["text"]
)


# ============================================================
# 5. Stage 2 - Information classification
# ============================================================

print("Running information classification...")

final_predictions = []


for index, row in data.iterrows():

    relevance_prediction = relevance_predictions[index]

    # --------------------------------------------------------
    # Irrelevant sentence
    # --------------------------------------------------------

    if relevance_prediction == 0:

        final_predictions.append(
            "IRRELEVANT"
        )

    # --------------------------------------------------------
    # Relevant sentence
    # --------------------------------------------------------

    else:

        prediction = information_model.predict(
            [row["sentence"]]
        )[0]

        final_predictions.append(
            prediction
        )


# ============================================================
# 6. Actual labels
# ============================================================

actual_labels = data["category"].tolist()


# ============================================================
# 7. Overall evaluation
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    final_predictions
)

precision = precision_score(
    actual_labels,
    final_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    actual_labels,
    final_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    final_predictions,
    average="weighted",
    zero_division=0
)


print("\n==============================================")
print("OVERALL PERFORMANCE")
print("==============================================")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")


# ============================================================
# 8. Classification report
# ============================================================

labels = [
    "ACTION",
    "DECISION",
    "IDEA",
    "INFORMATION",
    "PROBLEM",
    "IRRELEVANT"
]


print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================")

print(
    classification_report(
        actual_labels,
        final_predictions,
        labels=labels,
        zero_division=0
    )
)


# ============================================================
# 9. Confusion matrix
# ============================================================

matrix = confusion_matrix(
    actual_labels,
    final_predictions,
    labels=labels
)


print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")

print("Labels:")
print(labels)

print()

print(matrix)


# ============================================================
# 10. Relevance stage evaluation
# ============================================================

relevance_actual = data["relevance"]

relevance_accuracy = accuracy_score(
    relevance_actual,
    relevance_predictions
)

relevance_precision = precision_score(
    relevance_actual,
    relevance_predictions,
    zero_division=0
)

relevance_recall = recall_score(
    relevance_actual,
    relevance_predictions,
    zero_division=0
)

relevance_f1 = f1_score(
    relevance_actual,
    relevance_predictions,
    zero_division=0
)


print("\n==============================================")
print("RELEVANCE CLASSIFIER PERFORMANCE")
print("==============================================")

print(
    f"Accuracy :  {relevance_accuracy:.4f}"
)

print(
    f"Precision:  {relevance_precision:.4f}"
)

print(
    f"Recall   :  {relevance_recall:.4f}"
)

print(
    f"F1 Score :  {relevance_f1:.4f}"
)
# ============================================================
# 11. Error analysis
# ============================================================

print("\n==============================================")
print("MISCLASSIFIED EXAMPLES")
print("==============================================")

errors = 0

for index, row in data.iterrows():

    actual = row["category"]
    predicted = final_predictions[index]

    if actual != predicted:

        errors += 1

        print(f"\nExample {errors}")
        print("----------------------------------------------")
        print(f"Agenda   : {row['agenda']}")
        print(f"Sentence : {row['sentence']}")
        print(f"Actual   : {actual}")
        print(f"Predicted: {predicted}")
        print(f"Relevance: {row['relevance']}")
        
print("\nTotal misclassified examples:", errors)