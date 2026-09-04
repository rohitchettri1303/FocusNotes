import pandas as pd

from src.classification import InformationClassifier


# Load dataset
data = pd.read_csv(
    "data/conversation_dataset.csv"
)


# Create classifier
classifier = InformationClassifier()


# Train classifier
classifier.train(data)


# Test sentences
test_sentences = [
    "The website is loading very slowly.",
    "Maybe we should optimize the images.",
    "I'll check the image sizes.",
    "We decided to use WebP.",
    "The website currently has large PNG files."
]


print("\nClassification Results:\n")


for sentence in test_sentences:

    category, confidence = classifier.predict(
        sentence
    )

    print(
        f"{category:12} | "
        f"Confidence: {confidence:.3f} | "
        f"{sentence}"
    )