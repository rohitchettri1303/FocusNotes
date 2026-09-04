import joblib


# Load trained model
model = joblib.load(
    "models/information_classifier.pkl"
)


test_sentences = [
    "The website is taking too long to load.",
    "Maybe we should compress the images.",
    "I'll investigate the server performance.",
    "We decided to use WebP images.",
    "The website currently has large PNG files.",
]


print("\n======================================")
print("FOCUSNOTES SAVED MODEL TEST")
print("======================================\n")


for sentence in test_sentences:

    prediction = model.predict(
        [sentence]
    )[0]

    print(
        f"{prediction:12} | {sentence}"
    )