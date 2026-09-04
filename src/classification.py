import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class InformationClassifier:
    """
    Classifies relevant conversation sentences into
    Problem, Idea, Action, Decision, or Information.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )

        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        self.is_trained = False

    def train(self, dataframe):
        """
        Train the classifier using the conversation dataset.
        """

        # Only use relevant examples
        training_data = dataframe[
            dataframe["relevance"] == 1
        ].copy()

        X = training_data["sentence"]
        y = training_data["category"]

        # Convert text into TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X)

        # Train Logistic Regression
        self.model.fit(X_tfidf, y)

        self.is_trained = True

    def predict(self, sentence):
        """
        Predict the category of a sentence.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Classifier has not been trained yet."
            )

        X = self.vectorizer.transform([sentence])

        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        confidence = probabilities.max()

        return prediction, float(confidence)