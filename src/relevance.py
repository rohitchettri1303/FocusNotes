from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RelevanceDetector:
    """
    Determines how relevant a conversation sentence is
    to the given meeting agenda.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, agenda, sentence):
        """
        Calculate semantic similarity between the agenda
        and a conversation sentence.
        """

        embeddings = self.model.encode(
            [agenda, sentence]
        )

        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(similarity)

    def is_relevant(self, agenda, sentence, threshold=0.20):
        """
        Determine whether a sentence is relevant to the agenda.
        """

        score = self.calculate_similarity(
            agenda,
            sentence
        )

        return score >= threshold, score
    