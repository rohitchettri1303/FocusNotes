from src.preprocessing import split_sentences
from src.relevance import RelevanceDetector


agenda = "Website Performance Optimization"

conversation = """
The homepage is taking almost five seconds to load.

Maybe we should compress the images.

I'll investigate whether WebP would help.

Did you guys watch the football match yesterday?
"""


# Split conversation into sentences
sentences = split_sentences(conversation)

# Create relevance detector
detector = RelevanceDetector()


print("\nRelevance Analysis:\n")

for sentence in sentences:

    relevant, score = detector.is_relevant(
        agenda,
        sentence
    )

    status = "RELEVANT" if relevant else "IRRELEVANT"

    print(
        f"{status} | "
        f"Score: {score:.3f} | "
        f"{sentence}"
    )