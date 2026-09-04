import re
import nltk

# Download the tokenizer data if it is not already available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def clean_text(text):
    """
    Basic text cleaning.
    """

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def split_sentences(text):
    """
    Split conversation into individual sentences.
    """

    text = clean_text(text)

    sentences = nltk.sent_tokenize(text)

    # Remove empty sentences
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences

