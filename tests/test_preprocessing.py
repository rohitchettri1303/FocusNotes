from src.preprocessing import split_sentences


conversation = """
Yesterday we discussed the website.

The homepage is taking almost five seconds to load.

Maybe we should compress the images to improve the loading time.

I'll investigate whether WebP would help.

Also, did you guys watch the football match yesterday?
"""


sentences = split_sentences(conversation)


print("\nDetected Sentences:\n")

for i, sentence in enumerate(sentences, start=1):
    print(f"{i}. {sentence}")