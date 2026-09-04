from src.pipeline import FocusNotesPipeline


# ============================================================
# Test conversation
# ============================================================

agenda = "Website Performance Optimization"


conversation = """
Yesterday we discussed the website.
The homepage takes almost five seconds to load.
Maybe we should compress the images.
I'll investigate whether WebP would help.
Did you guys watch the football match yesterday?
"""


# ============================================================
# Create pipeline
# ============================================================

pipeline = FocusNotesPipeline()


# ============================================================
# Analyze conversation
# ============================================================

output = pipeline.analyze(
    agenda,
    conversation
)

results = output["results"]
sticky_notes = output["sticky_notes"]


# ============================================================
# Display results
# ============================================================

print("\n")
print("=" * 60)
print("FOCUSNOTES NLP PIPELINE")
print("=" * 60)


for result in results:

    print("\nSentence:")
    print(result["sentence"])

    print(
        "Relevance:",
        result["relevance"]
    )

    print(
        "Category:",
        result["category"]
    )

    print("-" * 60)

print("\n")
print("=" * 60)
print("GENERATED STICKY NOTES")
print("=" * 60)

for category, sentences in sticky_notes.items():

    print(f"\n📌 {category}")

    for sentence in sentences:
        print(f"   • {sentence}")

        