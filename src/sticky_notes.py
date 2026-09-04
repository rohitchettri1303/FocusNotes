from collections import defaultdict


class StickyNoteGenerator:

    def generate(self, results):

        notes = defaultdict(list)

        for result in results:

            if result["relevance"] != "RELEVANT":
                continue

            category = result["category"]

            if category is None:
                continue

            notes[category].append(
                result["sentence"]
            )

        return dict(notes)