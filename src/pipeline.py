from src.sticky_notes import StickyNoteGenerator
from src.summarization import NoteSummarizer
from src.extraction import (
    extract_named_entities,
    detect_repetition,
    detect_qa_pairs,
    extract_speaker_turns_and_interruptions,
    transcribe_audio_whisper,
    extract_prosodic_features
)
import joblib
import re


class FocusNotesPipeline:

    def __init__(self, mode="extractive"):
        self.mode = mode
        self.note_generator = StickyNoteGenerator()
        self.summarizer = NoteSummarizer()
        
        self.relevance_model = joblib.load(
            "models/relevance_classifier.pkl"
        )

        self.information_model = joblib.load(
            "models/information_classifier.pkl"
        )

    def split_sentences(self, conversation):
        sentences = re.split(
            r'(?<=[.!?])\s+',
            conversation.strip()
        )
        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def analyze(self, agenda, conversation=None, audio_path=None, mode=None):
        active_mode = mode if mode else self.mode
        utterance_timestamps = None
        asr_transcript = ""

        # Step 1: Audio ASR Transcription if text conversation is missing/empty
        if audio_path and (not conversation or not conversation.strip()):
            asr_res = transcribe_audio_whisper(audio_path)
            if asr_res.get("success") and asr_res.get("transcript"):
                conversation = asr_res["transcript"]
                utterance_timestamps = asr_res.get("chunks", [])
                asr_transcript = conversation

        if not conversation:
            conversation = ""

        sentences = self.split_sentences(conversation)
        results = []

        for sentence in sentences:
            relevance_input = agenda + " [SEP] " + sentence
            relevance = self.relevance_model.predict([relevance_input])[0]

            if relevance == 0:
                results.append({
                    "sentence": sentence,
                    "relevance": "IRRELEVANT",
                    "category": None
                })
                continue

            category = self.information_model.predict([sentence])[0]

            results.append({
                "sentence": sentence,
                "relevance": "RELEVANT",
                "category": category
            })

        # Extractive notes
        extractive_notes = self.note_generator.generate(results)

        # Abstractive notes if generative mode requested
        sticky_notes = {}
        if active_mode == "generative":
            for cat, sents in extractive_notes.items():
                sticky_notes[cat] = self.summarizer.summarize_category_notes(cat, sents)
        else:
            sticky_notes = extractive_notes

        # Extended feature extraction
        entities = extract_named_entities(sentences)
        repetitions = detect_repetition(sentences)
        qa_pairs = detect_qa_pairs(sentences)
        turn_stats = extract_speaker_turns_and_interruptions(sentences)

        # Prosodic Feature Extraction (Optional, Gated by has_audio)
        prosodic_res = {"has_audio": False, "prosodic_features": [{"pitch_variance": 0.0, "pause_duration": 0.0} for _ in sentences]}
        if audio_path:
            prosodic_res = extract_prosodic_features(audio_path, utterance_timestamps, num_sentences=len(sentences))

        return {
            "results": results,
            "sticky_notes": sticky_notes,
            "asr_transcript": asr_transcript,
            "features": {
                "entities": entities,
                "repetitions": repetitions,
                "qa_pairs": qa_pairs,
                "turn_stats": turn_stats,
                "prosody": prosodic_res
            }
        }