import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model lazily or globally
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def extract_named_entities(sentences):
    """
    Extracts PERSON, DATE, ORG (Project), GPE (Location) per utterance.
    """
    if nlp is None:
        try:
            spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            return [{"entities": [], "person": [], "date": [], "org": [], "gpe": []} for _ in sentences]
    else:
        spacy_nlp = nlp

    results = []
    for sentence in sentences:
        doc = spacy_nlp(sentence)
        entities = []
        persons, dates, orgs, gpes = [], [], [], []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.append(ent.text)
                entities.append((ent.text, "PERSON"))
            elif ent.label_ == "DATE":
                dates.append(ent.text)
                entities.append((ent.text, "DATE"))
            elif ent.label_ in ["ORG", "PROJECT"]:
                orgs.append(ent.text)
                entities.append((ent.text, "ORG"))
            elif ent.label_ in ["GPE", "LOC"]:
                gpes.append(ent.text)
                entities.append((ent.text, "GPE"))

        results.append({
            "sentence": sentence,
            "entities": entities,
            "person": persons,
            "date": dates,
            "org": orgs,
            "gpe": gpes
        })
    return results


def detect_repetition(sentences, model=None, similarity_threshold=0.85):
    """
    Detects repeated/near-duplicate utterances using sentence embeddings.
    """
    if not sentences:
        return []
    
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)
    
    results = []
    for i, sent in enumerate(sentences):
        duplicates = []
        for j in range(len(sentences)):
            if i != j and sim_matrix[i][j] >= similarity_threshold:
                duplicates.append({"index": j, "sentence": sentences[j], "score": float(sim_matrix[i][j])})
        
        results.append({
            "sentence_index": i,
            "sentence": sent,
            "is_repeated": len(duplicates) > 0,
            "duplicate_matches": duplicates
        })
    return results


def detect_qa_pairs(utterances):
    """
    Detects Q&A pairs using question marks and speaker turn adjacency.
    Accepts list of sentence strings or list of dicts with 'speaker' and 'text'.
    """
    qa_pairs = []
    
    for i in range(len(utterances) - 1):
        curr = utterances[i]
        next_utt = utterances[i + 1]

        curr_text = curr["text"] if isinstance(curr, dict) else curr
        curr_speaker = curr.get("speaker", f"Speaker_{i}") if isinstance(curr, dict) else None

        next_text = next_utt["text"] if isinstance(next_utt, dict) else next_utt
        next_speaker = next_utt.get("speaker", f"Speaker_{i+1}") if isinstance(next_utt, dict) else None

        is_question = "?" in curr_text or bool(re.search(r"^(who|what|where|when|why|how|can|could|would|should|is|are|do|does|did)\b", curr_text.strip(), re.I))

        if is_question:
            # Different speaker or adjacent turn answer
            if curr_speaker is None or curr_speaker != next_speaker:
                qa_pairs.append({
                    "question_index": i,
                    "question_speaker": curr_speaker,
                    "question": curr_text,
                    "answer_index": i + 1,
                    "answer_speaker": next_speaker,
                    "answer": next_text
                })
    return qa_pairs


def extract_speaker_turns_and_interruptions(utterances):
    """
    Computes turn count, turn length, and overlap/interruption flags if speaker labels present.
    Format expected: list of dicts with keys: 'speaker', 'text', optional 'start_time', 'end_time'.
    """
    has_labels = False
    if utterances and isinstance(utterances[0], dict) and "speaker" in utterances[0]:
        has_labels = True

    if not has_labels:
        return {
            "has_speaker_labels": False,
            "note": "TODO: Standardize transcript format to list of dicts with 'speaker' and 'text' keys to enable interruption detection.",
            "total_turns": len(utterances),
            "speaker_stats": {},
            "turns": []
        }

    turns = []
    speaker_stats = {}
    prev_speaker = None
    prev_end_time = None

    for idx, utt in enumerate(utterances):
        speaker = utt.get("speaker", "Unknown")
        text = utt.get("text", "")
        start_time = utt.get("start_time", None)
        end_time = utt.get("end_time", None)
        word_count = len(text.split())

        is_interruption = False
        if prev_speaker and prev_speaker != speaker:
            if start_time is not None and prev_end_time is not None:
                if start_time < prev_end_time:
                    is_interruption = True

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"turn_count": 0, "total_words": 0, "interruptions": 0}
        
        speaker_stats[speaker]["turn_count"] += 1
        speaker_stats[speaker]["total_words"] += word_count
        if is_interruption:
            speaker_stats[speaker]["interruptions"] += 1

        turns.append({
            "index": idx,
            "speaker": speaker,
            "text": text,
            "word_count": word_count,
            "is_interruption": is_interruption
        })
        
        prev_speaker = speaker
        if end_time is not None:
            prev_end_time = end_time

    return {
        "has_speaker_labels": True,
        "total_turns": len(turns),
        "speaker_stats": speaker_stats,
        "turns": turns
    }


def transcribe_audio_whisper(audio_path, model_name="openai/whisper-tiny"):
    """
    Transcribes an audio file using HuggingFace Whisper ASR pipeline with timestamp chunks.
    """
    try:
        from transformers import pipeline
        asr_pipe = pipeline("automatic-speech-recognition", model=model_name)
        res = asr_pipe(audio_path, return_timestamps=True)
        chunks = res.get("chunks", [])
        transcript = res.get("text", "")
        return {
            "transcript": transcript,
            "chunks": chunks,
            "success": True
        }
    except Exception as e:
        return {
            "transcript": "",
            "chunks": [],
            "error": str(e),
            "success": False
        }


def extract_prosodic_features(audio_path, utterance_timestamps=None, num_sentences=1):
    """
    Extracts pitch variance and pause duration per utterance using librosa.
    If utterance_timestamps is None, divides audio into equal segments based on num_sentences.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        segments = []
        if utterance_timestamps and len(utterance_timestamps) == num_sentences:
            for item in utterance_timestamps:
                ts = item.get("timestamp", (0, duration))
                start = ts[0] if ts[0] is not None else 0.0
                end = ts[1] if ts[1] is not None else duration
                segments.append((start, end))
        else:
            # Segment equally
            seg_len = duration / max(1, num_sentences)
            for i in range(num_sentences):
                segments.append((i * seg_len, (i + 1) * seg_len))

        results = []
        prev_end = 0.0
        for start, end in segments:
            # Pause duration before this utterance
            pause_before = max(0.0, start - prev_end)
            
            # Slice audio segment
            s_sample = int(start * sr)
            e_sample = int(min(duration, end) * sr)
            y_seg = y[s_sample:e_sample]

            # Pitch variance via pyin
            if len(y_seg) > sr * 0.1:  # at least 100ms
                try:
                    f0, _, _ = librosa.pyin(y_seg, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
                    valid_f0 = f0[~np.isnan(f0)]
                    pitch_var = float(np.var(valid_f0)) if len(valid_f0) > 1 else 0.0
                except Exception:
                    pitch_var = 0.0
            else:
                pitch_var = 0.0

            results.append({
                "start_time": start,
                "end_time": end,
                "pitch_variance": pitch_var,
                "pause_duration": pause_before
            })
            prev_end = end

        return {
            "has_audio": True,
            "duration": duration,
            "prosodic_features": results
        }
    except Exception as e:
        return {
            "has_audio": False,
            "error": str(e),
            "prosodic_features": [{"pitch_variance": 0.0, "pause_duration": 0.0} for _ in range(num_sentences)]
        }
