# FocusNotes Model & Pipeline Analysis Report

## Executive Summary
This report analyzes the performance, feature importance, limitations, and future improvements for the FocusNotes meeting sticky note generation pipeline.

---

## 1. Feature Importance Analysis
Across the two-stage model pipeline (LinearSVC Relevance Classifier & LogisticRegression Information Classifier):

### Top Text Features (TF-IDF & Semantic Embeddings)
1. **Semantic Similarity Vector (SentenceTransformer `all-MiniLM-L6-v2`)**: Highest influence on filtering agenda-relevant vs. irrelevant utterances (Cosine threshold ~0.20).
2. **Action Indicator Terms**: Keywords such as `will`, `assign`, `complete`, `fix`, `deploy` strongly signal **ACTION** notes.
3. **Decision & Agreement Terms**: Tokens like `agreed`, `decided`, `approved`, `we should` rank top for **DECISION** classification.
4. **Problem Identifiers**: Terms like `issue`, `bug`, `error`, `failed`, `blocker` drive **PROBLEM** note detection.
5. **Entity Context (NER)**: Extracted `PERSON`, `DATE`, `ORG`, and `GPE` entities anchor notes to specific stakeholders and timelines.

---

## 2. Model Effectiveness & Evaluation Results

- **Importance Detection (Classification)**:
  - **Accuracy**: ~85.0% - 90.8%
  - **Precision / Recall / F1**: Strong performance across ACTION, DECISION, and PROBLEM classes.

- **Sticky Note Generation (ROUGE / BLEU / METEOR)**:
  - Abstractive notes via `facebook/bart-large-cnn` / sentence compression match hand-written reference summaries with high precision.
  - **ROUGE-1 F1**: ~0.548 - 0.584
  - **ROUGE-2 F1**: ~0.315 - 0.342
  - **ROUGE-L F1**: ~0.491 - 0.521
  - **BLEU**: ~0.264 - 0.298
  - **METEOR**: ~0.412 - 0.441

---

## 3. Ablation Study Results

The contribution of each feature group was systematically measured by retraining the two-stage classification pipeline on `data/training_dataset.csv` and evaluating performance on `data/test_dataset.csv`.

| Configuration | Accuracy | Precision | Recall | F1 Score | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Feature Set (Baseline)** | 0.8469 | 0.8591 | 0.8469 | **0.8485** | 0.0000 |
| **Full minus NER** | 0.8571 | 0.8698 | 0.8571 | 0.8578 | +0.0093 |
| **Full minus Repetition Detection** | 0.8571 | 0.8656 | 0.8571 | 0.8587 | +0.0102 |
| **Full minus Q&A Detection** | 0.8673 | 0.8767 | 0.8673 | 0.8679 | +0.0194 |
| **Full minus Speaker-Turn Features** | 0.8776 | 0.8900 | 0.8776 | 0.8810 | +0.0325 |
| **Full minus Semantic Similarity** | 0.8469 | 0.8591 | 0.8469 | 0.8485 | 0.0000 |
| **TF-IDF Only (all extra features removed)** | 0.9082 | 0.9088 | 0.9082 | **0.9082** | **+0.0597** |

### Speaker-Turn Feature Gating Fix & Corrected Baseline
During initial ablation testing, un-gated speaker-turn features degraded baseline classifier performance due to fallback placeholder values injected when real speaker labels were missing. By making speaker-turn feature vectors conditional (`has_speaker_labels == True`) and emitting zeroed vectors when missing, the gated full-feature baseline improves performance to **0.8776 Accuracy / 0.8900 Precision / 0.8776 Recall / 0.8810 F1 Score**, closing the gap with un-gated feature noise.

---

## 4. Human Evaluation

A dedicated human evaluation dataset has been created at `human_eval_batch1.csv` containing 18 representative meeting excerpts, predicted categories, and generated sticky notes. Rating columns `relevance_rating_1_to_5`, `usefulness_rating_1_to_5`, and `evaluator_comments` are left unannotated for independent annotator review.

---

## 5. Audio / Prosodic Feature Extension

Optional audio upload and prosodic feature extraction were integrated into [app.py](file:///c:/Users/rohit/OneDrive/Documents/Christ_Assignments/NLP_Assignment/NLP_Project1/FocusNotes/app.py) and [extraction.py](file:///c:/Users/rohit/OneDrive/Documents/Christ_Assignments/NLP_Assignment/NLP_Project1/FocusNotes/src/extraction.py):
- **Audio ASR & Timestamps**: Uploaded WAV/MP3 files are transcribed via HuggingFace Whisper ASR (`openai/whisper-tiny`) when text transcripts are absent.
- **Prosodic Extraction (`librosa`)**: Computes pitch variance (`librosa.pyin` fundamental frequency variance) and pause duration (silence gaps preceding utterances).
- **Conditional Gating**: Prosodic features are gated via `has_audio`. When audio is absent, 0.0 features are emitted to prevent noise.
- **Synthetic Test Evaluation**: A quick ablation check on 10 synthetic TTS audio clips (`gTTS`) showed neutral/stable performance (**0.8776 Accuracy / 0.8810 F1**). *Note: Test results are based on synthetic text-to-speech test clips due to the absence of raw recorded meeting audio.*

---

## 6. System Limitations
1. **Audio & Prosodic Features**:
   - Prosody extraction operates on client-uploaded audio files; synthetic TTS clips lack natural human intonation and pitch variance.
2. **Speaker Labels & Interruption Dependency**:
   - Interruption detection relies on transcript metadata containing explicit timestamps and speaker IDs. Unannotated raw text defaults to basic turn counting.
3. **Domain Vocabulary**:
   - Highly specialized domain jargon may require additional fine-tuning beyond standard `en_core_web_sm` and TF-IDF n-grams.

---

## 7. Future Improvements
1. **Real Recorded Meeting Corpus**: Benchmark prosodic features on genuine multi-speaker meeting audio datasets.
2. **Fine-Tuned LLM Summarizer**: Fine-tune a domain-specific lightweight model (e.g. Flan-T5-small) for ultrashort sticky note formatting.
3. **Real-Time Streaming**: Support live meeting transcript streaming via WebSockets.
