<div align="center">

# 🗒️ FocusNotes

**Turn casual meeting conversations into concise, actionable sticky notes — automatically.**

*A context-aware conversation & audio intelligence platform that filters signal from noise, classifies what matters, and generates clean summaries — from text or audio.*

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-BART%20%7C%20Whisper-yellow)](https://huggingface.co/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

</div>

---

## ✨ What it does

Meetings generate a lot of talk and very little that's actually worth remembering. **FocusNotes** listens (or reads) so you don't have to re-listen — it takes a raw meeting transcript or audio recording and turns it into a clean set of categorized sticky notes:

- 🎯 **Filters** agenda-irrelevant chatter from what actually matters
- 🏷️ **Classifies** relevant content into `PROBLEM`, `IDEA`, `ACTION`, `DECISION`, or `INFORMATION`
- 📝 **Generates** short, abstractive sticky notes (≤ 20 words) via BART
- 🎙️ **Accepts audio directly** — Whisper transcribes it, librosa reads the prosody
- 👥 **Understands conversation structure** — who spoke, who interrupted, what was repeated, which questions got answered

---

## 🖼️ How it looks

FocusNotes ships with a restrained, Apple-inspired interface — neutral surfaces, a single accent color, soft-shadowed cards, and color-coded category badges so you can triage a meeting at a glance.

| Element | Style |
|---|---|
| Background | `#F5F5F7` |
| Card surface | `#FFFFFF` |
| Primary text | `#1D1D1F` |
| Accent | `#0071E3` |
| Typography | `-apple-system`, `SF Pro Display`, `SF Pro Text` |

---

## 🧠 How it works

FocusNotes runs a six-stage pipeline, from raw multimodal input to structured sticky notes:

```
 ┌─────────────────┐    ┌───────────────┐    ┌──────────────────┐
 │  1. Audio Input  │───▶│ 2. Cleaning & │───▶│ 3. Relevance      │
 │  & ASR (Whisper) │    │  Tokenizing   │    │  Filtering (SVM)  │
 └─────────────────┘    └───────────────┘    └──────────────────┘
                                                       │
                                                       ▼
 ┌─────────────────┐    ┌───────────────┐    ┌──────────────────┐
 │ 6. Extended      │◀───│ 5. Sticky Note│◀───│ 4. Category       │
 │  Feature Extract │    │  Synthesis    │    │  Classification    │
 │  (NER, Q&A, etc.)│    │  (BART)       │    │  (Logistic Reg.)   │
 └─────────────────┘    └───────────────┘    └──────────────────┘
```

| Stage | What happens | Key module |
|---|---|---|
| 1. Audio & ASR | Transcribes uploaded `.wav` / `.mp3` / `.m4a` via Whisper when no text is supplied | `transcribe_audio_whisper()` in `src/extraction.py` |
| 2. Cleaning & Tokenizing | Normalizes whitespace, splits into sentences | `src/preprocessing.py` |
| 3. Relevance Filtering | Filters out off-topic utterances using TF-IDF + semantic similarity | `src/relevance.py` (`LinearSVC`, `all-MiniLM-L6-v2`) |
| 4. Category Classification | Labels relevant sentences: Problem / Idea / Action / Decision / Information | `src/classification.py` (`LogisticRegression`) |
| 5. Sticky Note Synthesis | Groups by category, generates abstractive notes | `src/summarization.py` (`facebook/bart-large-cnn`) |
| 6. Extended Feature Extraction | NER, repetition, Q&A pairing, speaker turns, prosody | `src/extraction.py` |

**Smart gating:** optional features never introduce noise where the underlying data isn't there.
- 🔇 No audio uploaded → prosodic features are skipped, not faked (`has_audio` flag)
- 🔇 No speaker labels in transcript → interruption analysis falls back gracefully (`has_speaker_labels` flag)

---

## 🧩 Features at a glance

| Feature | Method | Status |
|---|---|---|
| Agenda relevance scoring | TF-IDF + Sentence-Transformer cosine similarity | ✅ Always on (with agenda) |
| Category classification | TF-IDF + Logistic Regression | ✅ Always on |
| Named entity recognition | spaCy `en_core_web_sm` | ✅ Always on |
| Repetition / near-duplicate detection | Sentence-embedding cosine similarity | ✅ Always on |
| Q&A pair detection | Regex + adjacent-turn linking | ✅ Always on |
| Speaker turns & interruptions | Speaker metadata + timestamp overlap | ⚙️ Gated on speaker labels |
| Automatic speech recognition | Whisper (`openai/whisper-tiny`) | ⚙️ Gated on audio upload |
| Prosodic pitch & pause features | `librosa.pyin` + silence detection | ⚙️ Gated on audio upload |
| Abstractive sticky note generation | BART (`facebook/bart-large-cnn`) | ✅ Generative mode |

---

## 📊 Performance

**Classification** (held-out test set, gated baseline)

| Metric | Score |
|---|---|
| Accuracy | 87.8% |
| Precision (weighted) | 89.0% |
| Recall (weighted) | 87.8% |
| F1-score (weighted) | 88.1% |

**Generation quality** (vs. hand-written reference notes)

| Metric | Score |
|---|---|
| ROUGE-1 | 0.55 – 0.58 |
| ROUGE-2 | 0.32 – 0.34 |
| ROUGE-L | 0.49 – 0.52 |
| BLEU | 0.26 – 0.30 |
| METEOR | 0.41 – 0.44 |

**Key finding from ablation testing:** semantic similarity is the strongest single feature; naively-included speaker-turn features initially *hurt* performance due to noisy fallback values on unlabeled transcripts — fixed by conditional gating, recovering full F1 to 0.881. See [`analysis.md`](analysis.md) for the full breakdown.

---

## 🚀 Getting started

```bash
# clone and enter the project
git clone <repo-url>
cd focusnotes

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

Then open the app, paste a transcript (or upload audio), add your meeting agenda, and hit generate.

---

## 🗂️ Project structure

```
focusnotes/
├── app.py                          # Streamlit frontend (Apple-inspired design)
├── analysis.md                     # Technical report: metrics, ablations, roadmap
├── ablation_study.py               # Feature ablation runner (7 configurations)
├── human_eval_batch1.csv           # Human-evaluation scaffold (18 samples)
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── src/
│   ├── pipeline.py                 # FocusNotesPipeline orchestrator
│   ├── extraction.py               # NER, repetition, Q&A, speaker turns, ASR, prosody
│   ├── classification.py           # Information category classifier
│   ├── relevance.py                # Relevance detector (semantic similarity)
│   ├── summarization.py            # BART-based abstractive note generator
│   ├── preprocessing.py            # Text cleaning & sentence splitting
│   ├── sticky_notes.py             # Sticky note grouping logic
│   ├── train_classifier.py         # Trains the category classifier
│   └── train_relevance.py          # Trains the relevance classifier
├── models/
│   ├── information_classifier.pkl
│   └── relevance_classifier.pkl
├── data/
│   ├── training_dataset.csv
│   ├── test_dataset.csv
│   ├── conversation_dataset.csv / _v2.csv
│   └── human_evaluation_scaffold.csv
└── tests/
    ├── test_full_evaluation.py     # End-to-end pipeline evaluation
    ├── test_generation_metrics.py  # ROUGE / BLEU / METEOR
    └── ...                         # Unit tests per module
```

---

## 🔬 Evaluation methodology

FocusNotes is evaluated across three complementary layers:

1. **Classification metrics** — accuracy, precision, recall, F1 on held-out data
2. **Generation metrics** — ROUGE-1/2/L, BLEU, and METEOR against hand-written reference notes
3. **Ablation studies** — every feature group is tested by systematic removal, to separate what actually helps from what just adds complexity
4. **Human evaluation** — a prepared, double-blind scaffold (relevance + usefulness ratings) awaiting annotator input

---

## ⚠️ Known limitations

- **Prosodic features are validated only on synthetic (TTS) audio** — real recorded meeting audio is needed to properly assess pitch/pause signal, since synthetic speech lacks natural intonation variance.
- **Speaker-turn analysis depends on labeled transcripts** — plain-text input without speaker tags falls back to a coarser approximation.
- **Domain-specific jargon** may need additional fine-tuning beyond the general-purpose spaCy model and TF-IDF features.

---

## 🗺️ Roadmap

- [ ] Benchmark prosodic features on real multi-speaker meeting recordings
- [ ] Fine-tune a lightweight summarizer (e.g. Flan-T5-small) on meeting-specific data
- [ ] Real-time streaming transcription support
- [ ] Cross-meeting tracking of recurring action items and decisions
- [ ] Complete and analyze the human-evaluation study

---

## 🛠️ Tech stack

`Python` · `Streamlit` · `scikit-learn` · `Sentence-Transformers` · `HuggingFace Transformers` · `BART` · `Whisper` · `spaCy` · `librosa` · `NLTK` · `pandas`

---

<div align="center">

*Built to make meetings a little less forgettable.*

</div>

