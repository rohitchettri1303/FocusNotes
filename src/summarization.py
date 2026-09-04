import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


class NoteSummarizer:
    """
    Abstractive sticky note summarizer using HuggingFace Transformers
    (BART-CNN, BART-XSUM, PEGASUS-XSUM) with structural tagging.
    """
    SUPPORTED_MODELS = {
        "bart-cnn": "facebook/bart-large-cnn",
        "bart-xsum": "facebook/bart-large-xsum",
        "pegasus-xsum": "google/pegasus-xsum",
        "distilbart-cnn": "sshleifer/distilbart-cnn-12-6"
    }

    def __init__(self, model_key="bart-cnn"):
        self.model_key = model_key
        self.model_name = self.SUPPORTED_MODELS.get(model_key, model_key)
        self.tokenizer = None
        self.model = None
        self._initialized = False

    def set_model(self, model_key):
        if model_key != self.model_key:
            self.model_key = model_key
            self.model_name = self.SUPPORTED_MODELS.get(model_key, model_key)
            self.tokenizer = None
            self.model = None
            self._initialized = False

    def _load_model(self):
        if not self._initialized:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            except Exception:
                self.tokenizer = None
                self.model = None
            self._initialized = True

    def format_structured_utterances(self, category, utterances, metadata_list=None):
        """
        Formats utterances with structural tags:
        [SPEAKER: {name}] [TYPE: {category}] [ENTITIES: {ents}] {text}
        """
        formatted_items = []
        for idx, utt in enumerate(utterances):
            meta = metadata_list[idx] if metadata_list and idx < len(metadata_list) else {}
            speaker = meta.get("speaker", "SPEAKER")
            
            ents_str = ""
            if nlp is not None:
                doc = nlp(utt)
                ents = [f"{e.text}({e.label_})" for e in doc.ents]
                if ents:
                    ents_str = f" [ENTITIES: {', '.join(ents)}]"

            formatted = f"[SPEAKER: {speaker}] [TYPE: {category}]{ents_str} {utt}"
            formatted_items.append(formatted)
        
        return " ".join(formatted_items)

    def summarize_category_notes(self, category, sentences, metadata_list=None, max_words=20, use_structure=True):
        """
        Summarizes a group of sentences belonging to a category into short bullet points.
        """
        if not sentences:
            return []
        
        if use_structure:
            text_block = self.format_structured_utterances(category, sentences, metadata_list)
        else:
            text_block = " ".join(sentences)
        
        self._load_model()
        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(text_block, return_tensors="pt", max_length=512, truncation=True)
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=35,
                    min_length=6,
                    num_beams=2,
                    early_stopping=True
                )
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                clean_summary = summary
                for tag in ["[SPEAKER:", "[TYPE:", "[ENTITIES:"]:
                    clean_summary = clean_summary.replace(tag, "")

                bullets = [s.strip() for s in clean_summary.split(".") if s.strip()]
                return [b[:120] for b in bullets if len(b.split()) <= max_words or True][:3]
            except Exception:
                pass

        # Fallback abstractive/extractive compression
        notes = []
        for s in sentences:
            words = s.split()
            if len(words) > max_words:
                compressed = " ".join(words[:max_words]) + "..."
            else:
                compressed = s
            notes.append(compressed)
        return notes
