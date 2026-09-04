import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.extraction import extract_named_entities, detect_repetition, detect_qa_pairs, extract_speaker_turns_and_interruptions

# Load exact data split
train_df = pd.read_csv("data/training_dataset.csv")
test_df = pd.read_csv("data/test_dataset.csv")

train_sentences = train_df["sentence"].tolist()
train_agendas = train_df["agenda"].tolist()
train_text = [f"{a} [SEP] {s}" for a, s in zip(train_agendas, train_sentences)]

test_sentences = test_df["sentence"].tolist()
test_agendas = test_df["agenda"].tolist()
test_text = [f"{a} [SEP] {s}" for a, s in zip(test_agendas, test_sentences)]

# Check speaker turns
tr_turn_res = extract_speaker_turns_and_interruptions(train_sentences)
te_turn_res = extract_speaker_turns_and_interruptions(test_sentences)

tr_has_speaker = tr_turn_res.get("has_speaker_labels", False)
te_has_speaker = te_turn_res.get("has_speaker_labels", False)

# Gate speaker turns (omit/0.0 when has_speaker_labels is False)
train_turns = np.array([len(s.split()) if tr_has_speaker else 0.0 for s in train_sentences]).reshape(-1, 1)
test_turns = np.array([len(s.split()) if te_has_speaker else 0.0 for s in test_sentences]).reshape(-1, 1)

train_ner = np.array([len(r["entities"]) for r in extract_named_entities(train_sentences)]).reshape(-1, 1)
train_rep = np.array([1 if r["is_repeated"] else 0 for r in detect_repetition(train_sentences)]).reshape(-1, 1)
qa_tr = detect_qa_pairs(train_sentences)
q_tr_indices = {p["question_index"] for p in qa_tr} | {p["answer_index"] for p in qa_tr}
train_qa = np.array([1 if i in q_tr_indices else 0 for i in range(len(train_df))]).reshape(-1, 1)

test_ner = np.array([len(r["entities"]) for r in extract_named_entities(test_sentences)]).reshape(-1, 1)
test_rep = np.array([1 if r["is_repeated"] else 0 for r in detect_repetition(test_sentences)]).reshape(-1, 1)
qa_te = detect_qa_pairs(test_sentences)
q_te_indices = {p["question_index"] for p in qa_te} | {p["answer_index"] for p in qa_te}
test_qa = np.array([1 if i in q_te_indices else 0 for i in range(len(test_df))]).reshape(-1, 1)

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
X_tr_tfidf = vectorizer.fit_transform(train_text).toarray()
X_te_tfidf = vectorizer.transform(test_text).toarray()


def run_fixed_baseline_ablation():
    # Fixed Gated Full Feature Set
    X_tr = np.hstack([X_tr_tfidf, train_ner, train_rep, train_qa, train_turns])
    X_te = np.hstack([X_te_tfidf, test_ner, test_rep, test_qa, test_turns])
    
    rel_clf = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    rel_clf.fit(X_tr, train_df["relevance"])
    rel_preds = rel_clf.predict(X_te)
    
    rel_mask = train_df["relevance"] == 1
    info_clf = LogisticRegression(max_iter=1000, random_state=42)
    info_clf.fit(X_tr[rel_mask], train_df.loc[rel_mask, "category"])
    
    preds = []
    for idx, r in enumerate(rel_preds):
        if r == 0:
            preds.append("IRRELEVANT")
        else:
            preds.append(info_clf.predict([X_te[idx]])[0])
            
    actuals = test_df["category"].tolist()
    acc = accuracy_score(actuals, preds)
    prec = precision_score(actuals, preds, average="weighted", zero_division=0)
    rec = recall_score(actuals, preds, average="weighted", zero_division=0)
    f1 = f1_score(actuals, preds, average="weighted", zero_division=0)
    
    print("\n=================== FIXED FULL FEATURE SET (BASELINE) ===================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    return acc, prec, rec, f1


if __name__ == "__main__":
    run_fixed_baseline_ablation()
