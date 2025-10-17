import os
import json
import math
from typing import List, Dict, Tuple
from collections import defaultdict

import nltk
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "feedback.json")

class IntelligentTagger:
    def __init__(self, feedback_path: str = DEFAULT_FEEDBACK_PATH, tfidf_kwargs: dict = None):
        self.feedback_path = feedback_path
        self.tfidf_kwargs = tfidf_kwargs or {"max_features": 2000, "stop_words": "english", "ngram_range": (1, 2)}
        self.vectorizer = None
        self.corpus_filenames = []
        self.corpus_texts = []
        self.tfidf_matrix = None
        self.feature_names = []
        self.feedback = self._load_feedback()

    def _load_feedback(self):
        if os.path.exists(self.feedback_path):
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            initial = {"tag_weights": {}, "doc_feedback": {}}
            with open(self.feedback_path, "w", encoding="utf-8") as f:
                json.dump(initial, f, indent=2)
            return initial

    def _save_feedback(self):
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback, f, indent=2)

    # ----- ingestion -----
    def ingest_folder(self, folder_path: str):
        files = []
        for name in os.listdir(folder_path):
            if name.lower().endswith((".txt", ".md")):
                files.append(name)
        files.sort()
        texts = []
        for fname in files:
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        self.corpus_filenames = files
        self.corpus_texts = texts
        return files

    # ----- TF-IDF fit -----
    def fit_tfidf(self):
        if not self.corpus_texts:
            raise ValueError("No corpus loaded. Call ingest_folder first.")
        self.vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

    # ----- simple noun phrase extraction -----
    def _extract_noun_phrases(self, text: str, max_phrases: int = 20) -> List[str]:
        # naive: collect consecutive nouns / proper nouns as candidate phrases
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        phrases = []
        current = []
        for word, pos in tags:
            if pos.startswith("NN") or pos == "JJ":  # include adjectives to form phrases like 'design doc'
                current.append(word.lower())
            else:
                if current:
                    phrase = " ".join(current)
                    phrases.append(phrase)
                    current = []
        if current:
            phrases.append(" ".join(current))
        # deduplicate while preserving order
        seen = set()
        out = []
        for p in phrases:
            if p not in seen and len(p) > 1:
                seen.add(p)
                out.append(p)
        return out[:max_phrases]

    # ----- candidate tag generation -----
    def _tfidf_candidates_for_doc(self, doc_index: int, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF not fitted; call fit_tfidf().")
        row = self.tfidf_matrix[doc_index]
        # get nonzero features and their scores
        indices = row.nonzero()[1]
        scores = zip(indices, [row[0, i] for i in indices])
        # sort by score desc
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.feature_names[i], float(s)) for i, s in sorted_scores]

    def suggest_tags_for_doc(self, doc_index: int, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.corpus_texts:
            raise ValueError("No corpus loaded.")
        text = self.corpus_texts[doc_index]
        # tfidf candidates
        tfidf_cands = self._tfidf_candidates_for_doc(doc_index, top_k=top_k * 3)
        # noun phrase candidates
        np_cands = self._extract_noun_phrases(text, max_phrases=top_k * 3)
        # merge candidates with base score
        scores = defaultdict(float)
        for term, s in tfidf_cands:
            key = term.lower()
            scores[key] += s  # tfidf score
        for idx, phrase in enumerate(np_cands):
            # add a small boost for noun phrase; earlier noun phrases slightly higher
            scores[phrase] += max(0.5, 0.2 + (len(np_cands) - idx) * 0.05)
        # apply learned tag weights (feedback)
        for tag in list(scores.keys()):
            weight = self.feedback.get("tag_weights", {}).get(tag, 0.0)
            # final score mixes base score with learned adjustment
            # using multiplier (1 + weight) keeps update interpretable
            multiplier = 1.0 + float(weight)
            scores[tag] = scores[tag] * multiplier
        # sort and return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ----- feedback loop -----
    def apply_feedback(self, doc_name: str, tag: str, approve: bool, adjust_amount: float = 0.25):
        tag = tag.lower()
        # ensure doc_feedback structure
        if "doc_feedback" not in self.feedback:
            self.feedback["doc_feedback"] = {}
        if doc_name not in self.feedback["doc_feedback"]:
            self.feedback["doc_feedback"][doc_name] = {"approved": [], "rejected": []}

        if approve:
            if tag not in self.feedback["doc_feedback"][doc_name]["approved"]:
                self.feedback["doc_feedback"][doc_name]["approved"].append(tag)
            # remove from rejected if present
            if tag in self.feedback["doc_feedback"][doc_name]["rejected"]:
                self.feedback["doc_feedback"][doc_name]["rejected"].remove(tag)
            delta = adjust_amount
        else:
            if tag not in self.feedback["doc_feedback"][doc_name]["rejected"]:
                self.feedback["doc_feedback"][doc_name]["rejected"].append(tag)
            if tag in self.feedback["doc_feedback"][doc_name]["approved"]:
                self.feedback["doc_feedback"][doc_name]["approved"].remove(tag)
            delta = -adjust_amount

        # update global tag weight
        w = self.feedback.get("tag_weights", {}).get(tag, 0.0)
        w_new = w + delta
        # clamp weights to reasonable range
        w_new = max(-0.9, min(2.0, w_new))
        self.feedback.setdefault("tag_weights", {})[tag] = w_new
        self._save_feedback()
        return w_new

    def get_weights(self) -> Dict[str, float]:
        return dict(self.feedback.get("tag_weights", {}))

    # ----- convenience -----
    def suggest_all(self, top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        if not self.corpus_texts or self.tfidf_matrix is None:
            raise ValueError("Please ingest a folder and fit TF-IDF first.")
        results = {}
        for i, fname in enumerate(self.corpus_filenames):
            results[fname] = self.suggest_tags_for_doc(i, top_k=top_k)
        return results
