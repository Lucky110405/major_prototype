# retrieval/bm25_index.py
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

class BM25Index:
    def __init__(self, docs: List[Dict]):
        """
        docs: list of {"id": id, "text": text, "metadata": {...}}
        """
        self.docs = docs
        self.texts = [d["text"] or "" for d in docs]
        self.ids = [d["id"] for d in docs]
        tokenized = [word_tokenize(t.lower()) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, q: str, top_k: int = 10) -> List[Dict]:
        tokens = word_tokenize(q.lower())
        scores = self.bm25.get_scores(tokens)
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in ranked_idx:
            results.append({"id": self.ids[i], "score": float(scores[i]), "text": self.texts[i]})
        return results
