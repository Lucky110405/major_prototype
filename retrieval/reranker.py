import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None):
        """Lightweight cross-encoder reranker using a HuggingFace sequence classification model.

        The model should accept a pair (query, document) and produce a relevance score (logit).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load reranker model {model_name}: {e}")
            raise

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None, batch_size: int = 16) -> List[Dict[str, Any]]:
        """Rerank candidates.

        candidates should be a list of dicts containing at least an identifier and a text snippet under metadata['text_excerpt'] or metadata['text'].
        Returns the same candidate dicts augmented with 'rerank_score' and sorted descending.
        """
        if not candidates:
            return []

        pairs = []
        texts = []
        for c in candidates:
            text = c.get('metadata', {}).get('text_excerpt') or c.get('metadata', {}).get('text') or c.get('text') or ''
            pairs.append((query, text))
            texts.append(text)

        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                # For cross-encoder, encode pairs as two sequences
                queries = [p[0] for p in batch_pairs]
                docs = [p[1] for p in batch_pairs]
                inputs = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                # If logits is (batch,1) or (batch,num_labels) take appropriate scalar
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().tolist()
                else:
                    # fallback: take the first logit
                    batch_scores = logits[:, 0].cpu().tolist()
                scores.extend(batch_scores)

        # Attach scores to candidates
        for c, s in zip(candidates, scores):
            c['rerank_score'] = float(s)

        # Sort by rerank_score descending
        ranked = sorted(candidates, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        return ranked
# retrieval/reranker.py
from typing import List, Dict, Any
import os

class LLMReranker:
    def __init__(self, llm_fn=None):
        """
        llm_fn: a function that accepts (prompt) -> text; you can pass OpenAI/GPT wrapper or an internal LLM
        """
        self.llm_fn = llm_fn

    def rerank(self, query: str, docs: List[Dict[str,Any]], top_k: int = 5) -> List[Dict[str,Any]]:
        """
        docs: list of dicts {id, text, metadata, score}
        Returns docs re-ordered by relevance per LLM.
        """
        if not self.llm_fn:
            # fallback: return docs as-is
            return docs[:top_k]

        # build compact prompt
        snippet_lines = []
        for i, d in enumerate(docs):
            snippet = d.get("text", "")
            snippet = snippet.replace("\n", " ")[:500]  # keep small
            snippet_lines.append(f"[{i}] {snippet}")

        prompt = (
            f"Given the user query:\n\n\"{query}\"\n\n"
            "Rank the following document snippets in order of relevance. "
            "Return a comma-separated list of indices, most relevant first.\n\n"
            "Snippets:\n" + "\n".join(snippet_lines) + "\n\nAnswer:"
        )

        resp = self.llm_fn(prompt)
        # parse indices
        # accept "0,2,1" or "1 0 2" etc.
        tokens = [t for t in resp.replace(",", " ").split() if t.isdigit()]
        order = [int(t) for t in tokens if 0 <= int(t) < len(docs)]
        if not order:
            return docs[:top_k]
        ordered = [docs[i] for i in order]
        return ordered[:top_k]
