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
