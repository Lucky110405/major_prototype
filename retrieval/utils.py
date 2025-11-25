# retrieval/utils.py
from typing import Dict, Any, List

def apply_metadata_filter(results: List[Dict], metadata_filter: Dict) -> List[Dict]:
    if not metadata_filter:
        return results
    out = []
    for r in results:
        md = r.get("metadata", {}) or {}
        match = True
        for k, v in metadata_filter.items():
            # simple equality filter; expand with ranges, regex later
            if md.get(k) != v:
                match = False
                break
        if match:
            out.append(r)
    return out

def multimodal_merge(text_results: List[Dict], image_results: List[Dict], audio_results: List[Dict], weights=(0.6,0.3,0.1)):
    """
    Basic fusion of results from different modalities.
    Each list contains {id, score, text, metadata}
    We normalize scores per modality and produce fused score.
    """
    import numpy as np
    def normalize(scores):
        arr = np.array(scores, dtype=float)
        if arr.max() == 0:
            return arr
        return arr / (arr.max() + 1e-12)

    combined = {}
    # text
    t_scores = [r["score"] for r in text_results]
    tn = normalize(t_scores) if len(t_scores)>0 else []
    for i, r in enumerate(text_results):
        combined.setdefault(r["id"], {"id": r["id"], "text": r.get("text"), "metadata": r.get("metadata", {}), "score": 0.0})
        combined[r["id"]]["score"] += weights[0] * (tn[i] if len(tn)>i else r["score"])
    # image
    i_scores = [r["score"] for r in image_results]
    in_ = normalize(i_scores) if len(i_scores)>0 else []
    for i, r in enumerate(image_results):
        combined.setdefault(r["id"], {"id": r["id"], "text": r.get("text"), "metadata": r.get("metadata", {}), "score": 0.0})
        combined[r["id"]]["score"] += weights[1] * (in_[i] if len(in_)>i else r["score"])
    # audio
    a_scores = [r["score"] for r in audio_results]
    an = normalize(a_scores) if len(a_scores)>0 else []
    for i, r in enumerate(audio_results):
        combined.setdefault(r["id"], {"id": r["id"], "text": r.get("text"), "metadata": r.get("metadata", {}), "score": 0.0})
        combined[r["id"]]["score"] += weights[2] * (an[i] if len(an)>i else r["score"])

    # sort by fused score
    out = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return out
