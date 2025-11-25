# retrieval/query_expansion.py
from typing import List
import re

class SimpleQueryExpander:
    def __init__(self):
        pass

    def expand(self, query: str) -> List[str]:
        """
        Return small list of variant queries.
        Heuristics: expand acronyms, synonyms, quarter/year expansions, date phrases.
        """
        variations = [query]
        # numeric quarter expansion e.g., Q3 -> "July August September"
        q_match = re.search(r'\bQ([1-4])\b', query, flags=re.IGNORECASE)
        if q_match:
            q = int(q_match.group(1))
            month_map = {1:"Jan Feb Mar", 2:"Apr May Jun", 3:"Jul Aug Sep", 4:"Oct Nov Dec"}
            variations.append(f"{query} {month_map.get(q)}")
        # year shorthand: 'FY24' -> 'Fiscal Year 2024'
        fy = re.search(r'FY(\d{2,4})', query, flags=re.IGNORECASE)
        if fy:
            year = fy.group(1)
            if len(year) == 2:
                year = "20" + year
            variations.append(query.replace(fy.group(0), f"Fiscal Year {year}"))
        # TODO: integrate LLM-based variant generator (optional)
        return list(dict.fromkeys(variations))  # dedupe, preserve order
