"""Category weighting for chunk and query ranking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Any

DEFAULT_CATEGORIES: Dict[str, List[str]] = {
    "definitions": [
        "definition",
        "means",
        "refers to",
        "defined",
        "term",
        "includes",
    ],
    "requirements": [
        "shall",
        "must",
        "required",
        "requirement",
        "prohibited",
        "must not",
    ],
    "exceptions": [
        "except",
        "unless",
        "however",
        "provided that",
        "notwithstanding",
    ],
    "procedures": [
        "procedure",
        "process",
        "steps",
        "how to",
        "method",
        "submit",
        "file",
    ],
    "scope": [
        "scope",
        "applicability",
        "applies to",
        "covered",
        "effective date",
    ],
    "reporting": [
        "report",
        "disclose",
        "record",
        "retention",
        "notify",
        "filing",
    ],
}


@dataclass
class CategoryScorer:
    """Scores text against a multi-label category schema."""

    categories: Dict[str, List[str]]
    smoothing: float = 0.05

    def score_text(self, text: str) -> Dict[str, float]:
        """Return normalized category weights for text."""
        if not self.categories:
            return {}

        lower = text.lower()
        counts = {category: 0 for category in self.categories}

        for category, keywords in self.categories.items():
            counts[category] = self._count_keywords(lower, keywords)

        total_hits = sum(counts.values())
        weights: Dict[str, float] = {}

        if total_hits == 0:
            uniform = 1.0 / float(len(self.categories))
            for category in self.categories:
                weights[category] = uniform
            return weights

        for category in counts:
            weights[category] = counts[category] + self.smoothing

        total_weight = sum(weights.values())
        for category in weights:
            weights[category] = weights[category] / total_weight

        return weights

    def serialize_weights(self, weights: Dict[str, float]) -> str:
        """Serialize weights to a JSON string for metadata storage."""
        return json.dumps(weights, sort_keys=True)

    def parse_weights(self, value: Any) -> Dict[str, float]:
        """Parse weights from metadata."""
        if not value:
            return {}
        if isinstance(value, dict):
            return {key: float(val) for key, val in value.items()}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return {key: float(val) for key, val in parsed.items()}
            except json.JSONDecodeError:
                return {}
        return {}

    def top_labels(self, weights: Dict[str, float], top_n: int = 3) -> List[str]:
        """Return the top N category labels by weight."""
        if not weights:
            return []
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        return [label for label, _ in ranked[:top_n]]

    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        count = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if " " in keyword_lower:
                count += text.count(keyword_lower)
            else:
                count += len(re.findall(r"\\b" + re.escape(keyword_lower) + r"\\b", text))
        return count
