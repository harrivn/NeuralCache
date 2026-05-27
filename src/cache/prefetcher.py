"""
Prefetching policy for NeuralCache.

Responsible for deciding which model predictions to execute and how aggressively
to prefetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Prefetcher:
    """
    Prefetching policy based on ranked model predictions.

    Parameters
    ----------
    max_prefetch:
        Upper bound on the number of prefetches to issue per access.
    confidence_threshold:
        Minimum score required for a prediction to be considered for prefetch.
    """

    max_prefetch: int = 4
    confidence_threshold: float = 0.5

    def select_prefetches(
        self,
        ranked_candidates: Iterable[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Filter and truncate a ranked list of (key, score) pairs.
        """
        selected: List[Tuple[str, float]] = []
        for key, score in ranked_candidates:
            if score < self.confidence_threshold:
                continue
            selected.append((key, score))
            if len(selected) >= self.max_prefetch:
                break
        return selected

