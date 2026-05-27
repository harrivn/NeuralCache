"""
Metrics utilities for NeuralCache.

Provides a simple in-memory tracker for cache and prefetch performance
statistics, suitable for quick experiments and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricsTracker:
    """
    Tracks basic cache and prefetch metrics.

    Metrics include:
    - accesses, hits, misses
    - evictions
    - prefetches issued
    """

    accesses: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetches: int = 0

    _prefetch_sum_score: float = field(default=0.0, init=False)

    def record_access(self, hit: bool) -> None:
        self.accesses += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def record_eviction(self, key: str) -> None:  # noqa: ARG002
        self.evictions += 1

    def record_prefetch(self, key: str, score: float, pf_hit: bool) -> None:  # noqa: ARG002
        self.prefetches += 1
        self._prefetch_sum_score += float(score)
        # In a fuller implementation we would also record whether prefetches were
        # subsequently used (hit) or wasted (never accessed).

    def snapshot(self) -> Dict[str, float | int]:
        hit_rate = (self.hits / self.accesses) if self.accesses else 0.0
        avg_prefetch_score = (
            self._prefetch_sum_score / self.prefetches if self.prefetches else 0.0
        )
        return {
            "accesses": self.accesses,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "prefetches": self.prefetches,
            "avg_prefetch_score": avg_prefetch_score,
        }

