"""
Cache core for NeuralCache.

Implements a simple key-value cache with LRU-like eviction and hooks into the
neural prediction model and prefetcher.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable
from typing import List, Tuple

from ..model.neural_cache_model import NeuralCacheModel
from .prefetcher import Prefetcher
from ..utils.metrics import MetricsTracker


Loader = Callable[[str], Any]


@dataclass
class CacheCore:
    """
    High-level cache interface with neural prefetching.

    Parameters
    ----------
    capacity:
        Maximum number of entries to store.
    model:
        Predictive model used to rank candidate keys for prefetching.
    prefetcher:
        Policy object that decides which predictions to execute as prefetches.
    """

    capacity: int
    model: NeuralCacheModel
    prefetcher: Prefetcher

    _store: "OrderedDict[str, Any]" = field(default_factory=OrderedDict, init=False)
    _metrics: MetricsTracker = field(default_factory=MetricsTracker, init=False)

    def get(self, key: str) -> Any | None:
        """
        Get a value from the cache, returning None on miss.
        """
        hit = key in self._store
        self._metrics.record_access(hit=hit)

        self.model.register_access(key)

        if hit:
            value = self._store.pop(key)
            self._store[key] = value  # mark as most recently used
            return value
        return None

    def put(self, key: str, value: Any) -> None:
        """
        Insert or update a key/value pair in the cache.
        """
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.capacity:
            evicted_key, _ = self._store.popitem(last=False)
            self._metrics.record_eviction(evicted_key)
        self._store[key] = value

    def get_or_load(self, key: str, loader: Loader) -> Any:
        """
        Get the value for `key`, loading it with `loader` on miss.

        This method also triggers prefetching based on model predictions.
        """
        value = self.get(key)
        miss = value is None

        if miss:
            value = loader(key)
            self.put(key, value)

        # Decide on prefetch candidates after observing this access.
        candidates = self._candidate_keys()
        ranked = self.model.predict_topk(candidates)
        prefetch_keys = self.prefetcher.select_prefetches(ranked)

        for pf_key, score in prefetch_keys:
            if pf_key in self._store:
                continue
            pf_value = loader(pf_key)
            self.put(pf_key, pf_value)
            self._metrics.record_prefetch(pf_key, score, pf_hit=False)

        return value

    def _candidate_keys(self) -> Iterable[str]:
        """
        Candidate keys for prefetching.

        For a starter implementation, we consider the keys currently in the cache.
        Future implementations could include keys from backing storage.
        """
        return list(self._store.keys())

    def get_stats(self) -> Dict[str, float | int]:
        """
        Return a snapshot of cache and prefetch performance metrics.
        """
        stats = self._metrics.snapshot()
        stats["size"] = len(self._store)
        stats["capacity"] = self.capacity
        return stats

