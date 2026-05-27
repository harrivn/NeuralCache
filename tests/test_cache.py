"""
Basic tests for the CacheCore and prefetch integration.
"""

from __future__ import annotations

from typing import Any

from src.cache.cache_core import CacheCore
from src.cache.prefetcher import Prefetcher
from src.model.neural_cache_model import NeuralCacheModel


def _loader(key: str) -> Any:
    return f"value-{key}"


def test_cache_hit_and_miss_behavior() -> None:
    model = NeuralCacheModel()
    prefetcher = Prefetcher(max_prefetch=0)  # disable for basic test
    cache = CacheCore(capacity=2, model=model, prefetcher=prefetcher)

    # Miss then load
    assert cache.get("a") is None
    v = cache.get_or_load("a", loader=_loader)
    assert v == "value-a"
    assert cache.get("a") == "value-a"

    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


def test_eviction_on_capacity() -> None:
    model = NeuralCacheModel()
    prefetcher = Prefetcher(max_prefetch=0)
    cache = CacheCore(capacity=2, model=model, prefetcher=prefetcher)

    cache.get_or_load("a", loader=_loader)
    cache.get_or_load("b", loader=_loader)
    cache.get_or_load("c", loader=_loader)  # should evict one entry

    stats = cache.get_stats()
    assert stats["size"] == 2
    assert stats["evictions"] >= 1

