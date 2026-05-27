"""
Cache package for NeuralCache.

Provides the cache core abstraction and prefetching logic that leverage the
neural model to reduce miss rates.
"""

from .cache_core import CacheCore
from .prefetcher import Prefetcher

__all__ = ["CacheCore", "Prefetcher"]

