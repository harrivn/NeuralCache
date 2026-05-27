"""
Model package for NeuralCache.

Contains implementations of lightweight neural predictors that operate over
recent access histories and produce ranked candidate keys for prefetching.
"""
from .neural_cache_model import NeuralCacheModel

__all__ = ["NeuralCacheModel"]

