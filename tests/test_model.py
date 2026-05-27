"""
Basic tests for NeuralCacheModel behavior.
"""

from __future__ import annotations

from src.model.neural_cache_model import NeuralCacheModel


def test_model_register_and_predict() -> None:
    model = NeuralCacheModel(history_length=4, embedding_size=4, hidden_size=8)

    # Warm up with some accesses
    for key in ["a", "b", "c", "a", "b", "a"]:
        model.register_access(key)

    # Ask for predictions among candidates
    candidates = ["a", "b", "c", "d"]
    ranked = model.predict_topk(candidates, k=2)

    assert len(ranked) <= 2
    keys = [k for k, _ in ranked]
    for k in keys:
        assert k in candidates

