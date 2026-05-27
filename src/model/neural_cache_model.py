"""
Lightweight neural model for NeuralCache.

This module defines `NeuralCacheModel`, a small online-trainable predictor that
uses NumPy-based embeddings and a single hidden layer to estimate the probability
that a key will be accessed in the near future.

The interface is intentionally simple so different models can be swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class NeuralCacheModel:
    """
    Simple fully-connected neural predictor for cache access patterns.

    The model maintains a learnable embedding per key and a small MLP that
    consumes a window of recent key embeddings and predicts which keys are
    likely to be accessed soon.
    """

    history_length: int = 16
    embedding_size: int = 8
    hidden_size: int = 32
    learning_rate: float = 0.01
    random_seed: int = 42

    _key_to_index: Dict[str, int] = field(default_factory=dict, init=False)
    _embeddings: np.ndarray | None = field(default=None, init=False)
    _W1: np.ndarray | None = field(default=None, init=False)
    _b1: np.ndarray | None = field(default=None, init=False)
    _W2: np.ndarray | None = field(default=None, init=False)
    _b2: np.ndarray | None = field(default=None, init=False)
    _history: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        # Initialize MLP weights lazily when we know the number of keys.
        input_size = self.history_length * self.embedding_size + self.embedding_size
        self._W1 = rng.normal(scale=0.1, size=(self.hidden_size, input_size))
        self._b1 = np.zeros(self.hidden_size)
        self._W2 = rng.normal(scale=0.1, size=(1, self.hidden_size))
        self._b2 = np.zeros(1)

    # --- Public API -----------------------------------------------------

    def register_access(self, key: str) -> None:
        """
        Record an access to `key` and update the model state.

        This method should be called on every cache access, regardless of hit/miss.
        """
        self._ensure_key(key)
        self._history.append(key)
        if len(self._history) > self.history_length:
            self._history.pop(0)

        # Optional: perform an online training step when we have enough history.
        if len(self._history) == self.history_length:
            self._online_update(target_key=key)

    def predict_topk(
        self,
        candidate_keys: Iterable[str],
        k: int = 4,
    ) -> List[Tuple[str, float]]:
        """
        Rank `candidate_keys` by predicted likelihood of future access.

        Returns a list of (key, score) pairs sorted by descending score.
        """
        if not candidate_keys:
            return []

        candidates = list(candidate_keys)
        scores: List[Tuple[str, float]] = []
        for key in candidates:
            self._ensure_key(key)
            score = float(self._score_candidate(key))
            scores.append((key, score))

        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores[:k]

    # --- Internal helpers -----------------------------------------------

    def _ensure_key(self, key: str) -> None:
        if key in self._key_to_index:
            return
        idx = len(self._key_to_index)
        self._key_to_index[key] = idx

        # Grow embeddings matrix as new keys appear.
        if self._embeddings is None:
            rng = np.random.default_rng(self.random_seed)
            self._embeddings = rng.normal(
                scale=0.1,
                size=(1, self.embedding_size),
            )
        else:
            rng = np.random.default_rng(self.random_seed + idx)
            new_vec = rng.normal(scale=0.1, size=(1, self.embedding_size))
            self._embeddings = np.vstack([self._embeddings, new_vec])

    def _get_embedding(self, key: str) -> np.ndarray:
        assert self._embeddings is not None
        idx = self._key_to_index[key]
        return self._embeddings[idx]

    def _encode_history(self) -> np.ndarray:
        """
        Encode the recent history into a flattened vector.
        """
        if not self._history:
            return np.zeros(self.history_length * self.embedding_size)

        padded_history = list(self._history)
        if len(padded_history) < self.history_length:
            pad = [padded_history[0]] * (self.history_length - len(padded_history))
            padded_history = pad + padded_history
        else:
            padded_history = padded_history[-self.history_length :]

        embeds = [self._get_embedding(k) for k in padded_history]
        stacked = np.concatenate(embeds, axis=0)
        return stacked.reshape(-1)

    def _forward(self, key: str) -> float:
        """
        Forward pass: given a candidate key, produce a score.
        """
        assert self._W1 is not None and self._W2 is not None
        assert self._b1 is not None and self._b2 is not None

        h_vec = self._encode_history()
        x_vec = self._get_embedding(key).reshape(-1)
        full_input = np.concatenate([h_vec, x_vec])

        hidden = np.tanh(self._W1 @ full_input + self._b1)
        logit = float(self._W2 @ hidden + self._b2)
        # Sigmoid to map to (0, 1)
        score = 1.0 / (1.0 + np.exp(-logit))
        return score

    def _score_candidate(self, key: str) -> float:
        return self._forward(key)

    def _online_update(self, target_key: str) -> None:
        """
        Placeholder for an online learning update.

        For a starter scaffold, this implementation only calls the forward pass
        to keep the model state "warm". A full implementation would compute
        gradients and update the parameters using the learning rate.
        """
        _ = self._forward(target_key)
        # TODO: implement gradient-based update in future iterations.

