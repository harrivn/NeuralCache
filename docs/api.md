# NeuralCache API

This document describes the main public-facing Python API.

## `CacheCore`

```python
from src.cache.cache_core import CacheCore
from src.cache.prefetcher import Prefetcher
from src.model.neural_cache_model import NeuralCacheModel

model = NeuralCacheModel()
prefetcher = Prefetcher()
cache = CacheCore(capacity=1024, model=model, prefetcher=prefetcher)
```

### Methods

- **`get(key: str) -> Any | None`**
  - Returns the cached value for `key`, or `None` if it is not present.
  - Updates hit/miss metrics and the internal recency order.

- **`put(key: str, value: Any) -> None`**
  - Insert or update a cache entry.
  - Evicts the least recently used item when capacity is exceeded.

- **`get_or_load(key: str, loader: Callable[[str], Any]) -> Any`**
  - Returns the cached value if present; otherwise, calls `loader(key)`,
    inserts the result, and returns it.
  - Triggers model updates and prefetch decisions.

- **`get_stats() -> dict`**
  - Returns a dictionary of metrics including hit rate, evictions, and
    prefetch-related statistics.

## `NeuralCacheModel`

```python
from src.model.neural_cache_model import NeuralCacheModel

model = NeuralCacheModel(history_length=16, embedding_size=8, hidden_size=32)
```

### Methods

- **`register_access(key: str) -> None`**
  - Records that `key` was accessed and optionally performs an online update.

- **`predict_topk(candidate_keys: Iterable[str], k: int = 4) -> list[tuple[str, float]]`**
  - Given a list of candidate keys, returns up to `k` keys with associated
    scores, sorted by descending score.

## `Prefetcher`

```python
from src.cache.prefetcher import Prefetcher

prefetcher = Prefetcher(max_prefetch=4, confidence_threshold=0.6)
```

### Methods

- **`select_prefetches(ranked_candidates: Iterable[tuple[str, float]]) -> list[tuple[str, float]]`**
  - Filters and truncates the ranked predictions according to the policy
    parameters.

## CLI Entry Point

The `src/main.py` module exposes a simple CLI for running synthetic experiments:

```bash
python -m src.main --capacity 1024 --requests 10000
```

This will construct a NeuralCache instance, run a workload, and print metrics
to stdout.

