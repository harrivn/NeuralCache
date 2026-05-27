# NeuralCache Architecture

This document provides a more detailed look at the architecture of NeuralCache.

## High-Level Components

- **CacheCore**
  - Wraps an in-memory key-value store with LRU-style eviction.
  - Emits access events (hits, misses, evictions) to the metrics tracker.
  - For every access, notifies the `NeuralCacheModel` and executes prefetches
    selected by the `Prefetcher`.

- **NeuralCacheModel**
  - Maintains embeddings for keys and a small MLP that operates on recent access
    history.
  - Produces a scalar score per candidate key representing predicted likelihood
    of near-future access.
  - Intended to support online learning, making it responsive to workload phase changes.

- **Prefetcher**
  - Takes a ranked list of (key, score) predictions and applies policy rules:
    - Thresholding by confidence.
    - Capping the number of prefetches per decision.
  - Returns the final prefetch set back to `CacheCore`.

- **Utilities**
  - `MetricsTracker` aggregates performance counters and exposes a simple API.
  - `logging_utils` formats and emits metric snapshots for experiments.

## Data Flow

1. A client accesses the cache via `get_or_load(key, loader)`.
2. `CacheCore` checks if the key is present:
   - On hit: returns the value and updates hit/miss counters.
   - On miss: uses `loader` to fetch the value, inserts it, and updates counters.
3. `CacheCore` calls `model.register_access(key)` to update history and state.
4. `CacheCore` creates a list of candidate keys (currently the ones in cache) and
   asks the model to `predict_topk`.
5. `Prefetcher` filters and truncates the predictions and returns a set of keys
   to prefetch.
6. `CacheCore` prefetches these keys (if not already cached) using the loader,
   and any changes are recorded in metrics.

## Extensibility

- **Alternative Cache Backends**: Replace the in-memory `OrderedDict` with
  a backend that wraps Redis, Memcached, or a disk-based KV store.
- **Advanced Models**: Swap `NeuralCacheModel` with models that use sequence
  models (RNNs, Transformers) or exploit address-space structure.
- **Policy Research**: Implement alternative policies in the `Prefetcher`
  (e.g., cost-aware, deadline-aware prefetching).

