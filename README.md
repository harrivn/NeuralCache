# NeuralCache

NeuralCache is an adaptive caching system that leverages a lightweight neural network to predict memory access patterns and pre-fetch data intelligently, reducing cache miss rates in high-throughput systems. It is designed as an experimentation framework for computer engineering and systems research.

## Architecture Overview

NeuralCache is organized into three main layers:

- **Access Stream & Cache Interface**
  - Exposes a simple API (`CacheCore`) that mimics a key-value cache.
  - Collects access traces (hits, misses, temporal locality) from client code.
  - Emits events used to train and evaluate the predictive model.

- **Predictive Model (`model` package)**
  - Implements a lightweight neural predictor using pure Python and NumPy.
  - Consumes a fixed-size window of recent accesses and outputs:
    - A probability distribution over candidate keys likely to be accessed soon.
    - Confidence scores used to throttle prefetching aggressiveness.
  - Supports online, incremental training to adapt to workload phase changes.

- **Prefetching & Policy Engine (`cache` package)**
  - Uses the model’s predictions to decide:
    - Which keys to prefetch into the cache.
    - When to prefetch (e.g., on every access vs. only on misses).
  - Wraps a pluggable underlying cache implementation (e.g., LRU).
  - Exposes metrics (hit rate, prefetch precision/recall, overhead).

High-level data flow:

1. Client calls `get(key)` / `put(key, value)` on `CacheCore`.
2. `CacheCore` records the access and queries the `NeuralCacheModel`.
3. The model predicts future keys; `Prefetcher` selects a subset to prefetch.
4. Prefetch operations warm the underlying cache before actual requests arrive.
5. Metrics are updated continuously for evaluation and tuning.

## Setup Instructions

### Prerequisites

- Python 3.9+ recommended
- `pip` (Python package manager)
- A virtual environment tool such as `venv` or `conda`

### Installation

From the `project` directory:

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Example

Below is a minimal example showing how to use NeuralCache in a Python script.

```python
from src.cache.cache_core import CacheCore
from src.model.neural_cache_model import NeuralCacheModel
from src.cache.prefetcher import Prefetcher

# Initialize model and components
model = NeuralCacheModel(
    history_length=16,
    embedding_size=8,
    hidden_size=32,
)
prefetcher = Prefetcher(max_prefetch=4, confidence_threshold=0.6)
cache = CacheCore(
    capacity=1024,
    model=model,
    prefetcher=prefetcher,
)

# Populate the cache
for i in range(2000):
    cache.put(f"key-{i}", f"value-{i}")

# Simulate a workload
for i in range(2000):
    key = f"key-{i % 500}"  # some reuse
    value = cache.get_or_load(
        key=key,
        loader=lambda k: f"loaded-{k}",  # called on misses
    )

# Inspect metrics
stats = cache.get_stats()
print("Hit rate:", stats["hit_rate"])
print("Prefetches:", stats["prefetches"])
print("Avg prefetch score:", stats["avg_prefetch_score"])
print("Evictions:", stats["evictions"])
```

## Project Layout

- `src/`
  - `main.py`: CLI entrypoint and sample runner.
  - `model/`: Neural prediction model implementations.
  - `cache/`: Cache core, policies, and prefetching logic.
  - `utils/`: Shared helpers (metrics, logging, etc.).
- `tests/`: Unit tests for cache behavior and model components.
- `docs/`: Additional documentation (architecture deep-dive, API reference).

## Roadmap

- Add alternative models (e.g., recurrent and attention-based predictors).
- Integrate with real key-value stores (Redis, RocksDB) as backends.
- Add trace replay tools for standard cache benchmark suites.
- Implement configuration files for tuning and experiment management.

