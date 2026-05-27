"""
CLI entrypoint for running NeuralCache experiments.

This module wires together the cache core, neural model, and prefetcher,
and provides a simple command-line interface for running synthetic workloads.
"""

from __future__ import annotations

import argparse
from typing import Any

from .cache.cache_core import CacheCore
from .cache.prefetcher import Prefetcher
from .model.neural_cache_model import NeuralCacheModel


def build_cache(capacity: int) -> CacheCore:
    model = NeuralCacheModel(
        history_length=16,
        embedding_size=8,
        hidden_size=32,
    )
    prefetcher = Prefetcher(
        max_prefetch=4,
        confidence_threshold=0.6,
    )
    cache = CacheCore(
        capacity=capacity,
        model=model,
        prefetcher=prefetcher,
    )
    return cache


def run_synthetic_workload(cache: CacheCore, num_requests: int = 10_000) -> None:
    """
    Run a simple synthetic workload to exercise the cache and model.
    """

    def loader(key: str) -> Any:
        return f"value-for-{key}"

    for i in range(num_requests):
        key = f"addr-{i % 512}"
        cache.get_or_load(key=key, loader=loader)

    stats = cache.get_stats()
    print("=== NeuralCache Synthetic Run ===")
    for name, value in stats.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuralCache experiment runner")
    parser.add_argument(
        "--capacity",
        type=int,
        default=1024,
        help="Cache capacity (number of entries).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10_000,
        help="Number of synthetic requests to issue.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cache = build_cache(capacity=args.capacity)
    run_synthetic_workload(cache, num_requests=args.requests)


if __name__ == "__main__":
    main()

