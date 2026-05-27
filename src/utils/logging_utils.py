"""
Logging utilities for NeuralCache.

Provides simple wrappers for structured logging so experiments can record
comparable results without pulling in a heavy logging framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict


def get_logger(name: str = "neuralcache") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any]) -> None:
    parts = [f"{k}={v}" for k, v in sorted(metrics.items())]
    logger.info("metrics " + " ".join(parts))

