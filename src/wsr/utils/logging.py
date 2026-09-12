"""Logging setup: console plus an optional run log under results/logs."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: str = "INFO", logfile: Path | None = None) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly (handlers are not duplicated)."""
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    has_console = any(type(h) is logging.StreamHandler for h in logger.handlers)
    if not has_console:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(stream)
    if logfile is not None:
        target = Path(logfile).resolve()
        has_file = any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target for h in logger.handlers)
        if not has_file:
            target.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(fh)
    return logger
