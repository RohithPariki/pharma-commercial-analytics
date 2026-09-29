"""Configuration loading and shared path helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the YAML config with resolved absolute paths."""

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        p = PROJECT_ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
