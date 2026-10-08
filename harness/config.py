"""Config loading (config.yaml + .env)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return
    load_dotenv(ROOT / ".env")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


def api_key(cfg: dict) -> str:
    load_env()
    key = os.environ.get(cfg.get("api_key_env", "NVIDIA_API_KEY"), "").strip()
    if not key:
        raise SystemExit(
            f"{cfg.get('api_key_env', 'NVIDIA_API_KEY')} is not set. "
            "Create a .env file containing: NVIDIA_API_KEY=nvapi-..."
        )
    return key


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
