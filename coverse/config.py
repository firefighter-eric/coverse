from __future__ import annotations

import os

import dotenv

dotenv.load_dotenv()


def env_with_fallback(primary: str, legacy: str | None, default: str) -> str:
    if primary in os.environ:
        return os.environ[primary]
    if legacy and legacy in os.environ:
        return os.environ[legacy]
    return default


DEFAULT_LLM_PROVIDER = env_with_fallback("LLM_PROVIDER", "VITE_LLM_PROVIDER", "deepseek")
DEFAULT_LLM_BASE_URL = env_with_fallback(
    "LLM_BASE_URL",
    "VITE_LLM_BASE_URL",
    "https://api.deepseek.com/v1",
)
DEFAULT_LLM_API_KEY_ENV = env_with_fallback(
    "LLM_API_KEY_ENV",
    "VITE_LLM_API_KEY_ENV",
    "LLM_API_KEY",
)
DEFAULT_LLM_MODEL = env_with_fallback("LLM_MODEL", "VITE_LLM_MODEL", "deepseek-chat")
