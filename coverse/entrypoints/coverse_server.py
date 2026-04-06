from coverse.apps.gradio_app import launch_app
from coverse.config import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)


if __name__ == "__main__":
    launch_app(
        provider=DEFAULT_LLM_PROVIDER,
        model=DEFAULT_LLM_MODEL,
        output_dir="outputs",
        host="127.0.0.1",
        port=7860,
        base_url=DEFAULT_LLM_BASE_URL,
        api_key_env=DEFAULT_LLM_API_KEY_ENV,
    )
