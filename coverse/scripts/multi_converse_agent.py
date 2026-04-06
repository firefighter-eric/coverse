from coverse.cli.main import main
from coverse.config import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "topic",
                "multi-chat",
                "--provider",
                DEFAULT_LLM_PROVIDER,
                "--model",
                DEFAULT_LLM_MODEL,
                "--base-url",
                DEFAULT_LLM_BASE_URL,
                "--api-key-env",
                DEFAULT_LLM_API_KEY_ENV,
                "--prompts-path",
                "data/coverse_pe/story_prompt.txt",
                "--tag",
                "legacy-script",
            ]
        )
    )
