from __future__ import annotations

import random
import time
from pathlib import Path

import gradio as gr

from coverse.core.agents import AgentConfig, ConversationAgent, GenerationConfig
from coverse.core.backends import ModelBackendConfig, OpenAIChatBackend
from coverse.core.io import ExperimentIO
from coverse.core.types import ExperimentMetadata, utc_timestamp
from coverse.topics.multi_chat.prompts import DEFAULT_STORY_SYSTEM_PROMPT


def build_chat_agent(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key_env: str | None,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> ConversationAgent:
    resolved_system_prompt = system_prompt or DEFAULT_STORY_SYSTEM_PROMPT
    backend = OpenAIChatBackend(
        ModelBackendConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
        )
    )
    config = AgentConfig(
        name="assistant",
        system_prompt=resolved_system_prompt,
        generation=GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ),
    )
    return ConversationAgent(config, backend)


def create_app(
    *,
    provider: str,
    model: str,
    output_dir: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
    system_prompt: str | None = DEFAULT_STORY_SYSTEM_PROMPT,
    min_latency: float = 0.0,
    max_latency: float = 0.0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024,
):
    agent = build_chat_agent(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        system_prompt=system_prompt or DEFAULT_STORY_SYSTEM_PROMPT,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    experiment_io = ExperimentIO(output_dir)

    def respond(message: str, chat_history: list[dict]):
        start_time = time.time()
        chat_history = list(chat_history)
        chat_history.append({"role": "user", "content": message})
        yield "", chat_history

        answer = agent.respond(chat_history)
        chat_history.append({"role": "assistant", "content": answer})
        elapsed = time.time() - start_time
        latency = min_latency + random.random() * max(0.0, max_latency - min_latency)
        if elapsed < latency:
            time.sleep(latency - elapsed)
        yield "", chat_history

    def save_chat(user_id: str, chat_history: list[dict]):
        run_name = f"{utc_timestamp().replace(':', '-')}-{user_id or 'unknown'}"
        run_dir = experiment_io.prepare_run_dir("chat_app", run_name)
        metadata = ExperimentMetadata(
            topic="chat_app",
            command="coverse app serve",
            args={"user_id": user_id},
            model={"provider": provider, "model": model},
            output_dir=str(run_dir),
        )
        experiment_io.write_metadata(run_dir, metadata)
        transcript_path = experiment_io.write_json(run_dir, "transcript.json", chat_history)
        return f"Saved as {Path(transcript_path)}"

    with gr.Blocks() as demo:
        gr.Markdown("# Coverse")
        user_id = gr.Textbox(label="User ID", value="unknown")
        chatbot = gr.Chatbot(type="messages", value=[], height=400)
        input_text = gr.Textbox(label="Input", max_lines=1)
        save_button = gr.Button(value="Save Chat")
        save_state = gr.Textbox(label="Save State", max_lines=1)
        gr.ClearButton([input_text, chatbot, save_state])

        input_text.submit(respond, inputs=[input_text, chatbot], outputs=[input_text, chatbot])
        save_button.click(save_chat, inputs=[user_id, chatbot], outputs=save_state)
    return demo


def launch_app(
    *,
    provider: str,
    model: str,
    output_dir: str,
    host: str,
    port: int,
    base_url: str | None = None,
    api_key_env: str | None = None,
    system_prompt: str | None = DEFAULT_STORY_SYSTEM_PROMPT,
    min_latency: float = 0.0,
    max_latency: float = 0.0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024,
):
    app = create_app(
        provider=provider,
        model=model,
        output_dir=output_dir,
        base_url=base_url,
        api_key_env=api_key_env,
        system_prompt=system_prompt or DEFAULT_STORY_SYSTEM_PROMPT,
        min_latency=min_latency,
        max_latency=max_latency,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    app.launch(server_name=host, server_port=port)
