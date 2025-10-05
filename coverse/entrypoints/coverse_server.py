import json
import os
import random
import time
from argparse import ArgumentParser
from datetime import datetime

import gradio as gr
from loguru import logger

from coverse.agents.converse_agent import ConverseAgent


def chat_with_openai_mock(messages):
    answer = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
    return answer


# %% gradio

def respond(message, chat_history: list[dict]):
    # logger.info(f'{chat_history=}')
    start_time = time.time()
    chat_history.append({'role': 'user', 'content': message})
    yield "", chat_history

    answer = agent.run(messages=chat_history)
    chat_history.append({'role': 'assistant', 'content': answer})
    elapsed = time.time() - start_time
    latency = args.min_latency + random.random() * (args.max_latency - args.min_latency)
    if elapsed < latency:
        time.sleep(latency - elapsed)
    logger.info(f'{chat_history=}')
    yield "", chat_history


def save_chat(user_id, chat_history):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_id = f"{user_id}_{timestamp}"

    example = {
        'data_id': data_id,
        'user_id': user_id,
        'messages': chat_history,
        'timestamp': timestamp,

    }
    output_path = f'data/chat_exp/{data_id}.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(json.dumps(example, ensure_ascii=False, indent=2))
    return f'Saved as {output_path}'


with gr.Blocks() as demo:
    gr.Markdown(f"# Coverse")
    user_id = gr.Textbox(label="User ID", value="unknown")
    chatbot = gr.Chatbot(type="messages", value=[], height=400)
    input_text = gr.Textbox(label='Input', max_lines=1)
    save_button = gr.Button(value="Save Chat")
    save_state = gr.Textbox(label="Save State", max_lines=1)
    clear_button = gr.ClearButton([input_text, chatbot, save_state])

    input_text.submit(respond, inputs=[input_text, chatbot], outputs=[input_text, chatbot])
    save_button.click(save_chat, inputs=[user_id, chatbot], outputs=save_state)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek-v3-1-terminus')
    parser.add_argument('--min-latency', type=float, default=5.0)
    parser.add_argument('--max-latency', type=float, default=8.0)
    args = parser.parse_args()
    agent = ConverseAgent(model_name=args.model)

    # warmup
    logger.info(respond('hi', []))
    demo.launch()
