import json
import os
import random
from datetime import datetime

import gradio as gr
from loguru import logger

from coverse.utils.model_clients import ModelClient

MODEL_NAME = "qwen3:30b"
model_client = ModelClient(MODEL_NAME)
system_prompt = """
你是一名心理学实验中的对话参与者，需要严格遵守以下规则：  
1. 我们轮流造句，每次只说一句话。  
2. 回答长度必须在10到20字之间。  
3. 不允许使用任何标点符号和换行。  
4. 每句话都必须有故事性和口语化风格，像一个人在自然讲故事。  
5. 只能输出回答，不允许有额外解释或格式。  
"""


def chat_with_openai_mock(messages):
    answer = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
    return answer


# %% gradio

def respond(message, chat_history: list[dict]):
    logger.info(f'{chat_history=}')
    chat_history.append({'role': 'user', 'content': message})
    messages = [{'role': 'system', 'content': system_prompt}] + chat_history
    answer = model_client.chat(messages)
    chat_history.append({'role': 'assistant', 'content': answer})
    logger.info(f'{chat_history=}')
    return "", chat_history


def save_chat(user_id, chat_history):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
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
        f.write(json.dumps(example, ensure_ascii=False))
    return example


with gr.Blocks() as demo:
    user_id = gr.Textbox(label="User ID", value="unknown")
    chatbot = gr.Chatbot(type="messages", value=[])
    msg = gr.Textbox()
    clear_button = gr.ClearButton([msg, chatbot])
    save_button = gr.Button("Save Chat")

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    save_button.click(save_chat, inputs=[user_id, chatbot])

if __name__ == "__main__":
    logger.info(respond('hi', []))
    demo.launch()
