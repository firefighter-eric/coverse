import gradio as gr
import openai
import random
from loguru import logger
import re

client = openai.OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

MODEL_NAME = "qwen3:30b"  # 用 ollama list 查到的模型名
system_prompt = "你是一个有帮助的助手，每次回答不超过10字， /set nothink"

def chat_with_openai(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    answer = re.sub('<think>.+</think>', '', answer, flags=re.MULTILINE)
    answer = answer.strip()
    return answer

def chat_with_openai_mock(messages):
    answer = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
    return answer


# %% gradio

def respond(message, chat_history: list[dict]):
    logger.info(f'{chat_history=}')
    chat_history.append({'role': 'user', 'content': message})
    messages = [{'role': 'system', 'content': system_prompt}] + chat_history
    answer = chat_with_openai(messages)
    chat_history.append({'role': 'assistant', 'content': answer})
    logger.info(f'{chat_history=}')
    return "", chat_history
    
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages", value=[])
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])


if __name__ == "__main__":
    logger.info(respond('hi', []))
    demo.launch()

