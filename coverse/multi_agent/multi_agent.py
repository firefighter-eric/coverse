
import gradio as gr
import openai
import random
from loguru import logger
import re
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()



# client = openai.OpenAI(
#     api_key="ollama",
#     base_url="http://127.0.0.1:11434/v1",
# )

client = openai.OpenAI(
    api_key=os.environ.get('ARK_API_KEY'),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

MODEL_NAME = "doubao-seed-1-6-250615"

def chat_with_openai(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    # postporocess
    answer = re.sub('<think>.+</think>', '', answer, flags=re.MULTILINE)
    answer = answer.strip()
    return answer

def chat_with_agent(messages, agent_name):
    # change role to agent_name
    messages = messages.copy()
    for msg in messages:
        if msg['role'] == agent_name:
            msg['role'] = 'assistant'
        elif msg['role'] == 'system':
            continue
        else:
            msg['role'] = 'user'
    answer = chat_with_openai(messages)
    return answer


def multi_agent(system_prompt = '', first_message: str=''):
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.append({'role': 'agent_1', 'content': first_message})
    for i in tqdm(range(5)):
        answer = chat_with_agent(messages=messages, agent_name='agent_1')
        messages.append({'role': 'agent_1', 'content': answer})
        answer = chat_with_agent(messages=messages, agent_name='agent_2')
        messages.append({'role': 'agent_2', 'content': answer})
    return messages

if __name__ == '__main__':
    system_prompt = "你是一个有帮助的助手，每次回答不超过10字， /set nothink"

    m = multi_agent(system_prompt, first_message='你好')
    




