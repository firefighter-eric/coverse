from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from coverse.multi_agent.multi_agent import MultiAgentChat

story_prompts_path = 'data/coverse_pe/story_prompt.txt'
# MODEL_NAME = 'qwen3:4b-instruct'
MODEL_NAME = 'qwen3:30b'

client = MultiAgentChat(MODEL_NAME)
SYSTEM_PROMPT = """
你是一名心理学实验中的对话参与者，需要严格遵守以下规则：  
1. 我们轮流造句，每次只说一句话。  
2. 回答长度必须在10到20字之间。  
3. 不允许使用任何标点符号。  
4. 每句话都必须有故事性和口语化风格，像一个人在自然讲故事。  
5. 只能输出回答，不允许有额外解释或格式。  
"""

story_prompts = open(story_prompts_path, 'r').readlines()
dataset = [{'first_message': p.strip()} for p in story_prompts]


def process_one_sample(example):
    first_message = example['first_message']
    messages = client.multi_agent_chat(system_prompt=SYSTEM_PROMPT, first_message=first_message)
    story = '\n'.join([m['content'] for m in messages if m['role'] in ['agent_1', 'agent_2']])
    example['messages'] = messages
    example['story'] = story
    return example


with ThreadPoolExecutor(max_workers=10) as executor:
    outputs = list(tqdm(executor.map(process_one_sample, dataset), total=len(dataset)))

df = pd.DataFrame(outputs)
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
output_path = f'data/coverse_pe/{MODEL_NAME}-{timestamp}.csv'.replace(':', '-')
df.to_csv(output_path, index=False)
