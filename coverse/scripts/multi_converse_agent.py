from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from coverse.agents.converse_agent import ConverseAgent
from coverse.agents.multi_agent import MultiAgentChat

# %% config
story_prompts_path = 'data/coverse_pe/story_prompt.txt'

# MODEL_NAME = 'qwen3:4b-instruct'
# MODEL_NAME = 'qwen3:30b'
# MODEL_NAME = 'doubao-seed-1-6-250615'
MODEL_NAME = 'deepseek-v3-1-terminus'
tag = 'temp0.7'

# %%
agent_1 = ConverseAgent(agent_name='agent_1', model_name=MODEL_NAME, temperature=0.7)
agent_2 = ConverseAgent(agent_name='agent_2', model_name=MODEL_NAME, temperature=0.7)
client = MultiAgentChat(agents=[agent_1, agent_2])

story_prompts = open(story_prompts_path, 'r').readlines()
dataset = [{'first_message': p.strip()} for p in story_prompts]


def process_one_sample(example):
    first_message = example['first_message']
    messages = client.run(first_message=first_message, n_turns=5, verbose=False)
    story = '\n'.join([m['content'] for m in messages if m['role'] in ['agent_1', 'agent_2']])
    example['messages'] = messages
    example['story'] = story
    return example


with ThreadPoolExecutor(max_workers=10) as executor:
    outputs = list(tqdm(executor.map(process_one_sample, dataset), total=len(dataset)))

df = pd.DataFrame(outputs)
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
output_path = f'data/coverse_pe/{MODEL_NAME}-{timestamp}-{tag}.csv'.replace(':', '-')
df.to_csv(output_path, index=False)
