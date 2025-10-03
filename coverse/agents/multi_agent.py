from copy import deepcopy

from loguru import logger
from tqdm import tqdm

from coverse.agents.converse_agent import ConverseAgent


class MultiAgentChat:
    def __init__(self, agents: list[ConverseAgent]):
        self.agents = agents
        logger.info(f'Loaded {len(agents)} agents')
        for i, agent in enumerate(agents):
            logger.info(f'agent {i+1}: {agent.agent_name}')

    def chat_with_agent(self, multi_agent_messages, agent):
        # change role to agent_name
        messages = deepcopy(multi_agent_messages)
        for msg in messages:
            if msg['role'] == agent.agent_name:
                msg['role'] = 'assistant'
            else:
                msg['role'] = 'user'
        # chat
        answer = agent.run(messages)
        return answer

    def run(self, first_message: str = '', n_turns: int = 5, verbose: bool = False):
        messages = [{'role': 'user', 'content': first_message}]
        for i in tqdm(range(n_turns), disable=not verbose):
            for agent in self.agents:
                answer = self.chat_with_agent(multi_agent_messages=messages, agent=agent)
                messages.append({'role': agent.agent_name, 'content': answer})
        return messages


if __name__ == '__main__':
    MODEL_NAME = 'qwen3:4b-instruct'
    agent_1 = ConverseAgent(agent_name='agent_1', model_name=MODEL_NAME)
    agent_2 = ConverseAgent(agent_name='agent_2', model_name=MODEL_NAME)
    agent_3 = ConverseAgent(agent_name='agent_3', model_name=MODEL_NAME)
    client = MultiAgentChat([agent_1, agent_2, agent_3])
    first_message = "小猫在阳光下睡觉"

    messages = client.run(first_message=first_message, n_turns=5, verbose=True)
    for m in messages:
        print(f'{m["role"]}: {m["content"]}')
