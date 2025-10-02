from copy import deepcopy

from tqdm import tqdm

from coverse.utils.model_clients import ModelClient


class MultiAgentChat:
    def __init__(self, model_name):
        self.client = ModelClient(model_name)

    def chat_with_agent(self, multi_agent_messages, agent_name):
        # change role to agent_name
        messages = deepcopy(multi_agent_messages)
        for msg in messages:
            if msg['role'] == agent_name:
                msg['role'] = 'assistant'
            elif msg['role'] == 'system':
                continue
            else:
                msg['role'] = 'user'
        # chat
        answer = self.client.chat(messages)
        return answer

    def multi_agent_chat(self, system_prompt='', first_message: str = '', n_turns: int = 5, verbose: bool = False):
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.append({'role': 'agent_1', 'content': first_message})
        for i in tqdm(range(n_turns), disable=not verbose):
            answer = self.chat_with_agent(multi_agent_messages=messages, agent_name='agent_1')
            messages.append({'role': 'agent_1', 'content': answer})
            answer = self.chat_with_agent(multi_agent_messages=messages, agent_name='agent_2')
            messages.append({'role': 'agent_2', 'content': answer})
        return messages


if __name__ == '__main__':
    MODEL_NAME = 'qwen3:4b-instruct'
    client = MultiAgentChat(MODEL_NAME)
    system_prompt = """
你是一名心理学实验中的对话参与者，需要严格遵守以下规则：  
1. 我们轮流造句，每次只说一句话。  
2. 回答长度必须在10到20字之间。  
3. 不允许使用任何标点符号和换行。  
4. 每句话都必须有故事性和口语化风格，像一个人在自然讲故事。  
5. 只能输出回答，不允许有额外解释或格式。  
    """
    first_message = "小猫在阳光下睡觉"

    messages = client.multi_agent_chat(
        system_prompt=system_prompt,
        first_message=first_message,
        n_turns=5,
        verbose=True
    )
    for m in messages:
        print(m['content'])
