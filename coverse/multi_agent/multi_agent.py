from tqdm import tqdm

from coverse.utils.model_clients import ModelClient


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
    # chat
    answer = client.chat(messages)
    return answer


def multi_agent(system_prompt='', first_message: str = ''):
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.append({'role': 'agent_1', 'content': first_message})
    for i in tqdm(range(5)):
        answer = chat_with_agent(messages=messages, agent_name='agent_1')
        messages.append({'role': 'agent_1', 'content': answer})
        answer = chat_with_agent(messages=messages, agent_name='agent_2')
        messages.append({'role': 'agent_2', 'content': answer})
    return messages


if __name__ == '__main__':
    MODEL_NAME = 'qwen3:4b-instruct'
    client = ModelClient(MODEL_NAME)
    system_prompt = """
你是一名心理学实验中的对话参与者，需要严格遵守以下规则：  
1. 我们轮流造句，每次只说一句话。  
2. 回答长度必须在10到20字之间。  
3. 不允许使用任何标点符号。  
4. 每句话都必须有故事性和口语化风格，像一个人在自然讲故事。  
5. 只能输出回答，不允许有额外解释或格式。  
    """
    first_message = "小猫在阳光下睡觉"

    messages = multi_agent(system_prompt=system_prompt, first_message=first_message)
    for m in messages:
        print(m['content'])
