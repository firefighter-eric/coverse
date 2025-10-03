import re

from coverse.utils.model_clients import ModelClient


class ConverseAgent:
    def __init__(
            self,
            agent_name='agent',
            model_name='qwen3:4b-instruct',
            system_prompt=None,
    ):
        self.agent_name = agent_name
        self.model_name = model_name
        self.system_prompt = system_prompt

        self.model_client = ModelClient(model_name)
        if self.system_prompt is None:
            self.system_prompt = self.default_system_prompt()

    def run(self, messages):
        if messages[0]['role'] != 'system':
            messages = [{'role': 'system', 'content': self.system_prompt}] + messages
        answer = self.model_client.generate(messages)
        answer = self.postprocess(answer)
        return answer

    def default_system_prompt(self):
        return """
你是一名心理学实验中的对话参与者，需要严格遵守以下规则：  
1. 我们轮流造句，每次只说一句话。  
2. 回答长度必须在10到20字之间。  
3. 不允许使用任何标点符号和换行。  
4. 每句话都必须有故事性和口语化风格，像一个人在自然讲故事。  
5. 只能输出回答，不允许有额外解释或格式。  
""".strip()

    def postprocess(self, answer):
        # remove think
        answer = re.sub('<think>.+</think>', '', answer, flags=re.MULTILINE)
        answer = answer.strip()
        return answer


if __name__ == '__main__':
    agent = ConverseAgent()
    messages = [
        {'role': 'user', 'content': '小猫在阳光下睡觉'},
    ]

    answer = agent.run(messages)
    print(f'Agent: {answer}')
