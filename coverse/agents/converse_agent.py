import re

from coverse.utils.model_clients import ModelClient


class ConverseAgent:
    def __init__(
            self,
            agent_name='agent',
            model_name='qwen3:4b-instruct',
            system_prompt=None,
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
    ):
        self.agent_name = agent_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.model_client = ModelClient(model_name)
        if self.system_prompt is None:
            self.system_prompt = self.default_system_prompt()

    def run(self, messages):
        if messages[0]['role'] != 'system':
            messages = [{'role': 'system', 'content': self.system_prompt}] + messages
        answer = self.model_client.generate(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        answer = self.postprocess(answer)
        return answer

    def default_system_prompt(self):
        return """
# 角色
你是一名心理学实验中的对话参与者，需要与另一位参与者合作完成造句游戏。  

# 规则
- 我们轮流造句，每次只说一句话  
- 造句的目的是一起创作一个故事  
- 回答长度必须在7到20字之间  
- 不允许使用任何标点符号和换行符号  
- 每句话必须有故事性并且口语化自然  
- 只能输出回答，不允许有额外解释或格式  

# 格式
只输出一句话的故事内容  
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
