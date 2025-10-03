import os

import dotenv
import openai
from loguru import logger

dotenv.load_dotenv()


class ModelClient:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name.startswith('qwen'):
            client = self.load_ollama_client()
        elif model_name.startswith('doubao'):
            client = self.load_ark_client()
        elif model_name.startswith('deepseek'):
            client = self.load_ark_client()
        else:
            raise ValueError(f"Model {model_name} not supported")
        self.client = client

    @staticmethod
    def load_ollama_client():
        client = openai.OpenAI(
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        )
        return client

    @staticmethod
    def load_ark_client():
        client = openai.OpenAI(
            api_key=os.environ["ARK_API_KEY"],
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        return client

    def generate(
            self,
            messages,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 1024
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content
        return answer


def test_doubao():
    model_name = 'doubao-seed-1-6-250615'
    client = ModelClient(model_name=model_name)
    messages = [{"role": "user", "content": "who are you?"}]
    answer = client.generate(messages)
    logger.info(f'{model_name=}\nanswer\n{answer}\n' + '-' * 20)


def test_deepseek():
    model_name = 'deepseek-v3-1-terminus'
    client = ModelClient(model_name=model_name)
    messages = [{"role": "user", "content": "who are you?"}]
    answer = client.generate(messages)
    logger.info(f'{model_name=}\nanswer\n{answer}\n' + '-' * 20)


if __name__ == '__main__':
    test_doubao()
    test_deepseek()
