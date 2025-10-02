import os
import re

import dotenv
import openai

dotenv.load_dotenv()


class ModelClient:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name.startswith('qwen'):
            client = self.load_ollama_client()
        elif model_name in {'doubao-seed-1-6-250615'}:
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

    def chat(
            self,
            messages,
            temperature: float = 0.7,
            max_tokens: int = 1024
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content
        # remove think
        answer = re.sub('<think>.+</think>', '', answer, flags=re.MULTILINE)
        answer = answer.strip()
        return answer
