from __future__ import annotations

import unittest

from coverse.core.agents import AgentConfig, ConversationAgent, GenerationConfig, MultiAgentRunner
from coverse.core.types import ConversationMessage


class StubBackend:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages, *, temperature, top_p, max_tokens):
        self.calls.append(messages)
        return self.responses.pop(0)


class RecordingAgent:
    def __init__(self, name: str, answer: str):
        self.config = type("Config", (), {"name": name})()
        self.answer = answer
        self.seen_messages = []

    def respond(self, messages):
        self.seen_messages.append(messages)
        return self.answer


class ConversationAgentTests(unittest.TestCase):
    def test_respond_injects_system_prompt_and_postprocesses(self):
        backend = StubBackend(["<think>hidden</think>\n  usable answer  "])
        agent = ConversationAgent(
            AgentConfig(
                name="agent_1",
                system_prompt="system prompt",
                generation=GenerationConfig(),
            ),
            backend,
        )

        answer = agent.respond([ConversationMessage(role="user", content="hello")])

        self.assertEqual(answer, "usable answer")
        self.assertEqual(backend.calls[0][0], {"role": "system", "content": "system prompt"})

    def test_empty_answer_raises(self):
        backend = StubBackend(["  "])
        agent = ConversationAgent(
            AgentConfig(name="agent_1", system_prompt="system prompt"),
            backend,
        )

        with self.assertRaises(ValueError):
            agent.respond([{"role": "user", "content": "hello"}])


class MultiAgentRunnerTests(unittest.TestCase):
    def test_multi_agent_runner_preserves_turn_order_and_view_mapping(self):
        agent_1 = RecordingAgent("agent_1", "first reply")
        agent_2 = RecordingAgent("agent_2", "second reply")
        runner = MultiAgentRunner([agent_1, agent_2])

        transcript = runner.run(first_message="opening line", n_turns=1)

        self.assertEqual(
            transcript,
            [
                {"role": "user", "content": "opening line"},
                {"role": "agent_1", "content": "first reply"},
                {"role": "agent_2", "content": "second reply"},
            ],
        )
        self.assertEqual(
            [(message.role, message.content) for message in agent_1.seen_messages[0]],
            [("user", "opening line")],
        )
        self.assertEqual(
            [(message.role, message.content) for message in agent_2.seen_messages[0]],
            [("user", "opening line"), ("user", "first reply")],
        )


if __name__ == "__main__":
    unittest.main()
