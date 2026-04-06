from __future__ import annotations

from coverse.core.agents.conversation import ConversationAgent
from coverse.core.types import ConversationMessage


class MultiAgentRunner:
    def __init__(self, agents: list[ConversationAgent]):
        if not agents:
            raise ValueError("MultiAgentRunner requires at least one agent.")
        self.agents = agents

    def build_agent_view(
        self, transcript: list[ConversationMessage], agent_name: str
    ) -> list[ConversationMessage]:
        visible_messages: list[ConversationMessage] = []
        for message in transcript:
            if message.role == "system":
                visible_messages.append(message)
            elif message.role == agent_name:
                visible_messages.append(
                    ConversationMessage(role="assistant", content=message.content)
                )
            else:
                visible_messages.append(ConversationMessage(role="user", content=message.content))
        return visible_messages

    def run(
        self,
        *,
        first_message: str,
        n_turns: int,
    ) -> list[dict[str, str]]:
        if not first_message.strip():
            raise ValueError("first_message must not be empty.")
        if n_turns <= 0:
            raise ValueError("n_turns must be greater than zero.")

        transcript = [ConversationMessage(role="user", content=first_message.strip())]
        for _ in range(n_turns):
            for agent in self.agents:
                answer = agent.respond(self.build_agent_view(transcript, agent.config.name))
                transcript.append(ConversationMessage(role=agent.config.name, content=answer))
        return [message.to_dict() for message in transcript]
