from abc import ABC, abstractmethod

from models import AgentState


class Agent(ABC):
    """An abstract base class for agents."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute_turn(
        self,
        agent_state: AgentState,
        attempt: int = 0,
        notes: str | None = None,
    ) -> AgentState:
        """
        Execute a single turn for the agent.
        """
        pass
