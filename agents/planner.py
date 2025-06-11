from pydantic import BaseModel, Field

from .base import Agent
from llm import call_llm
from models import AgentState, Plan


class PlannerLlmResponse(BaseModel):
    """
    A response from the PlannerAgent.
    """

    plan: Plan = Field(
        description="The plan of steps to take to answer the user query."
    )


class PlannerAgent(Agent):
    """
    A planner agent that plans the steps to take to answer the user query.
    """

    def __init__(self, user_query: str):
        super().__init__("PlannerAgent", "Plans the steps to answer the user query")
        self.user_query = user_query

    async def execute_turn(
        self,
        agent_state: AgentState,
        attempt: int = 0,
        notes: str | None = None,
    ) -> AgentState:
        """
        Execute a single turn for the planner agent. (For this agent, only one turn will be executed.)

        The purpose of this agent is to generate a plan of steps to take to answer the user query.

        The context must include:
        - The original user query
        - Instructions to the LLM about how to generate the plan.
        """

        # This agent ignores attempt and notes.

        system_prompt = """
You are a planning agent that creates step-by-step plans to answer user queries about legal case documents.

Your job is to analyze the user's query and create a logical sequence of steps that will lead to a comprehensive answer.

Each step will include a research phase, so you don't need a separate step just for that.

An initial phase might be to do a high-level review of the initial document set.

Subsequent steps might include:
- Deeper research and analysis of specific things seen in the initial research
- Synthesizing information across multiple documents
- Formatting the final response

Each step should be concrete and actionable.
"""

        user_prompt = f"""
<USER_QUERY>
{self.user_query}
</USER_QUERY>

Create a plan with 1-4 steps to comprehensively answer this query.

Keep in mind:
- Each step will perform its own document research and update the plan state.
- The plan should progressively enhance and refine the response document.
- A simple query should produce a simple plan with few steps, but a complex query should produce a more complex plan.
- The final step needs to produce an accurate, detailed, well-formatted response to the user query.
"""

        response = await call_llm(system_prompt, user_prompt, PlannerLlmResponse)
        agent_state.plan = response.plan
        return agent_state
