from pydantic import BaseModel, Field

from .base import Agent
from llm import call_llm
from models import AgentState


class StepReflectorLlmResponse(BaseModel):
    """
    A response from the StepReflectorAgent.
    """

    success: bool = Field(
        description="True iff the reflector determined that the step was executed successfully."
    )
    notes: str = Field(
        description="Notes about the step execution attempt. If unsuccessful, this must include the reasons for the rejection and suggestions for how to resolve the issue when the ExecutorAgent retries the step."
    )


class StepReflectorAgent(Agent):
    """
    A step reflector agent that judges whether the current step is complete.
    """

    def __init__(self, user_query: str):
        super().__init__("StepReflectorAgent", "Evaluates step completion quality")
        self.user_query = user_query

    async def execute_turn(
        self,
        agent_state: AgentState,
        attempt: int = 0,
        notes: str | None = None,
    ) -> AgentState:
        """
        Examine the response document and determine whether the current step
        is complete. If so, increment the current step index in the AgentState.
        """
        current_step = agent_state.plan.steps[agent_state.plan.current_step_index]
        latest_attempt = agent_state.history[-1] if agent_state.history else None

        print(
            f"🔄 Reflecting on step {current_step.step_number} ({current_step.step_description}). Attempt {attempt}."
        )

        if not latest_attempt:
            return agent_state

        system_prompt = """
You are a quality assurance agent that evaluates whether execution steps have been completed successfully.

Your job is to:
1. Examine the USER_QUERY, the FULL_PLAN, the CURRENT_STEP, and the EXECUTOR_RESULT.
2. Determine if the EXECUTOR_RESULT has successfully completed the CURRENT_STEP
3. Provide detailed feedback

A step is successful if:
- Relevant information was found and incorporated
- The response document was meaningfully updated
- The step objective was achieved

If unsuccessful, provide specific guidance for improvement.
"""

        user_prompt = f"""
<USER_QUERY>
{self.user_query}
</USER_QUERY>

<FULL_PLAN>
{agent_state.plan.model_dump_json(indent=2)}
</FULL_PLAN>

<CURRENT_STEP>
{current_step.model_dump_json(indent=2)}
</CURRENT_STEP>

<EXECUTOR_RESULT>
{latest_attempt.model_dump_json(indent=2)}
</EXECUTOR_RESULT>

Evaluate whether the EXECUTOR_RESULT has successfully completed the CURRENT_STEP,
and provide detailed feedback.
"""

        reflection_response = await call_llm(
            system_prompt, user_prompt, StepReflectorLlmResponse
        )

        # Update the latest attempt with reflection results
        latest_attempt.success = reflection_response.success
        latest_attempt.reflector_notes = reflection_response.notes

        # If successful, move to next step
        if reflection_response.success:
            print(f"Step {agent_state.plan.current_step_index} completed successfully")
            print(f"Reflector notes: {reflection_response.notes}")
            agent_state.plan.current_step_index += 1
        else:
            print(f"Step {agent_state.plan.current_step_index} failed")
            print(f"Reflector notes: {reflection_response.notes}")

        return agent_state
