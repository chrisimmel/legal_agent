from typing import TypeVar

from dotenv import load_dotenv
import instructor
from pydantic import BaseModel

load_dotenv()


MODEL_ID = "openai/gpt-4.1"
# MODEL_ID = "anthropic/claude-sonnet-4-0"


# Using the same model for all agents for now.
client = instructor.from_provider(MODEL_ID, async_client=True)


# Generic type for response models
T = TypeVar("T", bound=BaseModel)


async def call_llm(system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
    """
    Call an LLM with a parameterized response model type.

    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        response_model: The Pydantic model class to use for response
                       validation

    Returns:
        An instance of the specified response model type
    """
    print(f"Making LLM call with model {MODEL_ID}")

    return await client.chat.completions.create(
        response_model=response_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
    )
