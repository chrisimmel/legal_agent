import asyncio

from agents.planner import PlannerAgent
from agents.step_executor import StepExecutorAgent
from agents.step_reflector import StepReflectorAgent
from models import AgentState, MAX_STEP_ATTEMPTS, Plan, UserResponseDocument
from storage.document_manager import DocumentManager
from storage.vector_store_manager import VectorStoreManager


class StepAttemptLimitExceeded(Exception):
    """
    An exception raised when the maximum number of step attempts has been exceeded.
    """

    pass


class AgentOrchestrator:
    """Orchestrates the agents to create and execute a plan in order to answer a user query."""

    def execute(self, user_query: str) -> UserResponseDocument:
        """
        Create and execute the plan.

        To begin processing:
        1. Load documents
        2. Create vector store with the chunked documents
        3. Execute a turn of the PlannerAgent to generate a plan of steps to take to answer the user query.
        4. Begin executing the plan.

        To execute the plan:
        1. Construct an initial empty UserResponseDocument.
        2. Execute each step in the sequence until plan.is_complete is True.

        To iterate through the plan:
        1. Check if the plan is complete. If so, return the UserResponseDocument.
        2. Run a turn of the StepExecutorAgent to execute the current step.
        3. Run a turn of the StepReflectorAgent to judge whether the step is
        complete. (If it is, the StepReflectorAgent will have incremented
        the current step index in the AgentState.)
        4. Repeat from step 1.
        """
        return asyncio.run(self._execute_async(user_query))

    async def _execute_async(self, user_query: str) -> UserResponseDocument:
        """
        Async implementation of execute method.
        """
        print(f"🚀 Starting execution for query: {user_query[:100]}...")

        # Step 1: Load documents and create vector store
        print("📚 Loading documents...")
        doc_manager = DocumentManager("data/documents")
        documents = doc_manager.get_all_documents()

        if not documents:
            return UserResponseDocument(content="Error: No documents found to search.")

        print(f"✓ Loaded {len(documents)} documents")

        print("🔍 Initializing vector store...")
        vector_manager = VectorStoreManager()

        # Check if vector store is empty and add documents if needed
        info = vector_manager.get_collection_info()
        if info["total_chunks"] == 0:
            print("📊 Adding documents to vector store...")
            vector_manager.add_documents(documents)
        else:
            print(f"✓ Vector store already contains {info['total_chunks']} chunks")

        # Step 2: Initialize agents
        planner = PlannerAgent(user_query)
        executor = StepExecutorAgent(user_query, vector_manager)
        reflector = StepReflectorAgent(user_query)

        # Step 3: Create initial agent state
        agent_state = AgentState(
            plan=Plan(), response_document=UserResponseDocument(content="")
        )

        # Step 4: Generate plan
        print("📋 Generating execution plan...")
        agent_state = await planner.execute_turn(agent_state)
        print(f"✓ Created plan with {len(agent_state.plan.steps)} steps:")
        for i, step in enumerate(agent_state.plan.steps):
            print(f"  {i+1}. {step.step_description}")

        # Step 5: Execute plan
        print("\n⚡ Executing plan...")
        step_attempts = {}
        latest_attempt = None

        while not agent_state.plan.is_complete:
            current_step_idx = agent_state.plan.current_step_index
            current_step = agent_state.plan.steps[current_step_idx]

            # Track attempts for this step
            if current_step_idx not in step_attempts:
                step_attempts[current_step_idx] = 0

            step_attempts[current_step_idx] += 1

            # Check if we've exceeded max attempts
            if step_attempts[current_step_idx] > MAX_STEP_ATTEMPTS:
                error_msg = f"Step {current_step_idx + 1} exceeded maximum attempts ({MAX_STEP_ATTEMPTS})"
                print(f"❌ {error_msg}")
                raise StepAttemptLimitExceeded(error_msg)

            print(
                f"\n🔄 Executing step {current_step_idx + 1}/{len(agent_state.plan.steps)}: {current_step.step_description}"
            )
            print(f"   (Attempt {step_attempts[current_step_idx]}/{MAX_STEP_ATTEMPTS})")

            # Execute step
            agent_state = await executor.execute_turn(
                agent_state=agent_state,
                attempt=step_attempts[current_step_idx],
                notes=latest_attempt.reflector_notes if latest_attempt else None,
            )

            # Reflect on step
            agent_state = await reflector.execute_turn(
                agent_state=agent_state,
                attempt=step_attempts[current_step_idx],
                notes=latest_attempt.executor_notes if latest_attempt else None,
            )

            # Check if step was successful
            latest_attempt = agent_state.current_step_latest_attempt
            if latest_attempt and not latest_attempt.success:
                print(f"   ⚠️  Step needs retry: {latest_attempt.reflector_notes}")
            else:
                print("   ✓ Step completed successfully")

        print("\n🎉 Plan execution completed!")
        return agent_state.response_document


def main():
    """Execute the AgentOrchestrator with a user query from command line."""
    import sys

    if len(sys.argv) < 2:
        print("=== Agent Orchestrator ===\n")
        print('Usage: python agents.py "<your query>"')
        print("\nExample queries:")
        print(
            '  python agents.py "What are the main legal issues discussed in the case documents?"'
        )
        print(
            '  python agents.py "Summarize the key findings and outcomes from all cases."'
        )
        print(
            '  python agents.py "What types of legal proceedings are represented in the database?"'
        )
        sys.exit(1)

    # Get user query from command line argument
    user_query = " ".join(sys.argv[1:])
    print(f"=== Agent Orchestrator ===\n")
    print(f"Query: {user_query}\n")

    try:
        orchestrator = AgentOrchestrator()
        result = orchestrator.execute(user_query)

        print("\n" + "=" * 60)
        print("FINAL RESPONSE:")
        print("=" * 60)
        print(result.content)
        print("=" * 60)

    except StepAttemptLimitExceeded as e:
        print(f"\n❌ Execution failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
