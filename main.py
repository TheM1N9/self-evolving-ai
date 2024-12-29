from dotenv import load_dotenv
import os
from orchestrator import AgentOrchestrator
import asyncio


async def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    # Create orchestrator
    orchestrator = AgentOrchestrator(api_key)
    # Create all required agents
    orchestrator.create_agent("goal_generator", agent_type="screen")
    orchestrator.create_agent("goal_executor", agent_type="screen")
    orchestrator.create_agent("verifier", agent_type="verifier")
    try:
        # Run the orchestrated agent loop
        await orchestrator.run_agent_loop()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
