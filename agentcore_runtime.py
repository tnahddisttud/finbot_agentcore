"""Bedrock AgentCore runtime wrapper for FinBot.

"""
from dotenv import load_dotenv
from bedrock_agentcore.runtime import BedrockAgentCoreApp

load_dotenv()

from agent import build_agent  # noqa: E402  (import after load_dotenv)
from router import build_router  # noqa: E402

app = BedrockAgentCoreApp()

# Build once at startup (stateless — no checkpointer, like 01_agentcore_runtime.py)
agent = build_agent()
router = build_router()

REFUSAL = (
    "I'm FinBot, a financial research assistant. That request looks outside my "
    "scope (stocks, market news, general finance)."
)


@app.entrypoint
def agent_invocation(payload: str, context):
    """Handler for agent invocation in the AgentCore runtime."""
    print("Received payload:", payload)

    query = payload.get("prompt", "No prompt found in input")

    intent = router.classify(query)
    if intent is None:
        return {"result": REFUSAL, "intent": None}

    result = agent.invoke(
        {"messages": [("human", query)]},
        config={"recursion_limit": 15},
    )
    return {"result": result["messages"][-1].content, "intent": intent}


if __name__ == "__main__":
    app.run()
