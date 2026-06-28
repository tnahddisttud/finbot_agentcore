
import os
import uuid

from dotenv import load_dotenv
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

load_dotenv()

from langchain.agents.middleware import AgentMiddleware  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.config import get_config  # noqa: E402

from agent import build_agent  # noqa: E402
from router import build_router  # noqa: E402

REGION = os.getenv("AWS_REGION", "ap-south-1")
MEMORY_ID = os.getenv("MEMORY_ID")

app = BedrockAgentCoreApp()

REFUSAL = (
    "I'm FinBot, a financial research assistant. That request looks outside my "
    "scope (stocks, market news, general finance)."
)


class MemoryMiddleware(AgentMiddleware):
    """Saves the latest user/AI messages to long-term memory and recalls
    relevant past memories for the user, injecting them into context.

    Hook signatures mirror the crash course's 02_agentcore_memory.py:
    langgraph injects the `store` (because we pass store= to create_agent).
    All store operations are wrapped in try/except so a memory hiccup (or a
    memory resource without a long-term strategy) never breaks the agent.
    """

    @staticmethod
    def _actor_id() -> str:
        # runtime has no .config; read the invoke config via langgraph.
        try:
            return get_config().get("configurable", {}).get("actor_id", "default-user")
        except Exception:  # noqa: BLE001
            return "default-user"

    @staticmethod
    def _namespace(actor_id: str):
        # Long-term memories are scoped per user (actor), shared across threads.
        return ("memories", actor_id)

    def before_model(self, state, runtime):
        store = getattr(runtime, "store", None)
        if store is None:
            print("[memory] before_model: no store on runtime")
            return None
        actor_id = self._actor_id()
        messages = state.get("messages", [])
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if last_human is None:
            return None

        ns = self._namespace(actor_id)
        try:
            hits = store.search(ns, query=last_human.content, limit=5)
            print(f"[memory] recall actor={actor_id} hits={len(hits or [])}")
            lines = []
            for h in hits or []:
                msg = (h.value or {}).get("message")
                content = getattr(msg, "content", None)
                if content:
                    lines.append(f"- {content}")
            recalled = "\n".join(lines)
            if recalled:
                print(f"[memory] recalled: {recalled[:200]}")
                # add_messages appends; the model sees this note alongside the turn.
                return {
                    "messages": [
                        SystemMessage(
                            content=f"Relevant things you remember about this user:\n{recalled}"
                        )
                    ]
                }
        except Exception as e:  # noqa: BLE001
            print(f"[memory] recall error: {e}")
        return None

    def after_model(self, state, runtime):
        store = getattr(runtime, "store", None)
        if store is None:
            return None
        actor_id = self._actor_id()
        ns = self._namespace(actor_id)
        messages = state.get("messages", [])
        try:
            for m in reversed(messages):
                if isinstance(m, (HumanMessage, AIMessage)) and m.content:
                    # AgentCoreMemoryStore requires {"message": <BaseMessage>}.
                    store.put(ns, str(uuid.uuid4()), {"message": m})
                    print(f"[memory] saved actor={actor_id}: {str(m.content)[:80]}")
                    break
        except Exception as e:  # noqa: BLE001
            print(f"[memory] save error: {e}")
        return None


# Lazily built on first invoke (NOT at import) so container startup stays under
_cache = {}


def _get_agent_and_router():
    if "agent" not in _cache:
        checkpointer = (
            AgentCoreMemorySaver(MEMORY_ID, region_name=REGION) if MEMORY_ID else None
        )
        store = (
            AgentCoreMemoryStore(memory_id=MEMORY_ID, region_name=REGION)
            if MEMORY_ID
            else None
        )
        middleware = [MemoryMiddleware()] if MEMORY_ID else None
        _cache["agent"] = build_agent(
            checkpointer=checkpointer, store=store, middleware=middleware
        )
        _cache["router"] = build_router()
    return _cache["agent"], _cache["router"]


@app.entrypoint
def agent_invocation(payload, context):
    """Handler with short-term + long-term memory (per actor_id / thread_id)."""
    print("Received payload:", payload)

    agent, router = _get_agent_and_router()
    query = payload.get("prompt", "No prompt found in input")
    actor_id = payload.get("actor_id", "default-user")
    thread_id = payload.get("thread_id", payload.get("session_id", "default-session"))

    intent = router.classify(query)
    if intent is None:
        return {"result": REFUSAL, "intent": None, "actor_id": actor_id, "thread_id": thread_id}

    config = {
        "configurable": {"thread_id": thread_id, "actor_id": actor_id},
        # Memory middleware adds recall/save super-steps around every model
        # turn, so a multi-tool brief (fundamentals + 2 news tools + synthesis)
        # overruns the default 15. Give it headroom while still bounding loops.
        "recursion_limit": 30,
    }
    result = agent.invoke({"messages": [("human", query)]}, config=config)
    return {
        "result": result["messages"][-1].content,
        "intent": intent,
        "actor_id": actor_id,
        "thread_id": thread_id,
    }


if __name__ == "__main__":
    app.run()
