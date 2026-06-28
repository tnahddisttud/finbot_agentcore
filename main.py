"""Entrypoint: route a query, then answer in-scope ones with the agent.

Run:
    python main.py "Give me an analyst brief on NVDA"
    python main.py            # interactive REPL
"""
import sqlite3
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

from agent import build_agent  # noqa: E402  (import after load_dotenv)
from router import build_router  # noqa: E402


def ask(agent, router, query, thread_id="default"):
    """Route the query; refuse out-of-scope, otherwise run the agent."""
    intent = router.classify(query)
    if intent is None:
        return (
            "I'm FinBot, a financial research assistant. That request looks "
            "outside my scope (stocks, market news, general finance)."
        )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    result = agent.invoke({"messages": [("human", query)]}, config=config)
    print(f"[intent={intent}]")
    return result["messages"][-1].content


def main():
    import os

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY in your .env file first.")

    conn = sqlite3.connect("finance_agent.db", check_same_thread=False)
    agent = build_agent(checkpointer=SqliteSaver(conn))
    router = build_router()

    if len(sys.argv) > 1:
        print(ask(agent, router, " ".join(sys.argv[1:])))
        return

    print("FinBot ready. Ask a question ('exit' to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print(ask(agent, router, q, thread_id="cli"))


if __name__ == "__main__":
    main()
