import asyncio
import os
import sys

# This file lives in gateway/, but it reuses agent.py and router.py from the
# project root. Add the root to sys.path so those imports resolve regardless of
# where the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # noqa: E402  (import after load_dotenv)
from langchain_groq import ChatGroq  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402

# Reuse the existing tools + prompt unchanged — no edits to agent.py.
from agent import (  # noqa: E402
    MAIN_MODEL,
    SYSTEM_PROMPT,
    get_stock_fundamentals,
    search_news,
)
from router import build_router  # noqa: E402


async def main():
    if not os.getenv("GATEWAY_URL") or not os.getenv("GATEWAY_ACCESS_TOKEN"):
        sys.exit(
            "Set GATEWAY_URL and GATEWAY_ACCESS_TOKEN in .env first "
            "(see the GATEWAY setup steps)."
        )

    # Load the Gateway's MCP tools as LangChain tools.
    client = MultiServerMCPClient(
        {
            "finbot": {
                "transport": "streamable_http",
                "url": os.environ["GATEWAY_URL"],
                "headers": {"Authorization": f"Bearer {os.environ['GATEWAY_ACCESS_TOKEN']}"},
            }
        }
    )
    gw_tools = await client.get_tools()
    print(f"[gateway] loaded tools: {[t.name for t in gw_tools]}")

    params = dict(model=MAIN_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    if "gpt-oss" in MAIN_MODEL:
        params["reasoning_format"] = "parsed"  # mirror agent.py for gpt-oss models

    agent = create_agent(
        model=ChatGroq(**params),
        tools=[get_stock_fundamentals, search_news, *gw_tools],
        system_prompt=SYSTEM_PROMPT,
    )

    query = " ".join(sys.argv[1:]) or "Give me the latest news on NVDA"
    if build_router().classify(query) is None:
        print("I can only help with stock, market, and finance questions.")
        return

    # ainvoke (not invoke) — MCP tools are async-only.
    result = await agent.ainvoke(
        {"messages": [("human", query)]}, config={"recursion_limit": 15}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
