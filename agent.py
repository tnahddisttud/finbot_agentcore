
import os

# Identify our HTTP requests (e.g. YahooFinanceNewsTool's WebBaseLoader). Set
# before importing langchain_community so its USER_AGENT check finds a value.
# setdefault: a real value from .env (loaded by the entry point) wins.
os.environ.setdefault("USER_AGENT", "finbot-agent/1.0")

import threading

import requests

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_groq import ChatGroq

MAIN_MODEL = os.getenv("MAIN_MODEL", "openai/gpt-oss-20b")

# --- Deterministic policy (the AgentCore Policy concept, enforced at the agent
# layer). Rules run BEFORE a tool executes: a tool allow-list + a compliance
# restricted-ticker blocklist. ---
ALLOWED_TOOLS = {"get_stock_fundamentals", "yahoo_finance_news", "search_news"}
RESTRICTED_TICKERS = {
    t.strip().upper()
    for t in os.getenv("RESTRICTED_TICKERS", "TSLA").split(",")
    if t.strip()
}


class PolicyMiddleware(AgentMiddleware):
    """Deterministic tool-call guardrail.

    - Allow-list: only approved tools may run.
    - Restricted tickers: block fundamentals lookups for compliance-restricted
      symbols (demo of a regulatory restricted list).
    Denied calls return a ToolMessage instead of executing the tool.
    """

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        tool_call_id = call.get("id")

        if name not in ALLOWED_TOOLS:
            print(f"[policy] DENY tool '{name}' (not in allow-list)")
            return ToolMessage(
                content=f"Policy: the tool '{name}' is not permitted.",
                tool_call_id=tool_call_id,
            )

        ticker = str(args.get("ticker", "")).upper()
        if name == "get_stock_fundamentals" and ticker in RESTRICTED_TICKERS:
            print(f"[policy] DENY fundamentals ticker={ticker} (restricted)")
            return ToolMessage(
                content=(
                    f"Policy: {ticker} is on the restricted list; "
                    "fundamental analysis of this security is not permitted."
                ),
                tool_call_id=tool_call_id,
            )

        print(f"[policy] ALLOW tool '{name}' args={args}")
        return handler(request)
#MAIN_MODEL = os.getenv("MAIN_MODEL", "llama-3.1-8b-instant")


# Some upstream tools (e.g. YahooFinanceNewsTool's WebBaseLoader) make HTTP
# requests with no timeout and can hang indefinitely, which freezes the whole
# agent. Run them in a daemon thread with a hard wall-clock cap; on timeout we
# return a message instead of blocking. The thread is a daemon so a stalled call
# never blocks interpreter exit (the REPL stays responsive to 'exit').
def _with_timeout(fn, timeout_s, on_timeout):
    """Run fn() with a wall-clock cap; return on_timeout if it doesn't finish."""
    box = {}

    def _runner():
        box["value"] = fn()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return on_timeout
    return box.get("value", on_timeout)


@tool
def get_stock_fundamentals(ticker: str) -> dict:
    """Get current stock price, P/E ratio, market cap, and revenue growth for a ticker."""
    def _call():
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info
            return {
                "price": info.get("currentPrice"),
                "pe_ratio": info.get("trailingPE"),
                "market_cap": info.get("marketCap"),
                "revenue_growth": info.get("revenueGrowth"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception as e:
            return {"error": f"Could not fetch fundamentals for {ticker}: {e}"}

    return _with_timeout(
        _call, 15, {"error": f"Fundamentals lookup timed out for {ticker}."}
    )


@tool
def search_news(query: str) -> str:
    """Search last-24h Google News via SerpAPI. Returns news results with URLs."""
    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "tbm": "nws",          # Google News
                "tbs": "qdr:d",        # past 24 hours
                "api_key": os.getenv("SERPAPI_API_KEY"),
            },
            timeout=10,
        )
        data = resp.json()
        results = data.get("news_results", [])
        if not results:
            return "No recent news found."
        return "\n\n".join(
            f"{r.get('title')} — {r.get('link')}" for r in results[:5]
        )
    except Exception as e:
        return f"Search failed: {e}"


@tool
def yahoo_finance_news(ticker: str) -> str:
    """Get recent Yahoo Finance news headlines for a stock ticker."""
    def _call():
        try:
            return YahooFinanceNewsTool().run(ticker)
        except Exception as e:
            return f"Yahoo Finance news failed: {e}"

    return _with_timeout(
        _call, 15, f"Yahoo Finance news timed out for {ticker}."
    )


SYSTEM_PROMPT = """\
You are FinBot, an expert equity research analyst with deep knowledge of financial markets,
valuation methodologies, and macroeconomic trends.

## Task
Given a stock ticker or company name, produce a concise, structured analyst brief that helps
users evaluate the investment. Do not give buy/sell advice. Present data-driven signals only.

## Scope
You ONLY do equity/finance research (specific stocks, market news, finance concepts).
If the request is anything else — meal/diet plans, coding, travel, general chit-chat,
or any non-finance task, even if it mentions money or a "budget" — do NOT attempt it.
Reply with exactly: "I can only help with stock, market, and finance questions."

EXCEPTION — questions about you and what you do are always in scope. If the user
asks who/what you are, what you can do, your scope, or how to use you (e.g.
"Who are you?", "What can you help with?"), briefly introduce yourself as FinBot,
an equity-research assistant, and describe your capabilities. Do NOT refuse these.

## Rules
1. Gather data before analysis. Never rely on memory for numbers.
2. If a tool fails or returns empty data, state it and proceed.
3. Never fabricate prices, ratios, or news.
4. Always follow the output format.
5. Flag notable risks or red flags.

## Output Format

**[TICKER] — Analyst Brief**
- 📊 **Fundamentals:** price, P/E, market cap, revenue growth (one line)
- 📈 **Valuation Signal:** OVERVALUED / FAIRLY VALUED / UNDERVALUED + reason
- 📰 **News Sentiment:** bullish / neutral / bearish + key headline
- ⚠️ **Key Risks:** 1–2 bullets
- 🧭 **Outlook:** 1–2 sentence synthesis, no advice
"""


def build_agent(checkpointer=None, store=None, middleware=None):
    """Build the FinBot agent.

    Args:
        checkpointer: short-term (conversation/thread) memory backend.
        store:        long-term memory store (e.g. AgentCoreMemoryStore).
        middleware:   list of agent middleware (e.g. a memory middleware that
                      saves/recalls long-term memories).
    """
    params = dict(
        model=MAIN_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=2,
    )
    # reasoning_format is only supported by reasoning models (e.g. gpt-oss).
    if "gpt-oss" in MAIN_MODEL:
        params["reasoning_format"] = "parsed"
    llm = ChatGroq(**params)
    tools = [
        get_stock_fundamentals,
        yahoo_finance_news,
        search_news,
    ]
    # PolicyMiddleware is always on; any caller-supplied middleware stacks on top.
    mw = [PolicyMiddleware(), *(middleware or [])]
    agent_kwargs = dict(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=mw,
    )
    if store is not None:
        agent_kwargs["store"] = store
    return create_agent(**agent_kwargs)
