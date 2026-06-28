
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

GUARD_MODEL = os.getenv("GUARD_MODEL", "llama-3.1-8b-instant")

INTENTS = ("stock_analysis", "market_news", "general_finance", "meta")

ROUTER_PROMPT = """You are an intent classifier for a finance research assistant.
Classify the user's message into exactly one of these labels:

- "stock_analysis": about a specific company/stock's fundamentals, valuation,
  price, P/E, market cap, EPS, revenue.
- "market_news": about recent market/finance news, events, or announcements.
- "general_finance": general finance concepts or education (what is X, how does Y work).
- "meta": about the assistant itself — who/what it is, what it can do, its scope,
  or how to use it (e.g. "who are you?", "what can you help with?", "hi").
- "none": anything NOT about stocks, markets, or finance — e.g. diet, coding,
  weather, travel, chit-chat — even if it mentions money or a "budget".

Reply with ONLY a JSON object, no other text:
{"intent": "stock_analysis" | "market_news" | "general_finance" | "meta" | "none"}"""


class IntentRouter:
    """Classifies a query into a finance intent, or None if out of scope."""

    def __init__(self, llm: ChatGroq):
        self._llm = llm

    def classify(self, query: str):
        """Return one of INTENTS, or None for out-of-scope / parse failure."""
        try:
            resp = self._llm.invoke(
                [
                    SystemMessage(content=ROUTER_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            data = json.loads(resp.content)
            intent = data.get("intent")
            return intent if intent in INTENTS else None
        except (json.JSONDecodeError, AttributeError, TypeError):
            # Malformed classifier output -> fail closed (treat as out of scope).
            return None


def build_router() -> IntentRouter:
    """Build the intent router backed by a cheap, fast Groq model."""
    llm = ChatGroq(
        model=GUARD_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=64,
        max_retries=2,
    )
    return IntentRouter(llm)
