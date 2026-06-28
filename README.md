# FinBot : Finance Research Agent

FinBot is an AI **equity-research assistant**. Ask it about a stock and it gathers
live data (fundamentals, news) and returns a concise, structured analyst brief.
A lightweight **intent router** screens every question first, so off-topic
requests are refused cheaply before the main agent ever runs.

## What it does

1. **Routes** each query into an intent — `stock_analysis`, `market_news`,
   `general_finance`, or **out-of-scope** (refused).
2. For in-scope queries, the **FinBot agent** decides which tools to call:
   - Yahoo Finance fundamentals (price, P/E, market cap, revenue growth)
   - Yahoo Finance news headlines
   - Google News (last 24h) via SerpAPI
3. **Synthesizes** a structured analyst brief (fundamentals → valuation signal →
   news sentiment → risks → outlook). It never gives buy/sell advice.
4. **Remembers** the conversation so follow-ups work ("now compare it with AMD")
    local SQLite for dev, AgentCore Memory in the cloud.

---

## Architecture

```
                ┌─────────────────────────────────────────────────────┐
   user query   │                     ask() / entrypoint               │
 ─────────────► │                                                      │
                │   1. router.classify(query)   ◄── cheap LLM call     │
                │        │                                             │
                │   out-of-scope? ──► polite refusal (agent skipped)   │
                │        │                                             │
                │   in-scope (intent)                                  │
                │        ▼                                             │
                │   2. FinBot agent (LangGraph)                        │
                │        ├── get_stock_fundamentals  (yfinance)        │
                │        ├── yahoo_finance_news       (Yahoo)          │
                │        └── search_news              (SerpAPI)        │
                │        ▼                                             │
                │   3. structured analyst brief  ──────────────────►  │ answer
                └─────────────────────────────────────────────────────┘
                          memory: SQLite (local) | AgentCore (cloud)

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Orchestration | **LangGraph** (`langchain.agents.create_agent`) | Builds the agent state machine + tool loop |
| Framework | **LangChain** (`langchain`, `langchain-community`) | Tool definitions, message types |
| LLM | **Groq** (`langchain-groq`) | `gpt-oss-20b` (agent) + `llama-3.1-8b-instant` (router) |
| Tools | **yfinance**, **SerpAPI** (via `requests`), Yahoo Finance news | Live financial data |
| Memory (local) | **langgraph-checkpoint-sqlite** | Per-thread conversation memory on disk |
| Memory (cloud) | **langgraph-checkpoint-aws** + **AgentCore Memory** | Persistent memory in the cloud |
| Deployment | **bedrock-agentcore** + **bedrock-agentcore-starter-toolkit** | Serverless deploy to AWS |
| Cloud runtime | **Amazon Bedrock AgentCore** | Hosts the agent (region: `ap-south-1`) |

> Note: AgentCore is the **hosting/runtime** — the LLM is still Groq, not a
> Bedrock model. AgentCore provides the serverless container and managed memory.

---

## Project structure — module by module

```
finance-agent/
├── agent.py                    # the agent: LLM + tools + system prompt
├── router.py                   # the intent classifier (gatekeeper)
├── main.py                     # local CLI entrypoint
├── agentcore_runtime.py        # cloud deploy — stateless
├── agentcore_memory_runtime.py # cloud deploy — with persistent memory
├── pyproject.toml              # dependencies (used by `uv`)
└── .env                        # API keys (gitignored — never committed)
```


## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** — `pip install uv`
- **Groq API key** — free at [console.groq.com/keys](https://console.groq.com/keys)
- **SerpAPI key** — free tier at [serpapi.com/dashboard](https://serpapi.com/dashboard) (only needed for the news tool)
- **For cloud deploy only:** an AWS account + **AWS CLI** configured (`aws configure`)

---

## Setup (local)

```powershell
cd finance-agent

# 1. install dependencies into a virtual env
uv sync

# 2. create your .env with API keys
#    (create the file if it doesn't exist)
```

`.env` contents:
```bash
# Required
GROQ_API_KEY=<your-groq-key>
# Required for the news tool
SERPAPI_API_KEY=<your-serpapi-key>
```

---

## Run locally

```powershell
# one-shot
uv run python main.py "Give me an analyst brief on NVDA"

# interactive REPL (multi-turn; remembers context via SQLite)
uv run python main.py

# out-of-scope query → refused before any agent call
uv run python main.py "suggest me a diet plan"
```

(With the venv activated — `.\.venv\Scripts\Activate.ps1` — you can drop `uv run`.)

---

## Deploy to Amazon Bedrock AgentCore

### A. Runtime (stateless) — `agentcore_runtime.py`

```powershell
# 0. one-time: configure AWS credentials (region: ap-south-1)
aws configure

# 1. configure the deployment (generates .bedrock_agentcore.yaml)
agentcore configure -e agentcore_runtime.py

# 2. build + push + deploy to AWS
agentcore launch --env GROQ_API_KEY=<key> --env SERPAPI_API_KEY=<key>

# 3. call the deployed agent
agentcore invoke '{"prompt": "Give me an analyst brief on NVDA"}'
```

Useful: `agentcore status` (deployment state) · `agentcore destroy` (tear down).

### B. With memory — `agentcore_memory_runtime.py`

1. **Create an AgentCore Memory resource** (Console → Bedrock AgentCore → Memory →
   Create, region `ap-south-1`, short-term). Copy the **Memory ID**.
2. Add to `.env`: `MEMORY_ID=<id>` and `AWS_REGION=ap-south-1`.
3. Deploy:
   ```powershell
   agentcore configure -e agentcore_memory_runtime.py   # give it a distinct name, e.g. finbotmemory
   agentcore launch --env GROQ_API_KEY=<key> --env SERPAPI_API_KEY=<key> --env MEMORY_ID=<id>
   ```
4. Invoke with `actor_id` + `thread_id` (same values = same conversation):
   ```powershell
   agentcore invoke '{"prompt":"Brief on NVDA","actor_id":"alice","thread_id":"s1"}'
   agentcore invoke '{"prompt":"Now compare it with AMD","actor_id":"alice","thread_id":"s1"}'
   ```

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq LLM access (agent + router) |
| `SERPAPI_API_KEY` | for news | — | Google News tool |
| `MAIN_MODEL` | — | `openai/gpt-oss-20b` | Agent's reasoning model |
| `MEMORY_ID` | cloud memory only | — | AgentCore Memory resource id |

---

## Cost notes

- **Groq** — free tier has an 8,000 tokens/min limit; heavy multi-turn comparisons
  can hit it. Use one stock at a time, or upgrade Groq's tier.
- **AgentCore** — no per-service fee; you pay only for active runtime compute
  (~$0.0895/vCPU-hr, billed only while running, scales to zero when idle). New AWS
  accounts get up to **$200 in credits** — testing is effectively free.
- **AgentCore Memory** — a separate billable component (small). The stateless
  runtime avoids it.
- Run `agentcore destroy` when done to stop any lingering charges.
