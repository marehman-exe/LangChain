# Module 2 — Advanced Agent
### LangChain Academy · Introduction to LangChain (Python)

> **Course:** [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)  
> **Instructor:** Seán (LangChain Academy)  
> **Module:** 2 of 3 — *Advanced Agent*  
> **Model used:** `groq:openai/gpt-oss-120b` (Groq free tier)  
> **Tools:** LangChain · LangGraph · Tavily · MCP · SQLite · Kiwi Travel MCP · LangSmith

---

## What This Module Covers

Module 2 takes agent development beyond the basics — introducing plug-and-play tool servers (MCP), structured runtime context, mutable custom state, and full multi-agent orchestration. The capstone project wires all four together into a complete Destination Wedding Planner.

---

## Lessons & Notebooks

### Lesson 2.1 — Model Context Protocol (MCP)
**Notebooks:** `2.1_mcp.ipynb` · `2.1_travel_agent.ipynb`

**What we learned:**

- **What MCP is:** an open, vendor-agnostic protocol (by Anthropic) that standardises how AI applications connect to tools and data sources — the USB of AI
- The three things an MCP server can expose: **tools**, **resources** (read-only data), and **prompt templates**
- The two transport types: `stdio` (local subprocess) and `streamable_http` (remote/cloud server)
- **Building your own MCP server** from scratch using the `mcp` library with `@server.tool`, `@server.resource`, and `@server.prompt` decorators
- **Connecting to it** using `MultiServerMCPClient` from `langchain-mcp-adapters`
- **Consuming community servers** — browsing, copying a config snippet, and dropping it straight into `MultiServerMCPClient`
- Demonstrated with two real servers: the **MCP Time Server** (local, `stdio`) and the **Kiwi Travel MCP Server** (remote, `streamable_http`) for live flight search
- Why MCP matters for LangChain: built once, reusable across any agent or AI app that supports the protocol

**Key code patterns:**
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

# Local MCP server (stdio transport)
client = MultiServerMCPClient({
    "local_server": {
        "transport": "stdio",
        "command": "python",
        "args": ["resources/2.1_mcp_server.py"],
    }
})
tools = await client.get_tools()

# Remote MCP server (streamable HTTP transport)
client = MultiServerMCPClient({
    "travel_server": {
        "transport": "streamable_http",
        "url": "https://mcp.kiwi.com"
    }
})
tools = await client.get_tools()

agent = create_agent(model="groq:openai/gpt-oss-120b", tools=tools)
```

---

### Lesson 2.2 — Custom State & Runtime Context
**Notebooks:** `2.2_state.ipynb` · `2.2_runtime_context.ipynb`

**What we learned:**

**Runtime Context (`context_schema`):**
- A read-only, immutable data class passed to the agent at invocation time — the agent cannot change it
- Accessed inside tools via `runtime.context`
- Perfect for per-request data the caller controls: user role, preferred language, tenant ID
- Does not persist between turns — passed fresh every `invoke()`

**Custom State (`state_schema`):**
- Extends `AgentState` with additional mutable fields tracked across turns by the checkpointer
- Written from inside a tool using `Command(update={...})` — LangGraph's mechanism for state mutation
- Read from inside a tool using `runtime.state["field_name"]`
- Persisted per `thread_id` by `InMemorySaver` — survives across multiple `invoke()` calls on the same thread

**Key code patterns:**
```python
from dataclasses import dataclass
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

# ── Runtime context (immutable, per-invocation) ──
@dataclass
class UserContext:
    preferred_language: str = "English"
    user_role: str = "external"

# ── Custom state (mutable, persisted per thread) ──
class MyState(AgentState):
    favourite_colour: str

@tool
def update_favourite_colour(colour: str, runtime: ToolRuntime) -> Command:
    """Save the user's favourite colour to state."""
    return Command(update={
        "favourite_colour": colour,
        "messages": [ToolMessage("Saved!", tool_call_id=runtime.tool_call_id)],
    })

@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the user's favourite colour from state."""
    return runtime.state.get("favourite_colour", "unknown")

agent = create_agent(
    "groq:openai/gpt-oss-120b",
    tools=[update_favourite_colour, read_favourite_colour],
    state_schema=MyState,
    checkpointer=InMemorySaver(),
)
```

---

### Lesson 2.3 — Multi-Agent Systems
**Notebook:** `2.3_multi_agent.ipynb`

**What we learned:**

- Why single agents break down on large, complex tasks: too many tools → overloaded context → degraded output quality
- **Supervisor + sub-agent architecture**: one orchestrating agent delegates to specialised sub-agents — each with their own focused tool set and system prompt
- **The trick:** wrap each sub-agent in a `@tool` function — this is what the supervisor calls; from the supervisor's view it's just another tool call
- Sub-agent internals (their own tool calls, messages) are invisible to the supervisor — only the final answer surfaces
- LangSmith tracing is essential here: the outer message view hides sub-agent activity; the trace shows the full tree
- Architecture generalises to any depth — sub-agents can themselves be supervisors

**Key code patterns:**
```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

# Sub-agents
sqrt_agent = create_agent("groq:openai/gpt-oss-120b", tools=[square_root])
sq_agent   = create_agent("groq:openai/gpt-oss-120b", tools=[square])

# Wrap each sub-agent as a tool for the supervisor
@tool
def call_sqrt_agent(x: float) -> str:
    """Delegate square root calculation to the specialist sub-agent."""
    resp = sqrt_agent.invoke({"messages": [HumanMessage(f"Square root of {x}")]})
    return resp["messages"][-1].content

@tool
def call_sq_agent(x: float) -> str:
    """Delegate squaring to the specialist sub-agent."""
    resp = sq_agent.invoke({"messages": [HumanMessage(f"Square of {x}")]})
    return resp["messages"][-1].content

# Supervisor — just sees two tools
supervisor = create_agent(
    "groq:openai/gpt-oss-120b",
    tools=[call_sqrt_agent, call_sq_agent],
)
```

---

## Module 2 Project — Destination Wedding Planner

**File:** `2.4_wedding_planners.ipynb`

### What it does

A multi-agent wedding planning team. You describe your ideal wedding in one sentence — the coordinator gathers the details, delegates to three specialist sub-agents, and returns a complete plan:

| Sub-agent | Tool | Task |
|-----------|------|------|
| **Travel Agent** | Kiwi Travel MCP (live flights) | Finds the best flights from origin → destination |
| **Venue Agent** | Tavily web search | Finds venues matching the location and guest count |
| **Playlist Agent** | SQL query on Chinook DB | Curates a playlist for the requested genre |
| **Coordinator** | `update_state`, calls all 3 agents | Collects details, updates shared state, compiles the final plan |

### Architecture

```
User: "I'm from London, wedding in Paris, 100 guests, jazz"
             │
             ▼
      ┌─────────────────────────────────────────────┐
      │            Coordinator Agent                 │
      │  tools: update_state, search_flights,        │
      │          search_venues, suggest_playlist      │
      │  state_schema: WeddingState                  │
      └──┬──────────────────────────────────────────┘
         │
         │ 1. update_state(origin, destination,
         │                 guest_count, genre)
         │    → writes to WeddingState via Command
         │
         ├──► 2. search_flights(runtime)
         │        reads state["origin"] + state["destination"]
         │        ┌──────────────────────────────┐
         │        │  Travel Sub-Agent             │
         │        │  tool: Kiwi MCP (live search) │
         │        └──────────────────────────────┘
         │
         ├──► 3. search_venues(runtime)
         │        reads state["destination"] + state["guest_count"]
         │        ┌──────────────────────────────┐
         │        │  Venue Sub-Agent              │
         │        │  tool: web_search (Tavily)    │
         │        └──────────────────────────────┘
         │
         └──► 4. suggest_playlist(runtime)
                  reads state["genre"]
                  ┌──────────────────────────────┐
                  │  Playlist Sub-Agent           │
                  │  tool: query_playlist_db (SQL)│
                  └──────────────────────────────┘
             │
             ▼
      Coordinator compiles results → Final wedding plan
```

### Key implementation details

**`WeddingState`** — shared memory across all agents in the team:
```python
class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str
```

**`update_state` tool** — coordinator fills the state before delegating:
```python
@tool
def update_state(origin, destination, guest_count, genre, runtime: ToolRuntime) -> Command:
    """Update the wedding details in the shared state."""
    return Command(update={
        "origin": origin, "destination": destination,
        "guest_count": guest_count, "genre": genre,
        "messages": [ToolMessage("State updated", tool_call_id=runtime.tool_call_id)],
    })
```

**Sub-agent tool** — reads from state, delegates to sub-agent:
```python
@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """Travel agent searches for flights using state data."""
    origin      = runtime.state["origin"]
    destination = runtime.state["destination"]
    resp = await travel_agent.ainvoke({
        "messages": [HumanMessage(f"Find flights from {origin} to {destination}")]
    })
    return resp["messages"][-1].content
```

**MCP error handling** — `RetryMCPInterceptor` wraps every Kiwi MCP call with retries and graceful fallback so the agent degrades cleanly instead of crashing on transient network errors.

---

## Key Concepts Summary

| Concept | What it is | Where used |
|---------|-----------|------------|
| MCP (Model Context Protocol) | Universal standard for connecting agents to tools/data | 2.1 |
| `MultiServerMCPClient` | LangChain adapter to connect to any MCP server | 2.1 |
| `stdio` transport | Runs MCP server as a local subprocess | 2.1 |
| `streamable_http` transport | Connects to a remote/cloud MCP server over HTTP | 2.1 |
| `context_schema` | Immutable per-invocation data class; accessed via `runtime.context` | 2.2 |
| `state_schema` | Mutable fields added to `AgentState`; persisted per `thread_id` | 2.2 |
| `Command(update={...})` | LangGraph primitive — writes to state from inside a tool | 2.2 |
| `runtime.state` | Read live state fields from inside any tool | 2.2 |
| Supervisor + sub-agents | Orchestrator delegates to specialised agents wrapped as tools | 2.3 |
| `@tool` wrapping an agent | How a sub-agent becomes callable by the supervisor | 2.3 |
| `WeddingState` | Shared state passed between coordinator and all sub-agents | 2.4 |
| `RetryMCPInterceptor` | Custom middleware that retries transient MCP failures gracefully | 2.4 |
| `ainvoke` / `async` | Required when sub-agents use async MCP tools (e.g. Kiwi flight search) | 2.4 |
| LangSmith tracing | Essential for multi-agent — shows the full nested call tree | 2.4 |

---

## Project Folder Structure

```
notebooks/module-2/
├── 2.1_mcp.ipynb              # Local + remote MCP server connections
├── 2.1_travel_agent.ipynb     # Travel agent with Kiwi MCP live flight search
├── 2.2_state.ipynb            # Custom state: write + read via Command & ToolRuntime
├── 2.2_runtime_context.ipynb  # Immutable runtime context via context_schema
├── 2.3_multi_agent.ipynb      # Supervisor + sub-agent pattern
├── 2.4_wedding_planners.ipynb # Project: full multi-agent wedding planner
├── bonus_rag.ipynb            # Bonus: RAG agent
├── bonus_sql.ipynb            # Bonus: SQL agent
└── resources/
    ├── 2.1_mcp_server.py      # Custom local MCP server implementation
    ├── Chinook.db             # SQLite music database (used by playlist agent)
    └── acmecorp-employee-handbook.pdf  # Document for bonus RAG notebook
```

---

## Setup

### Prerequisites
- Python 3.12–3.13 · [uv](https://docs.astral.sh/uv/)

### Environment Variables
Add to your `.env` file (in the `Introduction to LangChain - Python by LangChain/` root):

```env
# Required
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional — LangSmith (strongly recommended for multi-agent tracing)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=lca-lc-foundation
```

> **Note:** The Kiwi Travel MCP server (`https://mcp.kiwi.com`) is a free public endpoint — no API key required.

---

## Next: Module 3 — Production-Ready Agent

Module 3 focuses on deploying agents in front of real users:
- **Middleware** — insert custom logic before/after model calls or tool calls
- **Managing Long Conversations** — summarise or trim messages to prevent context overflow
- **Human-in-the-Loop (HITL)** — approve, reject, or edit agent actions before they execute
- **Dynamic Agents** — swap prompts, tools, and even the model itself at runtime based on user role or conversation state
- **Project:** Email Assistant (authentication + HITL email approval)
