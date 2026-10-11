# Module 3 — Production-Ready Agent
### LangChain Academy · Introduction to LangChain (Python)

> **Course:** [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)  
> **Instructor:** Seán (LangChain Academy)  
> **Module:** 3 of 3 — *Production-Ready Agent*  
> **Model used:** `groq:openai/gpt-oss-120b` + `groq:openai/gpt-oss-20b`  
> **Tools:** LangChain · LangGraph · Middleware · HITL · LangGraph Studio · Agent Chat UI

---

## What This Module Covers

Module 3 takes an agent from prototype to production-ready by introducing **middleware** — functions inserted into the agent's execution loop that give fine-grained control over behaviour. The capstone is a complete Email Assistant with authentication, dynamic tools, dynamic prompts, and human-in-the-loop approval for sending emails.

---

## Lessons & Notebooks

### Lesson 3.1 — What is Middleware?
*(Introduction video — no notebook)*

**What we learned:**

- **Middleware** is a catch-all term for functions inserted within the agent's execution loop to control and customise its behaviour
- Two types:
  - **Node-style middleware** — hooks into the agent graph `before_agent` / `after_agent` / `before_model` / `after_model`; receives `state + runtime`; used to modify messages or state
  - **Wrap-style middleware** — wraps the model call itself via `@wrap_model_call`; receives the `ModelRequest`; used to change the model, tools, or system prompt dynamically
- Real-world use cases: output classifiers, human approval gates, prompt personalisation, permission-based tool access

---

### Lesson 3.2 — Managing Long Conversations
**Notebook:** `3.2_managing_messages.ipynb`

**What we learned:**

- Long conversations overflow the context window → slower, more expensive, lower quality
- **Fix 1 — Summarisation middleware** (`SummarizationMiddleware`): prebuilt drop-in that summarises the conversation when a token threshold is reached, then deletes older messages keeping only the most recent ones
- **Fix 2 — Custom trim middleware**: write your own `@before_agent` function that takes `state` and `runtime`, finds target messages (e.g. all `ToolMessage`s), and returns `RemoveMessage` objects to delete them before they reach the model

**Key code patterns:**
```python
from langchain.agents.middleware import SummarizationMiddleware, before_agent
from langchain.agents import AgentState, create_agent
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.runtime import Runtime
from typing import Any

# Option 1 — prebuilt summarisation
agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="groq:openai/gpt-oss-120b",
            trigger=("tokens", 100),   # summarise when >100 tokens
            keep=("messages", 1),      # keep only the last message after summarising
        )
    ],
)

# Option 2 — custom trim: remove all ToolMessages before the agent sees them
@before_agent
def trim_tool_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all tool messages from the state before the agent processes them."""
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    return {"messages": [RemoveMessage(id=m.id) for m in tool_msgs]}

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    checkpointer=InMemorySaver(),
    middleware=[trim_tool_messages],
)
```

---

### Lesson 3.3 — Human-in-the-Loop (HITL)
**Notebook:** `3.3_hitl.ipynb`

**What we learned:**

- `HumanInTheLoopMiddleware` — prebuilt middleware that pauses the agent before executing specified tool calls and waits for a human decision
- Configure per-tool: `interrupt_on={"send_email": True, "read_email": False}`
- Three decision types when resuming:
  - `"approve"` — execute the tool call as-is
  - `"reject"` — skip the tool call; agent is told the reason and may try again
  - `"edit"` — swap in different tool args, then approve immediately
- The paused state lives in `response['__interrupt__']` — inspect it to see what the agent wants to do
- Resume by invoking with `Command(resume={"decisions": [{"type": "approve"}]})` and the **same `thread_id`**

**Key code patterns:**
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

# Build agent with HITL on send_email only
agent = create_agent(
    "groq:openai/gpt-oss-120b",
    tools=[read_email, send_email],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True, "read_email": False})
    ],
)

config = {"configurable": {"thread_id": "1"}}
response = agent.invoke({"messages": [HumanMessage("Read my email and reply")]}, config)

# Agent paused — inspect what it wants to send
print(response["__interrupt__"][0].value["action_requests"][0]["args"]["body"])

# Approve it
agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

# Reject with a reason
agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "Too formal."}]}), config)

# Edit and approve
agent.invoke(Command(resume={"decisions": [{"type": "edit",
    "edited_action": {"name": "send_email", "args": {"body": "My revised text"}}}]}), config)
```

---

### Lesson 3.4 — Dynamic Agents
**Notebooks:** `3.4_dynamic_prompts.ipynb` · `3.4_dynamic_tools.ipynb` · `3.4_dynamic_models.ipynb`

**What we learned:**

**Dynamic prompts** (`@dynamic_prompt`):
- Wrap-style middleware that replaces the system prompt at runtime based on state or context
- Receives `ModelRequest`; returns a new system prompt string

**Dynamic tools** (`@wrap_model_call`):
- Swap the tool list at runtime based on user role or state
- `request.override(tools=[...])` replaces the tool list for this model call only
- Pattern: internal users get all tools; external users get a restricted set

**Dynamic models** (`@wrap_model_call`):
- Swap the LLM itself at runtime (e.g. light model for short conversations, large model for long ones)
- `request.override(model=...)` replaces the model for this call

**Key code patterns:**
```python
from langchain.agents.middleware import wrap_model_call, dynamic_prompt, ModelRequest, ModelResponse
from typing import Callable

# ── Dynamic prompt ──
@dynamic_prompt
def language_prompt(request: ModelRequest) -> str:
    lang = request.runtime.context.user_language
    base = "You are a helpful assistant."
    return f"{base} Respond only in {lang}." if lang != "English" else base

# ── Dynamic tools ──
@wrap_model_call
def role_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    if request.runtime.context.user_role == "internal":
        pass  # keep all tools
    else:
        request = request.override(tools=[web_search])  # restrict external users
    return handler(request)

# ── Dynamic model ──
@wrap_model_call
async def conversation_length_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    model = large_model if len(request.messages) > 10 else standard_model
    request = request.override(model=model)
    return await handler(request)
```

> **Node-style vs Wrap-style middleware:**
> | | Node-style (`@before_agent` etc.) | Wrap-style (`@wrap_model_call`) |
> |---|---|---|
> | **Accesses** | `state` + `runtime` | `ModelRequest` (model, tools, prompt, messages) |
> | **Used for** | Modifying state / messages | Changing model, tools, or prompt dynamically |
> | **Runs** | Once per agent run (or per model call) | On every model invocation |

---

## Module 3 Project — Email Assistant Agent

**Files:** `3.5_email_agent.ipynb` · `3.5_email_agent.py` · `langgraph.json`

### What it does

A secure email assistant that:
1. **Authenticates** the user — compares email + password against context; only grants inbox access on success
2. **Reads the inbox** — returns the latest dummy email
3. **Drafts a reply** — the agent writes an email response
4. **Waits for human approval** before sending — you can approve, reject with a note, or edit the draft

### Three middleware layers working together

```
User message
      │
      ▼
 ┌─────────────────────────────────────────────────┐
 │  dynamic_tool_call  (@wrap_model_call)           │
 │  if authenticated → tools: [check_inbox,         │
 │                              send_email]          │
 │  else             → tools: [authenticate]        │
 └─────────────────────┬───────────────────────────┘
                       │
      ▼
 ┌─────────────────────────────────────────────────┐
 │  dynamic_prompt_func  (@dynamic_prompt)          │
 │  if authenticated → "You are an email agent..."  │
 │  else             → "You are an authenticator..."│
 └─────────────────────┬───────────────────────────┘
                       │
      ▼
 ┌─────────────────────────────────────────────────┐
 │  HumanInTheLoopMiddleware                        │
 │  interrupt_on: send_email=True                   │
 │  → pauses before sending; waits for approval     │
 └─────────────────────────────────────────────────┘
```

### Key implementation details

```python
# Context — credentials the authenticate tool checks against (immutable, per-invocation)
@dataclass
class EmailContext:
    email_address: str = "julie@example.com"
    password: str = "password123"

# State — tracks authentication across turns (mutable, per thread_id)
class AuthenticatedState(AgentState):
    authenticated: bool

# authenticate tool — writes to state via Command
@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    if email == runtime.context.email_address and password == runtime.context.password:
        return Command(update={"authenticated": True, "messages": [...]})
    return Command(update={"authenticated": False, "messages": [...]})
```

### LangGraph Studio deployment

```bash
cd notebooks/module-3
uv run langgraph dev
# → https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### Agent Chat UI (Bonus)

The `agent-chat-ui/` folder contains a customised Next.js chat interface that:
- Connects to your locally running LangGraph Studio instance
- Renders HITL interrupts as clean approve / reject / edit UI elements
- Supports file attachments (PDF, images)

```bash
cd notebooks/module-3/agent-chat-ui
pnpm install
pnpm dev
# → visit https://agentchat.vercel.app or http://localhost:3000
```

---

## Key Concepts Summary

| Concept | What it is | Where used |
|---------|-----------|------------|
| `middleware=[...]` | List of middleware passed to `create_agent()` | All lessons |
| `@before_agent` | Node-style hook: runs once before the agent; receives `state + runtime` | 3.2 |
| `@after_agent` | Node-style hook: runs once after the agent | 3.2 |
| `@before_model` / `@after_model` | Node-style hooks: run before/after every model call | 3.2 |
| `SummarizationMiddleware` | Prebuilt: summarises + trims messages when token limit hit | 3.2 |
| `RemoveMessage` | LangGraph message-deletion primitive | 3.2 |
| `HumanInTheLoopMiddleware` | Prebuilt: pauses agent before specified tool calls | 3.3 |
| `__interrupt__` | Response field holding the paused tool call details | 3.3 |
| `Command(resume=...)` | Resumes a paused HITL agent with approve / reject / edit | 3.3 |
| `@dynamic_prompt` | Wrap-style: replaces system prompt at runtime | 3.4 |
| `@wrap_model_call` | Wrap-style: intercepts ModelRequest to change model/tools/prompt | 3.4 |
| `request.override(tools=...)` | Replaces the tool list for this model call | 3.4 |
| `request.override(model=...)` | Replaces the LLM for this model call | 3.4 |
| `context_schema` | Immutable per-invocation data class; read via `runtime.context` | 3.4, 3.5 |
| `AuthenticatedState` | Custom state tracking auth status; written by `authenticate` tool | 3.5 |
| `dynamic_tool_call` | Shows auth tools before login, inbox tools after | 3.5 |
| `dynamic_prompt_func` | Switches system prompt based on `authenticated` state field | 3.5 |
| `async` on `@wrap_model_call` | Required when the function uses `await handler(request)` for Studio | 3.5, 3.6 |
| `langgraph.json` | Points Studio to the agent `.py` file and `.env` | 3.5, 3.6 |
| Agent Chat UI | Open-source Next.js UI; renders HITL as approve/reject/edit buttons | 3.6 |

---

## Project Folder Structure

```
notebooks/module-3/
├── 3.2_managing_messages.ipynb   # Summarisation + custom trim middleware
├── 3.3_hitl.ipynb                # HumanInTheLoop: approve / reject / edit
├── 3.4_dynamic_prompts.ipynb     # @dynamic_prompt based on context
├── 3.4_dynamic_tools.ipynb       # @wrap_model_call: role-based tool access
├── 3.4_dynamic_models.ipynb      # @wrap_model_call: conversation-length model swap
├── 3.5_email_agent.ipynb         # Project notebook
├── 3.5_email_agent.py            # Agent file for LangGraph Studio
├── langgraph.json                # Studio config
├── agent-chat-ui/                # Customised Agent Chat UI (Next.js)
└── resources/                    # Shared resources (Chinook DB, etc.)
```

---

## Next Steps

You've now completed the full **Introduction to LangChain — Python** course. You can:
- Deploy your agents via LangGraph Cloud
- Connect real email APIs (Gmail, Outlook) to replace the dummy `check_inbox` / `send_email` tools
- Add a database-backed checkpointer (`langgraph-checkpoint-sqlite`) to persist sessions across restarts
- Explore the [LangChain Docs](https://python.langchain.com) and [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
