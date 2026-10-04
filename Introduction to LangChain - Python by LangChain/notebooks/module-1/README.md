# Module 1 — Create an Agent
### LangChain Academy · Introduction to LangChain (Python)

> **Course:** [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)  
> **Instructor:** Seán (LangChain Academy)  
> **Module:** 1 of 3 — *Create an Agent*  
> **Model used:** `groq:openai/gpt-oss-120b` (Groq free tier)  
> **Tools:** LangChain · LangGraph · Tavily Search · LangSmith · LangGraph Studio

---

## What This Module Covers

Module 1 builds the complete foundation of AI agent development — from invoking a raw language model all the way to deploying a working agent with tools, memory, and a live chat UI via LangGraph Studio.

---

## Lessons & Notebooks

### Lesson 1.1 — Foundational Models & Prompting
**Notebooks:** `1.1_foundational_models.ipynb` · `1.1_prompting.ipynb`

**What we learned:**

- How to initialise any chat model using `init_chat_model()` from LangChain
- LangChain is **model-agnostic** — swapping providers (OpenAI → Anthropic → Groq → Gemini) is a single-line change
- A model invocation returns an `AIMessage` object containing `content` and `response_metadata` (token usage, model info, latency, etc.)
- Key model parameters:
  - `temperature` — controls creativity vs determinism (0 = fully deterministic, 1+ = more random/creative)
  - `max_tokens` — caps the response length
  - `timeout` — max time to wait before cancelling a request
  - `max_retries` — how many times to retry a failed request
- Wrapping a model inside `create_agent()` — the core LangGraph abstraction used throughout the entire course
- Streaming tokens with `agent.stream()` to reduce perceived latency (same technique used by ChatGPT, Claude, etc.)
- **System prompts** — the primary mechanism to customise agent behaviour and persona
- **Prompt engineering techniques:**
  - Few-shot examples: input/output pairs embedded in the prompt
  - Structured output prompts: specify the exact output format in the prompt
  - Pydantic `BaseModel` output schemas: forces the agent to return a typed, parseable object instead of free text

**Key code patterns:**
```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel

# Initialise a model
model = init_chat_model("groq:openai/gpt-oss-120b")

# Wrap it in an agent
agent = create_agent("groq:openai/gpt-oss-120b")

# Invoke it
response = agent.invoke({"messages": [HumanMessage(content="Hello!")]})
print(response["messages"][-1].content)

# Structured output with Pydantic
class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    system_prompt="You are a sci-fi writer.",
    response_format=CapitalInfo
)
```

---

### Lesson 1.2 — Tools & Web Search
**Notebooks:** `1.2_tools.ipynb` · `1.2_web_search.ipynb`

**What we learned:**

- What makes an agent different from a plain chatbot: **tools** — the ability to take actions, observe results, and react accordingly
- Agents follow the **ReAct pattern** (Reason + Act): decide which tool to call → call it → observe the output → form a final response
- Defining a tool with the `@tool` decorator — turns any Python function into something the agent can call autonomously
- Tool `name` defaults to the function name; tool `description` defaults to the docstring — both are critical for the agent to understand *when* to use the tool
- The **tool call message flow** under the hood:
  1. `HumanMessage` — the user's question
  2. `AIMessage` with empty content but a `tool_calls` field — the agent decides to use a tool
  3. `ToolMessage` — the result returned by the tool
  4. `AIMessage` with the final answer — the agent synthesises the result
- **Tavily Search** — a real-time web search API that returns LLM-friendly results, fixing the model's knowledge-cutoff limitation
- **LangSmith tracing** — visualise full agent runs with latency, token usage, and tool inputs/outputs; enabled by setting `LANGSMITH_TRACING=true` in `.env`

**Key code patterns:**
```python
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for up-to-date information"""
    return tavily_client.search(query)

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[web_search]
)
```

---

### Lesson 1.3 — Short-Term Memory
**Notebook:** `1.3_memory.ipynb`

**What we learned:**

- By default, agents have **no memory between runs** — each `invoke()` call starts completely fresh
- The agent tracks messages in a **state** object, but the state is not persisted across separate invocations
- Fix: use a **checkpointer** — saves a snapshot of the full state at the end of each run
- `InMemorySaver` from LangGraph stores state in RAM — perfect for development and notebooks
- **Thread IDs** group related invocations into a single conversation — the same thread ID means the agent remembers all previous messages in that thread
- Agent state by default tracks only messages, but can be extended with custom fields (covered in Module 2)

**Key code patterns:**
```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "groq:openai/gpt-oss-120b",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

# First message
agent.invoke({"messages": [HumanMessage(content="My name is Ali")]}, config)

# Second message — agent remembers the first
agent.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
# → "Your name is Ali"
```

---

### Lesson 1.4 — Multimodal Messages *(Bonus)*
**Notebook:** `1.4_multimodal_messages.ipynb`

**What we learned:**

- LLMs are not limited to text — many modern models accept **image and audio** inputs
- All binary media (images, audio files) must be **Base64-encoded** before being sent over text-based HTTP APIs
- **Image input:** encode a PNG/JPEG file to Base64, pass it inside a `HumanMessage` content block with `"type": "image"`
- **Audio input:** record audio with `sounddevice`/`scipy`, encode to Base64 WAV, pass with `"type": "audio"` (requires an audio-capable model such as `gpt-4o-audio-preview` on OpenAI; not supported on Groq free tier)
- Multimodal messages use a **list of content blocks** rather than a plain string
- Demonstrated with a generated lunar-base image — the agent was able to describe the scene, pick out the gas giant in the background, and improvise a story about the colony

**Key code patterns:**
```python
import base64
from langchain.messages import HumanMessage

# Image input
with open("resources/moon.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type": "text",  "text": "Tell me about this image"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = agent.invoke({"messages": [multimodal_question]})
```

> ⚠️ **Note:** Audio input requires `gpt-4o-audio-preview` (OpenAI). Groq free tier does not support audio input — the audio cell requires `OPENAI_API_KEY`.

---

## Module 1 Project — Personal Chef Agent

**Files:** `1.5_personal_chef.ipynb` · `1.5_personal_chef.py` · `langgraph.json`

### What it does

A fully conversational AI personal chef. You give it a list of leftover ingredients and it:
1. Searches the web (via Tavily) for recipes that use those ingredients
2. Returns ranked recipe suggestions with instructions
3. Answers follow-up questions — remembering the full conversation via short-term memory
4. *(Bonus)* Accepts an image of your fridge/pantry and picks a recipe from what it sees

### Architecture

```
User Input (text ingredients  OR  image of fridge)
        │
        ▼
  create_agent()
  ├── Model:   groq:openai/gpt-oss-120b
  ├── Tools:   web_search (Tavily)
  ├── Prompt:  "You are a personal chef..."
  └── Memory:  InMemorySaver + thread_id  (notebook)
              LangGraph Studio manages persistence (when run via `langgraph dev`)
        │
        ▼
  Agent decides to call web_search(query)
  (may issue multiple queries to broaden results)
        │
        ▼
  Tavily returns live, LLM-friendly web results
        │
        ▼
  Agent synthesises recipe suggestions + instructions
        │
        ▼
  Final response streamed back to user
```

### Project Code — Notebook version (`1.5_personal_chef.ipynb`)

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

system_prompt = """
You are a personal chef. The user will give you a list of ingredients
they have left over in their house.

Using the web search tool, search the web for recipes that can be made
with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the
user, if requested.
"""

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

# First turn
response = agent.invoke(
    {"messages": [HumanMessage(content="I have leftover chicken and rice. What can I make?")]},
    config
)
print(response["messages"][-1].content)

# Follow-up — agent remembers context
response = agent.invoke(
    {"messages": [HumanMessage(content="Give me full instructions for the first recipe")]},
    config
)
print(response["messages"][-1].content)
```

### Running in LangGraph Studio

The `.py` file (`1.5_personal_chef.py`) and `langgraph.json` allow running the agent in **LangGraph Studio** — a local chat UI with full step-by-step execution visualisation.

```jsonc
// langgraph.json
{
    "dependencies": ["."],
    "graphs": { "agent": "./1.5_personal_chef.py:agent" },
    "env": "../../.env"
}
```

```bash
# From the module-1 directory
cd notebooks/module-1
uv run langgraph dev
# Open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

> When running under `langgraph dev`, remove `checkpointer=InMemorySaver()` from the `.py` file — LangGraph Studio manages its own persistence automatically.

---

## Key Concepts Summary

| Concept | What it is | Where used |
|---|---|---|
| `init_chat_model()` | Provider-agnostic model initialisation | 1.1 |
| `create_agent()` | Core LangGraph agent abstraction used throughout the course | All lessons |
| `temperature` | Controls output randomness — 0 = deterministic, 1+ = creative | 1.1 |
| `response_format` | Pydantic schema for typed, structured output | 1.1 |
| `agent.stream()` | Stream tokens as they arrive to reduce perceived latency | 1.1 |
| `@tool` decorator | Turns a Python function into an agent-callable tool | 1.2 |
| ReAct pattern | Reason → Act → Observe loop that drives tool-using agents | 1.2 |
| `ToolMessage` | Message type that carries a tool's return value back to the agent | 1.2 |
| Tavily Search | Real-time web search API; fixes the model knowledge-cutoff problem | 1.2 |
| LangSmith tracing | Visualise full agent runs: latency, token usage, tool calls | 1.2 |
| `InMemorySaver` | Checkpointer that persists agent state across invocations (in RAM) | 1.3 |
| Thread ID | Groups multiple invocations into one continuous conversation | 1.3 |
| Base64 encoding | Converts binary media (images, audio) to text for API transmission | 1.4 |
| Multimodal messages | Content blocks combining text + image/audio in one message | 1.4 |
| `langgraph.json` | Config file pointing LangGraph Studio to your agent and `.env` | 1.5 |
| `langgraph dev` | Starts a local LangGraph Studio server with a chat UI | 1.5 |

---

## Project Folder Structure

```
notebooks/module-1/
├── 1.1_foundational_models.ipynb   # Model init, invocation, parameters
├── 1.1_prompting.ipynb             # System prompts, few-shot, structured output
├── 1.2_tools.ipynb                 # @tool decorator, tool call message flow
├── 1.2_web_search.ipynb            # Tavily integration, LangSmith tracing
├── 1.3_memory.ipynb                # InMemorySaver, thread IDs, stateful agents
├── 1.4_multimodal_messages.ipynb   # Base64 image & audio inputs
├── 1.5_personal_chef.ipynb         # Project notebook (with InMemorySaver)
├── 1.5_personal_chef.py            # Agent file for LangGraph Studio
├── langgraph.json                  # Studio config
└── resources/
    └── moon.png                    # Image used in multimodal lesson
```

---

## Setup

### Prerequisites
- Python 3.12–3.13
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Installation
```bash
# From the repo root (Introduction to LangChain - Python by LangChain/)
uv sync
```

### Environment Variables
Create a `.env` file in the `Introduction to LangChain - Python by LangChain/` root directory:

```env
# Required for Module 1
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional — for LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=lca-lc-foundation

# Optional — for Lesson 1.4 audio input only
OPENAI_API_KEY=your_openai_api_key_here
```

Get your keys:
- **Groq** (free tier): https://console.groq.com
- **Tavily** (free tier): https://tavily.com
- **LangSmith** (optional, free tier): https://smith.langchain.com
- **OpenAI** (optional, audio only): https://platform.openai.com

### Run Jupyter
```bash
# From the repo root
uv run jupyter lab
```

### Verify setup
```bash
# PowerShell (Windows)
$env:PYTHONUTF8 = "1"; uv run python env_utils.py
```

---

## Tech Stack

![LangChain](https://img.shields.io/badge/LangChain-≥1.3.10-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-≥1.0.3-blue)
![Groq](https://img.shields.io/badge/Groq-free_tier-orange)
![Tavily](https://img.shields.io/badge/Tavily-search_API-green)
![Python](https://img.shields.io/badge/Python-3.12--3.13-yellow)

---

## Next: Module 2 — Advanced Agent

Module 2 takes agent development to the next level:
- **Model Context Protocol (MCP)** — plug-and-play tool servers; build your own and consume community-built ones
- **Custom State & Runtime Context** — pass structured, typed data to agents at runtime; let agents update their own state
- **Multi-Agent Systems** — supervisor + specialist agent architectures for complex, long-running tasks
- **Project:** Destination Wedding Planner (real-time flights + venues + playlist, coordinated by a multi-agent team)
