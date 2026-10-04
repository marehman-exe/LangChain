# LangChain Learning Repository

> Personal study notes, notebooks, and projects from the **LangChain Academy — Introduction to LangChain (Python)** course.  
> GitHub: [marehman-exe/LangChain](https://github.com/marehman-exe/LangChain)

---

## Repository Structure

```
LangChain/
├── Introduction to LangChain - Python by LangChain/   ← Main course folder
│   ├── notebooks/
│   │   ├── module-1/      ← Create an Agent          ✅ Complete
│   │   ├── module-2/      ← Advanced Agent            (in progress)
│   │   └── module-3/      ← Production-Ready Agent    (in progress)
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── env_utils.py
│   └── README.md
│
├── Examplers/             ← RAG from Scratch reference notebooks & PDFs
│   ├── Notebooks/
│   └── Details/
│
├── Modernized/            ← Modernized / refactored versions of examples
│
└── LangChain_RAG-1-4.ipynb   ← Standalone RAG notebook
```

---

## Course Progress

### [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)
Instructor: Seán (LangChain Academy)

| Module | Title | Status | Project |
|--------|-------|--------|---------|
| [Module 1](Introduction%20to%20LangChain%20-%20Python%20by%20LangChain/notebooks/module-1/README.md) | Create an Agent | ✅ **Complete** | Personal Chef Agent |
| Module 2 | Advanced Agent | 🔄 In Progress | Wedding Planner Agent |
| Module 3 | Production-Ready Agent | 🔄 In Progress | Email Assistant Agent |

---

## Module 1 — Create an Agent ✅

> **Full documentation:** [`notebooks/module-1/README.md`](Introduction%20to%20LangChain%20-%20Python%20by%20LangChain/notebooks/module-1/README.md)

Module 1 covers the complete foundation of building AI agents with LangChain and LangGraph.

### What Was Learned

#### 1.1 — Foundational Models & Prompting
- Initialising any chat model with `init_chat_model()` — provider-agnostic by design
- LangChain supports OpenAI, Anthropic, Groq, Google Gemini and more with a single interface
- `AIMessage` response structure: `content`, `response_metadata` (tokens, latency)
- Model parameters: `temperature`, `max_tokens`, `timeout`, `max_retries`
- Building an agent with `create_agent()` — the core LangGraph abstraction
- Streaming responses with `agent.stream()` to reduce perceived latency
- System prompts, few-shot examples, and Pydantic structured output schemas

#### 1.2 — Tools & Web Search
- What makes an agent different: **tools** — the ability to act, observe, and react
- The **ReAct pattern**: Reason → Act (tool call) → Observe → Final response
- `@tool` decorator — turns any Python function into an agent-callable action
- Tool call message flow: `HumanMessage → AIMessage (tool_calls) → ToolMessage → AIMessage`
- **Tavily Search** integration — real-time web search, fixes the model knowledge-cutoff problem
- **LangSmith tracing** — visualise agent runs, latency, token usage, tool I/O

#### 1.3 — Short-Term Memory
- Default agents have no cross-invocation memory — state is wiped on each call
- **Checkpointers** (`InMemorySaver`) save state snapshots between runs
- **Thread IDs** group invocations into conversations — same thread = same memory
- Demonstrated: agent remembers name and preferences across multiple calls

#### 1.4 — Multimodal Messages *(Bonus)*
- Base64 encoding binary media (images/audio) for text-based HTTP APIs
- Image input via `HumanMessage` content blocks with `"type": "image"`
- Audio input via `"type": "audio"` (requires OpenAI `gpt-4o-audio-preview`)
- Multimodal messages use lists of content blocks instead of plain strings

### Module 1 Project — Personal Chef Agent

A fully conversational AI chef agent that:
1. Accepts leftover ingredients (text input) or an image of your fridge
2. Searches the web via Tavily for matching recipes
3. Returns ranked recipe suggestions with full instructions
4. Remembers the conversation to answer follow-up questions

**Tech used:** `create_agent()` · `@tool` · Tavily · `InMemorySaver` · LangGraph Studio

**Files:**
- [`notebooks/module-1/1.5_personal_chef.ipynb`](Introduction%20to%20LangChain%20-%20Python%20by%20LangChain/notebooks/module-1/1.5_personal_chef.ipynb) — interactive notebook
- [`notebooks/module-1/1.5_personal_chef.py`](Introduction%20to%20LangChain%20-%20Python%20by%20LangChain/notebooks/module-1/1.5_personal_chef.py) — agent file for LangGraph Studio
- [`notebooks/module-1/langgraph.json`](Introduction%20to%20LangChain%20-%20Python%20by%20LangChain/notebooks/module-1/langgraph.json) — Studio config

---

## Additional Resources

### RAG from Scratch (Examplers)

Reference notebooks and PDF overviews covering **Retrieval-Augmented Generation** from first principles:

| Notebook | Topics |
|----------|--------|
| `rag_from_scratch_1_to_4_Exampler.ipynb` | Indexing, embedding, vector stores, basic retrieval |
| `rag_from_scratch_5_to_9_Exampler.ipynb` | Query transformation, multi-query, RAG-Fusion |
| `rag_from_scratch_10_and_11_Exampler.ipynb` | Routing, query structuring |
| `rag_from_scratch_12_to_14_Exampler.ipynb` | Indexing techniques: multi-representation, RAPTOR |
| `rag_from_scratch_15_to_18_Exampler.ipynb` | CRAG, Self-RAG, Adaptive RAG |

---

## Setup

### Prerequisites
- Python 3.12–3.13
- [uv](https://docs.astral.sh/uv/) (recommended)

### Install

```bash
cd "Introduction to LangChain - Python by LangChain"
uv sync
```

### Environment Variables

Copy `example.env` to `.env` and fill in your keys:

```env
# Required for Module 1
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional — LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=lca-lc-foundation

# Optional — audio input in Lesson 1.4 only
OPENAI_API_KEY=your_openai_api_key_here
```

| Key | Required | Where to get it |
|-----|----------|-----------------|
| `GROQ_API_KEY` | ✅ Yes | https://console.groq.com (free) |
| `TAVILY_API_KEY` | ✅ Yes | https://tavily.com (free tier) |
| `LANGSMITH_API_KEY` | Optional | https://smith.langchain.com (free tier) |
| `OPENAI_API_KEY` | Optional | https://platform.openai.com (audio in 1.4 only) |

### Run Jupyter

```bash
uv run jupyter lab
```

### Verify setup

```bash
# PowerShell (Windows)
$env:PYTHONUTF8 = "1"; uv run python env_utils.py
```

### Run LangGraph Studio (Module 1 project)

```bash
cd "Introduction to LangChain - Python by LangChain/notebooks/module-1"
uv run langgraph dev
# Opens: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

---

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| [LangChain](https://python.langchain.com) | ≥ 1.3.10 | Agent framework, model abstraction, tools |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | ≥ 1.0.3 | Agent state machine, checkpointers, Studio |
| [Groq](https://console.groq.com) | free tier | Fast inference, `openai/gpt-oss-120b` model |
| [Tavily](https://tavily.com) | free tier | Real-time web search for agents |
| [LangSmith](https://smith.langchain.com) | optional | Tracing, debugging, observability |
| Python | 3.12–3.13 | Runtime |
| uv | latest | Package manager & virtual environment |

---

## Key Concepts Covered So Far

| Concept | Module |
|---------|--------|
| `init_chat_model()` — provider-agnostic model init | 1.1 |
| `create_agent()` — core LangGraph agent wrapper | 1.1 |
| System prompts & prompt engineering | 1.1 |
| Pydantic structured output (`response_format`) | 1.1 |
| Token streaming (`agent.stream()`) | 1.1 |
| `@tool` decorator — custom agent tools | 1.2 |
| ReAct agent loop (Reason → Act → Observe) | 1.2 |
| Tavily web search integration | 1.2 |
| LangSmith tracing & observability | 1.2 |
| `InMemorySaver` checkpointer — short-term memory | 1.3 |
| Thread IDs — multi-turn conversations | 1.3 |
| Base64 multimodal input (images & audio) | 1.4 |
| LangGraph Studio local deployment | 1.5 |

---

*Learning path: [LangChain Academy](https://academy.langchain.com) · [LangChain Docs](https://python.langchain.com) · [LangGraph Docs](https://langchain-ai.github.io/langgraph/)*
