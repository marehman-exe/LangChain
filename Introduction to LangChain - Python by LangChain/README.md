# 🔗 Introduction to LangChain — Python
### LangChain Academy · Hands-On Course Notes & Projects

> **Course:** [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)  
> **Instructor:** Seán (LangChain Academy)  
> **Learner:** Muhammad Ali Rehman  
> **GitHub:** [@marehman-exe](https://github.com/marehman-exe)  
> **Models used:** `groq:openai/gpt-oss-120b` (Groq free tier)  
> **Stack:** LangChain · LangGraph · Tavily Search · LangSmith · LangGraph Studio

---

## Overview

This repository contains my complete notes, hands-on notebooks, and finished projects from the **Introduction to LangChain (Python)** course — a 3-module curriculum that takes you from a raw language model invocation all the way to deploying production-grade agents with middleware, human-in-the-loop controls, and a live chat UI.

---

## 📁 Repository Structure

```
Introduction to LangChain - Python by LangChain/
│
├── notebooks/
│   ├── module-1/          ← Create an Agent (foundations)
│   │   ├── README.md      ← Detailed lesson notes for Module 1
│   │   ├── 1.1_foundational_models.ipynb
│   │   ├── 1.1_prompting.ipynb
│   │   ├── 1.2_tools.ipynb
│   │   ├── 1.2_web_search.ipynb
│   │   ├── 1.3_memory.ipynb
│   │   ├── 1.4_multimodal_messages.ipynb
│   │   ├── 1.5_personal_chef.ipynb   ← Project notebook
│   │   ├── 1.5_personal_chef.py      ← LangGraph Studio entry point
│   │   ├── langgraph.json
│   │   └── resources/moon.png
│   │
│   ├── module-2/          ← Advanced Agent (MCP, multi-agent)
│   │   ├── README.md      ← Detailed lesson notes for Module 2
│   │   └── ...
│   │
│   └── module-3/          ← Production-Ready Agent (middleware, HITL)
│       ├── README.md      ← Detailed lesson notes for Module 3
│       └── ...
│
├── env_utils.py           ← Environment verification script
├── pyproject.toml         ← Project dependencies
├── requirements.txt
├── example.env            ← Template for .env (copy → .env, fill in keys)
└── README.md              ← This file
```

---

## 🏗️ Modules at a Glance

| Module | Title | Project | Key Skills |
|--------|-------|---------|------------|
| **1** | Create an Agent | Personal Chef Agent | `create_agent`, `@tool`, `InMemorySaver`, multimodal |
| **2** | Advanced Agent | Destination Wedding Planner | MCP, custom state, multi-agent supervisor |
| **3** | Production-Ready Agent | Email Assistant | Middleware, HITL, dynamic prompts, LangGraph Studio |

---

## 📚 Module 1 — Create an Agent

> **Full lesson notes:** [`notebooks/module-1/README.md`](notebooks/module-1/README.md)

Module 1 builds the complete foundation of AI agent development from scratch.

### What We Learned

#### Lesson 1.1 — Foundational Models & Prompting
- Initialise any chat model with `init_chat_model()` — LangChain is **model-agnostic**
- Swap providers (OpenAI → Anthropic → Groq → Gemini) with a single-line change
- Key model parameters: `temperature`, `max_tokens`, `timeout`, `max_retries`
- `create_agent()` — the core LangGraph abstraction used throughout the entire course
- Stream tokens with `agent.stream()` to reduce perceived latency
- **Prompt engineering:** few-shot examples, structured output prompts, Pydantic `BaseModel` schemas

#### Lesson 1.2 — Tools & Web Search
- **ReAct pattern:** Reason → Act (tool call) → Observe (ToolMessage) → Final Answer
- Define tools with the `@tool` decorator; name = function name, description = docstring
- Tool call message flow: `HumanMessage` → `AIMessage(tool_calls=[...])` → `ToolMessage` → `AIMessage`
- **Tavily Search** API for real-time web search — fixes the model knowledge-cutoff limitation
- **LangSmith tracing** for visualising full agent runs with latency, token usage, tool I/O

#### Lesson 1.3 — Short-Term Memory
- Without a checkpointer each `invoke()` call starts completely fresh (no memory)
- `InMemorySaver` from LangGraph persists state snapshots across invocations (in RAM)
- **Thread IDs** group related invocations into one continuous conversation

#### Lesson 1.4 — Multimodal Messages *(Bonus)*
- Binary media (images, audio) must be **Base64-encoded** before transmission over HTTP
- `HumanMessage` with a list of content blocks: `{"type": "text", ...}` + `{"type": "image", ...}`
- Audio input requires an audio-capable model (`gpt-4o-audio-preview`); Groq free tier is text-only

#### Lesson 1.5 — Project: Personal Chef Agent
- Combined all Module 1 skills: `create_agent` + `web_search` tool + `InMemorySaver` + `langgraph.json`
- Bonus: accepts an image of a fridge → picks a recipe from what the agent sees
- Deployed to **LangGraph Studio** local chat UI via `uv run langgraph dev`

### Project: Personal Chef Agent

A fully conversational AI chef. Give it leftover ingredients → it searches the web via Tavily, returns ranked recipe suggestions with full instructions, and remembers the conversation for follow-up questions.

```
User Input (text OR image of fridge)
        │
        ▼
  create_agent()
  ├── Model:   groq:openai/gpt-oss-120b
  ├── Tools:   web_search (Tavily)
  ├── Prompt:  "You are a personal chef..."
  └── Memory:  InMemorySaver + thread_id
        │
        ▼
  Agent calls web_search(query) — may issue multiple queries
        │
        ▼
  Tavily returns live, LLM-friendly web results
        │
        ▼
  Agent synthesises recipe suggestions + full instructions
        │
        ▼
  Streamed response to user
```

**Sample interaction:**
```
User:  "I have leftover chicken and rice. What can I make?"
Agent: → web_search("chicken and rice leftover recipes")
       → web_search("quick meals with cooked chicken and rice")
       → Returns: Chicken Fried Rice, One-Pot Casserole, Asian-inspired skillet

User:  "Give me full instructions for the first recipe"
Agent: → Remembers conversation → gives complete step-by-step recipe
```

---

## 📚 Module 2 — Advanced Agent

> **Full lesson notes:** [`notebooks/module-2/README.md`](notebooks/module-2/README.md)

### What We Learned

#### Lesson 2.1 — Model Context Protocol (MCP)
- MCP is an open protocol (by Anthropic) standardising how LLM apps connect to tools & data sources
- **USB analogy:** MCP is to AI tools what USB is to hardware — a universal plug
- Build your own MCP server with `@mcp.tool`, `@mcp.resource`, `@mcp.prompt` decorators
- Connect to any public MCP server from the community library
- MCP servers are reusable across any MCP-compatible app (agents, IDEs, chatbots)

#### Lesson 2.2 — Context & State
- **Runtime context:** typed, immutable data passed to the agent at invocation time (credentials, user IDs)
- **Custom state:** extend `AgentState` with your own fields; tools update state via `Command(update={...})`
- Context = input-only; State = mutable memory tracked across the agent's execution

#### Lesson 2.3 — Multi-Agent Systems
- **Supervisor pattern:** one orchestrator agent delegates sub-tasks to specialist agents
- Each specialist has a focused role and a narrow tool set
- The supervisor routes based on the task, collects results, and synthesises a final response
- Parallelism: multiple specialists can run simultaneously for independent sub-tasks

#### Lesson 2.4 — Project: Destination Wedding Planner
- Multi-agent system: Supervisor → Flight Finder + Venue Scout + Playlist Curator
- Supervisor delegates, waits for all three, then produces a final itinerary
- Uses: MCP for Spotify playlist tool, custom state to carry wedding details, Tavily for flights/venues

---

## 📚 Module 3 — Production-Ready Agent

> **Full lesson notes:** [`notebooks/module-3/README.md`](notebooks/module-3/README.md)

### What We Learned

#### Lesson 3.1 — What is Middleware?
- Middleware sits between user input and the LLM, intercepting every model call
- Defined with `@wrap_model_call` — receives `ModelRequest`, returns `ModelResponse`
- Enables cross-cutting concerns: logging, guardrails, prompt injection, tool gating

#### Lesson 3.2 — Managing Long Conversations
- Agents accumulate messages → eventually exceeds the model's context window
- Solutions: trim oldest messages, summarise history, filter by role/type
- `trim_messages()` from LangChain applies a token budget to the message list

#### Lesson 3.3 — Human In The Loop (HITL)
- Pause agent execution mid-run and wait for human approval before continuing
- `HumanInTheLoopMiddleware` intercepts specified tool calls and raises an interrupt
- Resume with `agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), ...)`

#### Lesson 3.4 — Dynamic Agents
- **Dynamic prompts:** swap system prompt at runtime based on context (user tier, locale, mode)
- **Dynamic tools:** gate which tools are available per invocation (permissions, feature flags)
- Both use middleware — no agent rebuild required; behaviour changes per request

#### Lesson 3.5 — Project: Email Assistant
- Combines all Module 3 skills: authentication state, HITL on `send_email`, dynamic prompt swap
- Three middleware layers stacked: tool gate → prompt swap → HITL approval
- Deployed to LangGraph Studio

---

## ⚙️ Setup

### Prerequisites
- Python 3.12–3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/marehman-exe/LangChain.git
cd "LangChain/Introduction to LangChain - Python by LangChain"

# 2. Create your .env from the template
cp example.env .env
# Then edit .env and fill in your API keys

# 3. Install dependencies
uv sync
```

### Environment Variables

Copy `example.env` → `.env` and fill in:

```env
# Required for all modules
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional — LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=lca-lc-foundation

# Optional — provider alternatives (Module 1 Lesson 1 only)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API keys:
- **Groq** (free): https://console.groq.com
- **Tavily** (free): https://tavily.com
- **LangSmith** (optional, free): https://smith.langchain.com

### Verify Setup

```bash
# PowerShell (Windows)
$env:PYTHONUTF8 = "1"; uv run python env_utils.py
```

### Run Notebooks

```bash
uv run jupyter lab
```

### Run LangGraph Studio (Module 1 & 3 projects)

```bash
# From notebooks/module-1 or notebooks/module-3
cd notebooks/module-1
uv run langgraph dev
# Opens: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

---

## 🔑 Key Concepts — Quick Reference

| Concept | What It Is | Module |
|---------|-----------|--------|
| `init_chat_model()` | Provider-agnostic model initialisation | 1 |
| `create_agent()` | Core LangGraph agent abstraction | All |
| `temperature` | Output randomness (0 = deterministic, 1+ = creative) | 1 |
| `response_format` | Pydantic schema for typed structured output | 1 |
| `agent.stream()` | Stream tokens to reduce perceived latency | 1 |
| `@tool` | Turns a Python function into an agent-callable tool | 1 |
| ReAct pattern | Reason → Act → Observe loop driving tool-using agents | 1 |
| `ToolMessage` | Carries a tool's return value back to the agent | 1 |
| Tavily Search | Real-time web search API for up-to-date information | 1 |
| LangSmith tracing | Visualise full runs: latency, tokens, tool calls | 1 |
| `InMemorySaver` | Checkpointer persisting agent state across invocations | 1 |
| Thread ID | Groups multiple invocations into one conversation | 1 |
| Base64 encoding | Converts binary media to text for API transmission | 1 |
| Multimodal messages | Content blocks combining text + image/audio | 1 |
| `langgraph.json` | Config file pointing LangGraph Studio to your agent | 1 |
| `langgraph dev` | Starts local LangGraph Studio with chat UI | 1 |
| MCP | Open protocol standardising LLM-to-tool connections | 2 |
| Runtime context | Immutable typed data injected at invocation time | 2 |
| Custom state | Extend `AgentState` with own fields; tools update via `Command` | 2 |
| Supervisor pattern | Orchestrator delegating to specialist agents | 2 |
| `@wrap_model_call` | Middleware decorator intercepting every model call | 3 |
| `trim_messages()` | Trims conversation history to a token budget | 3 |
| HITL | Human-in-the-loop: pause execution for human approval | 3 |
| Dynamic prompts | Swap system prompt per request via middleware | 3 |
| Dynamic tools | Gate available tools per request via middleware | 3 |

---

## 🛠️ Tech Stack

![LangChain](https://img.shields.io/badge/LangChain-≥1.3.10-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-≥1.0.3-blue?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-free_tier-orange?style=flat-square)
![Tavily](https://img.shields.io/badge/Tavily-search_API-green?style=flat-square)
![LangSmith](https://img.shields.io/badge/LangSmith-tracing-purple?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.12--3.13-yellow?style=flat-square)

---

## 📌 Related Projects

Built as standalone portfolio projects from this course:

| Project | Description | Repo |
|---------|-------------|------|
| Personal Chef Agent | Tool-using chef that searches recipes from leftover ingredients | [personal-chef-agent](https://github.com/marehman-exe/personal-chef-agent) |
| Wedding Planner Agent | Multi-agent coordinator for destination wedding planning | [wedding-planner-agent](https://github.com/marehman-exe/wedding-planner-agent) |

---

*Course: LangChain Academy — Introduction to LangChain (Python)*  
*Learner notes by [@marehman-exe](https://github.com/marehman-exe)*
