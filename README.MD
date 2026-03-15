# Web Research Agent

A LangGraph-powered research agent that takes a question, searches the web across multiple queries, and returns a synthesized answer with citations.

Built with Gemini 2.0 Flash, DuckDuckGo search, FastAPI, and a plain HTML frontend.

---

## Features

- **Multi-search loop** — agent refines and repeats searches until it has enough information (max 3)
- **Automatic citations** — sources are numbered and tracked across all searches
- **FastAPI backend** — single `/research` endpoint, easy to extend
- **Minimal frontend** — plain HTML, no framework, no build step

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | Gemini 2.0 Flash (via Google GenAI SDK) |
| Web search | DuckDuckGo (`ddgs`) |
| API server | FastAPI + Uvicorn |
| Frontend | Plain HTML / CSS / JS |

---

## Project Structure

```
web-research-agent/
├── agent.py      # LangGraph agent — nodes, edges, state, tools
├── api.py        # FastAPI wrapper around the agent
├── index.html    # Frontend UI
├── .env          # API keys (never commit this)
├── .gitignore
└── README.md
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <your-repo-url>
cd web-research-agent

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

**2. Install dependencies**

```bash
pip install google-genai langgraph fastapi uvicorn ddgs python-dotenv
```

**3. Add your API key**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

---

## Running

**Start the API server:**

```bash
uvicorn api:api --reload
```

Server runs at `http://127.0.0.1:8000`.

**Open the frontend:**

Open `index.html` directly in your browser (double-click it).

---

## API

### `POST /research`

**Request:**
```json
{
  "question": "What LLM models did Anthropic release in 2025?"
}
```

**Response:**
```json
{
  "answer": "In 2025, Anthropic released...",
  "sources": [
    "https://...",
    "https://..."
  ],
  "search_count": 2
}
```

### `GET /health`

Returns `{ "status": "ok" }`. Use to verify the server is running.

---

## How It Works

```
User question
     ↓
call_model  →  Gemini decides whether to search or answer
     ↓
should_search?
     ├── YES → run_search (DuckDuckGo, track URLs) → loop back to call_model
     └── NO  → prepare_answer → return final answer + sources
```

The agent loops up to **3 times**, using a different search query each iteration. After 3 searches, `mode=NONE` forces the model to synthesize from what it has rather than searching again.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |

---

## .gitignore

Make sure your `.gitignore` includes:

```
.env
venv/
__pycache__/
```

---

## License

MIT