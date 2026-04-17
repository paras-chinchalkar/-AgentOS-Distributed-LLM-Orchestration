# AgentOS Runtime 🤖⚡

**Distributed LLM Agent Orchestration OS** — A production-grade runtime for multi-agent AI systems, built on FastAPI, LangGraph, Redis Streams, and PostgreSQL.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Runtime API                        │
│  REST /api/v1/workflows  │  WebSocket /ws/events (Live telemetry) │
└───────────────┬──────────────────────────┬───────────────────────┘
                │                          │
        ┌───────▼────────┐        ┌────────▼────────┐
        │  Lifecycle Mgr │        │  Observability   │
        │  (spawn/retry) │        │  Dashboard (UI)  │
        └───────┬────────┘        └─────────────────┘
                │
     ┌──────────▼──────────┐
     │   Redis Streams     │  ←── Inter-agent message bus
     │   Message Bus       │
     └──────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │   Background Worker │  ←── Consumes tasks, runs LangGraph
     └──────────┬──────────┘
                │
     ┌──────────▼──────────────────────────────┐
     │         LangGraph Agent Pipeline         │
     │  Researcher → Critic → Summarizer        │
     │         (Groq LLM backend)               │
     └──────────┬──────────────────────────────┘
                │
     ┌──────────▼──────────┐
     │     PostgreSQL       │  ←── State persistence
     │  Workflow + Agent DB │
     └─────────────────────┘
```

## Resume-Level Features

| Feature | Implementation |
|---|---|
| Distributed task scheduling | Redis Streams consumer groups |
| Agent lifecycle management | Dynamic spawn, retry (3x), status tracking |
| State persistence | PostgreSQL via async SQLAlchemy |
| Inter-agent communication | Redis Pub/Sub + Streams |
| Real-time observability | WebSocket telemetry → live dashboard |
| Multi-agent LLM workflow | LangGraph (Researcher → Critic → Summarizer) |
| LLM backend | Groq (llama-3.3-70b-versatile) |

---

## Quick Start

### Option A: Docker (recommended for local development)

```bash
# 1. Add your Groq API key to .env
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 2. Start everything
docker compose up -d

# 3. Open dashboard
open http://localhost:8000
```

### Option B: Local (requires Redis + PostgreSQL running)

```bash
# Install deps
uv sync

# Set env vars
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run
python -m src.api.main
```

---

## Production Deployment

### Railway (Recommended for Cloud)

Deploy to Railway.app in 5 minutes:

```bash
# 1. Push to GitHub
git add .
git commit -m "chore: prepare for Railway deployment"
git push origin main

# 2. Follow DEPLOYMENT.md for Railway setup
# - Add PostgreSQL & Redis plugins
# - Set environment variables (use .env.example as reference)
# - Railway auto-deploys from GitHub

# 3. Access your deployed app
# https://your-railway-url
```

For detailed instructions, see **[DEPLOYMENT.md](./DEPLOYMENT.md)**.

**Key points:**
- ✅ PostgreSQL & Redis auto-provisioned via Railway plugins
- ✅ Auto-scaling supported (set replica count in Railway UI)
- ✅ Health checks enabled at `/api/v1/health`
- ✅ Environment variables managed securely in Railway dashboard

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/workflows` | Launch a new agent workflow |
| `GET`  | `/api/v1/workflows` | List all workflow runs |
| `GET`  | `/api/v1/workflows/{id}` | Get workflow details |
| `GET`  | `/api/v1/workflows/{id}/agents` | Get agent execution records |
| `POST` | `/api/v1/workflows/{id}/retry` | Retry a failed workflow |
| `GET`  | `/api/v1/agents` | List registered agent plugins |
| `WS`   | `/ws/events` | Live event stream (WebSocket) |
| `GET`  | `/api/v1/health` | Health check |
| `GET`  | `/` | Interactive Dashboard |
| `GET`  | `/docs` | Swagger API docs |

---

## Configuration

All configuration is environment-driven. See `.env.example` for all available options:

```env
# LLM Backend
GROQ_API_KEY=your-key-here
GROQ_MODEL=llama-3.3-70b-versatile

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db

# Message Bus
REDIS_URL=redis://host:6379

# App
PORT=8000
LOG_LEVEL=INFO
```

**Local Development:** Use `.env.example` as a template  
**Production (Railway):** Set variables in Railway dashboard

---

## Tech Stack

- **Python 3.12** + **FastAPI** — Async REST + WebSocket API
- **LangGraph** — Agent workflow state machine
- **LangChain-Groq** — LLM integration (llama-3.3-70b)
- **Redis Streams** — Distributed message bus
- **PostgreSQL** + **SQLAlchemy** — Async state persistence
- **Docker Compose** — Full local orchestration
- **Railway** — Cloud deployment platform

---

## Development

### Project Structure

```
src/
├── api/              # REST + WebSocket endpoints
│   ├── main.py      # FastAPI app with lifespan
│   ├── routes.py    # Workflow & agent routes
│   └── telemetry.py # WebSocket event streaming
├── core/            # Core business logic
│   ├── config.py    # Environment settings
│   ├── db.py        # SQLAlchemy models
│   ├── message_bus.py     # Redis Streams
│   └── db.py        # Database initialization
├── runtime/         # Agent execution engine
│   ├── graph.py           # Dynamic LangGraph builder
│   ├── lifecycle.py       # Workflow state management
│   ├── worker.py          # Background task processor
│   └── agents/            # Agent implementations
│       ├── researcher.py
│       ├── critic.py
│       ├── summarizer.py
│       └── ...
└── ui/              # Dashboard (HTML/JavaScript)
    └── dashboard.html
```

### Adding New Agents

1. Create `src/runtime/agents/your_agent.py`:

```python
METADATA = {
    "name": "your_agent",
    "label": "Your Agent",
    "emoji": "🔧",
    "role": "Your Description",
    "output_key": "your_output",
}

def node_fn(state: dict) -> dict:
    # Your agent logic here
    return {**state, "your_output": "result"}
```

2. The agent is auto-discovered via `src/runtime/agents/__init__.py`
3. Use in workflows: add `your_agent` to topology

### Running Tests

```bash
python -m pytest
```

---

## Troubleshooting

### Summary not showing in dashboard?

1. **Hard refresh browser:** `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. **Check API:** `curl http://localhost:8000/api/v1/workflows`
3. **Verify backend:** Workflow should have `final_output` and agent should have `output`
4. **Check logs:** `docker compose logs orchestrator`

### Database connection errors?

```bash
# For local dev, ensure PostgreSQL is running
docker compose up postgres -d

# For Railway, verify DATABASE_URL in Variables tab
```

### Redis connection errors?

```bash
# For local dev, ensure Redis is running
docker compose up redis -d

# For Railway, verify REDIS_URL in Variables tab
```

### LLM API key not working?

1. Verify key format (should start with `gsk_` for Groq)
2. Check Groq console: https://console.groq.com/keys
3. Ensure key is set in `.env` (local) or Railway Variables (production)

---

## Security Best Practices

⚠️ **Never commit `.env` with secrets.** It's in `.gitignore` for your safety.

1. ✅ Use `.env.example` as template (no secrets)
2. ✅ Use Railway's **Variables** for production secrets
3. ✅ Rotate API keys regularly
4. ✅ Use HTTPS only (Railway enforces this)
5. ✅ Enable **encryption** for sensitive Railway Variables

---

## License

MIT

---

## Support

- 📖 **Docs:** [DEPLOYMENT.md](./DEPLOYMENT.md) for Railway deployment
- 🐛 **Issues:** Check existing issues or create a new one
- 🤖 **LLM Backend:** [Groq Console](https://console.groq.com)
- 📚 **Framework Docs:**
  - [FastAPI](https://fastapi.tiangolo.com)
  - [LangGraph](https://python.langchain.com/docs/langgraph)
  - [Railway](https://docs.railway.app)

#   - A g e n t O S - D i s t r i b u t e d - L L M - O r c h e s t r a t i o n  
 # -AgentOS-Distributed-LLM-Orchestration
