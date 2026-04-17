# ── Base Image ─────────────────────────────────────────────── #
FROM python:3.12-slim

# ── Set Workdir ────────────────────────────────────────────── #
WORKDIR /app

# ── Install system deps ────────────────────────────────────── #
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python deps ────────────────────────────────────── #
# Copy requirements first for caching
COPY files/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────── #
COPY src ./src

# ── Default command (for orchestrator) ─────────────────────── #
# Railway sets PORT env var; fallback to 8000 for local dev
CMD python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
