"""
Database — async SQLAlchemy with auto-fallback to SQLite when PostgreSQL is unavailable.
"""
import uuid
import enum
import json
from datetime import datetime

from sqlalchemy import String, DateTime, Enum as SAEnum, Text, Integer, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import text

from src.core.config import settings


class Base(DeclarativeBase):
    pass


class AgentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[AgentStatus] = mapped_column(SAEnum(AgentStatus), default=AgentStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    final_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Stores the topology as a JSON string, e.g. '[["researcher"],["critic","fact_checker"],["aggregator"]]'
    topology: Mapped[str | None] = mapped_column(Text, nullable=True)

    agents: Mapped[list["AgentRecord"]] = relationship(
        "AgentRecord", back_populates="workflow", lazy="select"
    )

    def get_topology(self) -> list[list[str]] | None:
        """Deserialise topology JSON, returning None if not set."""
        if not self.topology:
            return None
        try:
            return json.loads(self.topology)
        except Exception:
            return None


class AgentRecord(Base):
    __tablename__ = "agent_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id: Mapped[str] = mapped_column(String, ForeignKey("workflow_runs.id"))
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[AgentStatus] = mapped_column(SAEnum(AgentStatus), default=AgentStatus.PENDING)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    workflow: Mapped["WorkflowRun"] = relationship("WorkflowRun", back_populates="agents")


# ── Engine: try PostgreSQL, fall back to SQLite ───────────────────────────────
import logging
logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def _build_engine(url: str):
    return create_async_engine(url, echo=False)


async def _try_connect(url: str):
    eng = _build_engine(url)
    async with eng.connect() as conn:
        pass  # raises if unreachable
    return eng


async def _resolve_engine():
    global _engine, _session_factory
    if _engine is not None:
        return

    try:
        _engine = await _try_connect(settings.database_url)
        logger.info("Connected to PostgreSQL.")
    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s) — falling back to SQLite.", exc)
        _engine = _build_engine("sqlite+aiosqlite:///./agentos.db")

    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_db() -> AsyncSession:
    await _resolve_engine()
    async with _session_factory() as session:
        yield session


async def init_db() -> None:
    # Install aiosqlite if using SQLite
    await _resolve_engine()
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # SQLite doesn't auto-migrate schema changes; add missing columns safely.
        # This keeps older `agentos.db` files compatible after model updates.
        if conn.dialect.name == "sqlite":
            rows = await conn.execute(text("PRAGMA table_info(workflow_runs)"))
            cols = {r[1] for r in rows.fetchall()}  # (cid, name, type, notnull, dflt_value, pk)
            if "topology" not in cols:
                await conn.execute(text("ALTER TABLE workflow_runs ADD COLUMN topology TEXT"))


def get_session():
    """Return a new async session. Always called after init_db() so _session_factory is set."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _session_factory()
