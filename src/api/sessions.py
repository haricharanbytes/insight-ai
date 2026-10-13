"""In-memory, per-session storage. Sessions are isolated from each other.

Nothing is persisted: restarting the server clears every session.
"""
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache

from config import get_settings
from src.data.loader import LoadResult

MAX_SESSIONS = 50
MAX_RESULTS = 50


@dataclass
class Session:
    id: str
    created_at: float
    last_access: float
    load: LoadResult | None = None
    profile: dict | None = None
    pending: tuple[str, bytes] | None = None  # (filename, bytes) awaiting a sheet choice
    results: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    history: list = field(default_factory=list)  # used by the agent in Step 6
    report_items: list = field(default_factory=list)  # [{question, answer, unverified, tool_steps}]

    def set_dataset(self, load: LoadResult, profile: dict) -> None:
        self.load = load
        self.profile = profile
        self.pending = None
        self.results.clear()
        self.history.clear()

    def clear_dataset(self) -> None:
        self.load = None
        self.profile = None
        self.pending = None
        self.results.clear()
        self.history.clear()
        self.report_items.clear()

    def add_result(self, output: dict) -> str:
        result_id = uuid.uuid4().hex[:10]
        self.results[result_id] = {"result_id": result_id, "created_at": time.time(), "output": output}
        while len(self.results) > MAX_RESULTS:
            self.results.popitem(last=False)
        return result_id


class SessionStore:
    def __init__(self, ttl_seconds: float, max_sessions: int = MAX_SESSIONS):
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = time.monotonic()
        dead = [sid for sid, s in self._sessions.items() if now - s.last_access > self._ttl]
        for sid in dead:
            del self._sessions[sid]

    def create(self) -> Session:
        with self._lock:
            self._purge()
            while len(self._sessions) >= self._max:  # evict the least recently used
                oldest = min(self._sessions.values(), key=lambda s: s.last_access)
                del self._sessions[oldest.id]
            now = time.monotonic()
            session = Session(id=uuid.uuid4().hex, created_at=now, last_access=now)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._purge()
            session = self._sessions.get(session_id)
            if session:
                session.last_access = time.monotonic()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


@lru_cache
def get_store() -> SessionStore:
    return SessionStore(ttl_seconds=get_settings().session_ttl_minutes * 60)