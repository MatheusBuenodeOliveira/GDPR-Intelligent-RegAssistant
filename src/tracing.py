"""Optional LangSmith tracing integration.

Provides a lightweight context manager `span(name, metadata)` that creates a run
if LangSmith client is available; otherwise it is a no-op. This keeps core logic
independent of tracing availability.
"""
from __future__ import annotations
import contextlib

try:  # pragma: no cover - optional dependency
    from langsmith import Client  # type: ignore
    _CLIENT = Client()
except Exception:  # pragma: no cover
    _CLIENT = None

@contextlib.contextmanager
def span(name: str, metadata: dict | None = None):
    if _CLIENT is None:
        yield None
        return
    run = None
    try:
        run = _CLIENT.create_run(name=name, inputs=metadata or {}, run_type="chain")
    except Exception:
        # Fail gracefully; continue without tracing
        yield None
        return
    error: str | None = None
    try:
        yield run
    except Exception as e:  # capture error for run update
        error = str(e)
        raise
    finally:
        try:
            if error:
                _CLIENT.update_run(run_id=run.id, error=error)
            else:
                _CLIENT.update_run(run_id=run.id, outputs={"status": "ok"})
        except Exception:
            pass
