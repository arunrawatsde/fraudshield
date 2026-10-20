"""Langfuse tracing. Every agent run becomes one trace (LLM calls, tool calls, tokens, latency).

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (LANGFUSE_HOST picks the
region or a self-hosted server). Without keys, tracing is skipped and nothing else changes.
"""
import sys

from fraudshield.config import get_settings


def langfuse_callbacks() -> list:
    s = get_settings()
    if not s.use_langfuse:
        return []
    if s.langfuse_public_key.startswith("sk-") or s.langfuse_secret_key.startswith("pk-"):
        print("[tracing] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY look swapped (public starts with pk-, "
              "secret with sk-); Langfuse will reject them with 401.", file=sys.stderr)
    try:
        from langfuse.langchain import CallbackHandler      # Langfuse SDK v3
    except ImportError:
        from langfuse.callback import CallbackHandler       # Langfuse SDK v2
    return [CallbackHandler()]


def flush() -> None:
    """Send buffered events now (short-lived scripts like the eval would otherwise drop them)."""
    if not get_settings().use_langfuse:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except ImportError:
        pass
