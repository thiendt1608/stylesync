"""Phase 7: the shared llama-server backend (plan section 10.2).

One llama-server on 127.0.0.1 serves every Claude Code session: base GGUF + the stylesync LoRA GGUF. The first
MCP server that finds nothing listening starts it detached, and later sessions reuse it. Model files live in
`platformdirs.user_cache_dir("stylesync")` (`stylesync-export install` puts them there); environment variables
override every path.
"""
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
import platformdirs

BASE_NAME = "gemma-4-E2B-it-Q4_K_M.gguf"
LORA_NAME = "stylesync-lora.gguf"


class BackendError(RuntimeError):
    """The backend cannot serve a rewrite; the caller falls back to the neutral text."""


class Truncated(BackendError):
    """The rewrite hit max_tokens, so part of the draft is missing from it."""


@dataclass(frozen=True)
class Settings:
    url: str
    model_dir: Path
    base: Path
    lora: Path
    server_bin: str

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = Path(os.environ.get("STYLESYNC_MODEL_DIR") or platformdirs.user_cache_dir("stylesync"))
        return cls(url=os.environ.get("STYLESYNC_LLAMA_URL", "http://127.0.0.1:18080").rstrip("/"),
                   model_dir=model_dir,
                   base=Path(os.environ.get("STYLESYNC_BASE_GGUF") or model_dir / BASE_NAME),
                   lora=Path(os.environ.get("STYLESYNC_LORA_GGUF") or model_dir / LORA_NAME),
                   server_bin=os.environ.get("STYLESYNC_LLAMA_SERVER", "llama-server"))


def server_args(s: Settings) -> list[str]:
    """Greedy rewrites need nothing from the command line; -ngl 99 offloads to Metal/CUDA and is ignored on CPU builds."""
    port = urlparse(s.url).port or 80
    return [s.server_bin, "-m", str(s.base), "--lora", str(s.lora), "--host", "127.0.0.1", "--port", str(port),
            "-c", "2048", "-np", "1", "--reasoning", "off", "-ngl", "99"]


def healthy(client: httpx.Client) -> bool:
    try:
        return client.get("/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def listening(client: httpx.Client) -> bool:
    """Something answers on the port, even a 503: llama-server binds before it loads the model."""
    try:
        client.get("/health", timeout=3)
        return True
    except httpx.HTTPError:
        return False


def adapter_loaded(client: httpx.Client, s: Settings) -> bool:
    """A server on the port that was started without our LoRA (a Phase 0 bench, say) must not be used."""
    try:
        adapters = client.get("/lora-adapters", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        return False
    return any(Path(a.get("path", "")).name == s.lora.name and a.get("scale", 0) > 0 for a in adapters)


def spawn(s: Settings) -> subprocess.Popen:
    """Start llama-server detached from this process (it outlives the MCP session), never on our stdio."""
    s.model_dir.mkdir(parents=True, exist_ok=True)
    log = (s.model_dir / "llama-server.log").open("a", encoding="utf-8")
    if sys.platform == "win32":
        detach = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        detach = {"start_new_session": True}
    return subprocess.Popen(server_args(s), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **detach)


def ensure(client: httpx.Client, s: Settings, timeout_s: float = 90) -> bool:
    """Make sure a llama-server with the stylesync adapter answers at s.url. Returns True if this call started it."""
    if healthy(client):
        if not adapter_loaded(client, s):
            raise BackendError(f"the server at {s.url} has no {s.lora.name} adapter loaded")
        return False
    host = urlparse(s.url).hostname
    if host not in ("127.0.0.1", "localhost"):
        raise BackendError(f"no llama-server at {s.url}, and only a local one is started automatically")
    missing = [str(p) for p in (s.base, s.lora) if not p.is_file()]
    if missing:
        raise BackendError(f"model file missing: {', '.join(missing)} (run `stylesync-export install`)")
    try:
        proc = spawn(s)
    except OSError as e:
        raise BackendError(f"cannot start {s.server_bin}: {e}") from e
    log = s.model_dir / "llama-server.log"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if healthy(client):  # ours, or one another session started at the same time (ours then exits: port taken)
            if not adapter_loaded(client, s):
                raise BackendError(f"the server at {s.url} has no {s.lora.name} adapter loaded")
            return True
        if proc.poll() is not None and not listening(client):  # no other session's server is loading either
            raise BackendError(f"llama-server exited with {proc.returncode}, see {log}")
        time.sleep(0.25)
    raise BackendError(f"llama-server did not become ready in {timeout_s:.0f}s, see {log}")


def chat(client: httpx.Client, system: str, text: str, max_tokens: int = 512) -> tuple[str, float]:
    """Greedy rewrite, as in Phase 5 eval and Phase 6 parity. Returns (text, seconds); a cut-off output raises."""
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.perf_counter()
    r = client.post("/v1/chat/completions", json=body, timeout=60)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise Truncated(f"output hit max_tokens={max_tokens}")
    return choice["message"]["content"].strip(), time.perf_counter() - t
