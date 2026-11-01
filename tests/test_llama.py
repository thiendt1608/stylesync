"""Shared llama-server backend: settings, spawn decision and adapter check, with a fake HTTP server."""
from pathlib import Path

import httpx
import pytest

from stylesync import llama


def settings(tmp_path: Path, url: str = "http://127.0.0.1:18080", files: bool = True) -> llama.Settings:
    base, lora = tmp_path / "base.gguf", tmp_path / "stylesync-lora.gguf"
    if files:
        base.write_bytes(b"b")
        lora.write_bytes(b"l")
    return llama.Settings(url=url, model_dir=tmp_path, base=base, lora=lora, server_bin="llama-server")


def fake_server(state: dict) -> httpx.Client:
    """/health is up when state['up'] (503 while state['loading'] > 0 health checks remain);
    /lora-adapters lists state['adapters']; completions finish with state.get('finish', 'stop')."""
    def handle(req: httpx.Request) -> httpx.Response:
        if state.get("loading", 0) > 0 and req.url.path == "/health":
            state["loading"] -= 1
            if state["loading"] == 0:
                state["up"] = True
            return httpx.Response(503, json={"error": "Loading model"})
        if not state["up"]:
            raise httpx.ConnectError("refused", request=req)
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/lora-adapters":
            return httpx.Response(200, json=state["adapters"])
        if req.url.path == "/v1/chat/completions":
            state["body"] = req.read()
            return httpx.Response(200, json={"choices": [{"message": {"content": " fix r nha \n"},
                                                          "finish_reason": state.get("finish", "stop")}]})
        return httpx.Response(404)
    return httpx.Client(base_url="http://127.0.0.1:18080", transport=httpx.MockTransport(handle))


OURS = [{"id": 0, "path": "/x/stylesync-lora.gguf", "scale": 1.0}]


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("STYLESYNC_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("STYLESYNC_BASE_GGUF", raising=False)
    monkeypatch.setenv("STYLESYNC_LORA_GGUF", str(tmp_path / "other.gguf"))
    s = llama.Settings.from_env()
    assert s.base == tmp_path / llama.BASE_NAME and s.lora == tmp_path / "other.gguf"


def test_server_args_bind_localhost_with_the_lora(tmp_path):
    args = llama.server_args(settings(tmp_path, url="http://127.0.0.1:18099"))
    assert args[args.index("--host") + 1] == "127.0.0.1" and args[args.index("--port") + 1] == "18099"
    assert args[args.index("--lora") + 1].endswith("stylesync-lora.gguf")


def test_running_server_with_our_adapter_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(llama, "spawn", lambda s: pytest.fail("must not spawn"))
    assert llama.ensure(fake_server({"up": True, "adapters": OURS}), settings(tmp_path)) is False


def test_running_server_without_our_adapter_is_refused(tmp_path):
    for adapters in ([], [{"id": 0, "path": "/x/stylesync-lora.gguf", "scale": 0.0}]):
        with pytest.raises(llama.BackendError, match="no stylesync-lora.gguf adapter"):
            llama.ensure(fake_server({"up": True, "adapters": adapters}), settings(tmp_path))


def test_nothing_listening_and_files_missing(tmp_path):
    with pytest.raises(llama.BackendError, match="stylesync-export install"):
        llama.ensure(fake_server({"up": False}), settings(tmp_path, files=False))


def test_remote_url_is_never_spawned(tmp_path):
    with pytest.raises(llama.BackendError, match="only a local one"):
        llama.ensure(fake_server({"up": False}), settings(tmp_path, url="http://10.0.0.5:18080"))


class Proc:
    def __init__(self, code=None):
        self.returncode = code

    def poll(self):
        return self.returncode


def test_spawns_and_waits_until_ready(tmp_path, monkeypatch):
    state = {"up": False, "adapters": OURS}

    def spawn(s):
        state["up"] = True  # the server comes up right after the start
        return Proc()

    monkeypatch.setattr(llama, "spawn", spawn)
    assert llama.ensure(fake_server(state), settings(tmp_path), timeout_s=5) is True


def test_spawned_server_that_dies_is_reported_at_once(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(llama, "spawn", lambda s: Proc(code=1))
    t = time.monotonic()
    with pytest.raises(llama.BackendError, match="exited with 1"):
        llama.ensure(fake_server({"up": False}), settings(tmp_path), timeout_s=30)
    assert time.monotonic() - t < 5  # nothing listens on the port: no point waiting out the timeout


def test_dead_spawn_waits_for_another_sessions_server_that_is_loading(tmp_path, monkeypatch):
    state = {"up": False, "adapters": OURS}

    def spawn(s):
        state["loading"] = 4  # another session's server took the port first and is still loading
        return Proc(code=1)

    monkeypatch.setattr(llama, "spawn", spawn)
    assert llama.ensure(fake_server(state), settings(tmp_path), timeout_s=10) is True


def test_chat_is_greedy_with_the_given_system_prompt():
    import json

    state = {"up": True, "adapters": OURS}
    text, seconds = llama.chat(fake_server(state), "SYS", "Xin chào.")
    body = json.loads(state["body"])
    assert text == "fix r nha" and seconds >= 0
    assert body["temperature"] == 0 and body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_that_hits_max_tokens_raises():
    with pytest.raises(llama.Truncated, match="max_tokens=512"):
        llama.chat(fake_server({"up": True, "adapters": OURS, "finish": "length"}), "SYS", "Xin chào.")

