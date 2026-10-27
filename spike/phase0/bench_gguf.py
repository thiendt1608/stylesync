"""Phase 0: run the test set through llama-server and record latency, memory and output checks.

Usage: python spike/phase0/bench_gguf.py <model.gguf> <name> <metal|cpu>
"""
import subprocess
import sys
import threading
import time

import httpx
import psutil

from common import check, load_cases, messages, save, summarize

PORT = 18080


def main() -> None:
    model, name, device = sys.argv[1], sys.argv[2], sys.argv[3]
    args = ["llama-server", "-m", model, "--port", str(PORT), "--host", "127.0.0.1",
            "-c", "2048", "--reasoning", "off", "-np", "1"]
    args += ["-ngl", "99"] if device == "metal" else ["-ngl", "0", "-dev", "none", "-t", "4"]
    t0 = time.perf_counter()
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = {"rss": 0}
    stop = threading.Event()

    def sample() -> None:
        p = psutil.Process(proc.pid)
        while not stop.is_set():
            try:
                peak["rss"] = max(peak["rss"], p.memory_info().rss)
            except psutil.Error:
                return
            time.sleep(0.2)

    threading.Thread(target=sample, daemon=True).start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=120)
    try:
        while True:
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited with {proc.returncode}")
            time.sleep(0.2)
        load_s = time.perf_counter() - t0

        rows = []
        for case in load_cases():
            body = {
                "messages": messages(case),
                "temperature": 0,
                "max_tokens": 160,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            s = time.perf_counter()
            r = client.post("/v1/chat/completions", json=body).json()
            wall = time.perf_counter() - s
            out = r["choices"][0]["message"]["content"].strip()
            tm = r.get("timings", {})
            rows.append({
                **case, "output": out, "wall_s": round(wall, 3),
                "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
                "decode_tok_s": (tm["predicted_n"] / tm["predicted_ms"] * 1000) if tm.get("predicted_ms") else None,
                **check(case["text"], out),
            })
    finally:
        stop.set()
        proc.terminate()
        proc.wait(timeout=30)

    save(name, device, rows, summarize(name, device, rows, load_s, peak["rss"]))


if __name__ == "__main__":
    main()
