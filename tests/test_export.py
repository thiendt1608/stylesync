"""Phase 6 export/parity helpers, on synthetic rows only (no model, no server)."""
from pathlib import Path

import yaml

from stylesync.export import parity
from stylesync.neutralize.run import append_jsonl


def row(i: int, text: str) -> dict:
    return {"id": f"x-{i}", "prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": text}]}


def test_chrf_bounds_and_order():
    assert parity.chrf("mai m rảnh kh", "mai m rảnh kh") == 100.0
    assert parity.chrf("abc", "xyz") == 0.0
    close, far = parity.chrf("mai m rảnh k", "mai m rảnh kh"), parity.chrf("mai rảnh", "mai m rảnh kh")
    assert 0 < far < close < 100
    assert parity.chrf("a b", "ab") == 100.0  # spaces are ignored, as in sacrebleu


def test_sample_is_fixed_and_only_rows_with_a_reference():
    test = [row(i, "t") for i in range(10)]
    ids = {f"x-{i}" for i in range(8)}
    a, b = parity.sample(test, ids, 3, seed=7), parity.sample(test, ids, 3, seed=7)
    assert a == b and len(a) == 3 and all(r["id"] in ids for r in a)
    assert [r["id"] for r in a] == sorted((r["id"] for r in a), key=lambda i: int(i.split("-")[1]))
    assert len(parity.sample(test, ids, None, seed=7)) == 8


def test_request_masks_only_when_asked():
    r = row(1, "Sửa `get_user` nhé.")
    turns, spans = parity.request(r, masked=False)
    assert turns == r["prompt"] and spans == []
    turns, spans = parity.request(r, masked=True)
    assert turns[-1]["content"] == "Sửa <CODE1> nhé." and spans == ["`get_user`"] and turns[0] == r["prompt"][0]


def test_leading_counts_bos_pieces():
    toks = [{"id": 2, "piece": "<bos>"}, {"id": 2, "piece": "<bos>"}, {"id": 105, "piece": "<|turn>"}]
    assert parity.leading(toks, "<bos>") == 2 and parity.leading(toks[1:], "<bos>") == 1
    assert parity.leading(toks[2:], "<bos>") == 0 and parity.leading([], "<bos>") == 0


def test_server_args_per_device(tmp_path):
    cfg = {"base_gguf": "b.gguf", "lora_gguf": "l.gguf",
           "server": {"bin": "llama-server", "port": 1, "ctx": 2048, "threads": 4}}
    metal, cpu = parity.server_args(cfg, tmp_path, "metal"), parity.server_args(cfg, tmp_path, "cpu")
    assert metal[metal.index("--lora") + 1] == str(tmp_path / "l.gguf") and metal[-2:] == ["-ngl", "99"]
    assert cpu[cpu.index("-ngl") + 1] == "0" and cpu[-2:] == ["-t", "4"]
    assert metal[metal.index("--host") + 1] == "127.0.0.1"


def out(i: int, text: str, prompt_n: int = 50, bos: int = 1) -> dict:
    return {"id": f"x-{i}", "output": text, "wall_s": 0.1 * (i + 1), "prompt_n": prompt_n, "template_n": 50,
            "bos": bos, "predicted_n": 10, "predicted_ms": 200.0, "hit_max_tokens": False}


def test_summarize_parity_guard_and_bos():
    test = {"x-0": row(0, "Sửa `get_user` nhé."), "x-1": row(1, "Mai bạn rảnh không?")}
    ref = {"x-0": "sửa `get_user` nha", "x-1": "mai m rảnh kh"}
    rows = [out(0, "sửa get_user nha", prompt_n=51), out(1, "mai m rảnh kh")]
    s = parity.summarize(rows, test, ref)
    assert s["n"] == 2 and s["exact"] == 0.5 and 50 < s["chrf"] < 100
    assert s["guard"] == 0.5 and s["guard_ref"] == 1.0  # the GGUF output lost the backticks
    assert s["hard_tokens"] == 1.0  # the span's text is still there
    assert s["bos_ok"] == 0.5  # the server evaluated a token the template did not have
    assert s["decode_tok_s"] == 50.0 and s["wall_p90"] == 0.2
    assert parity.summarize([out(1, "mai m rảnh kh", bos=2)], test, ref)["bos_ok"] == 0.0


def test_report_lists_each_set_and_device_it_has(tmp_path: Path):
    append_jsonl(tmp_path / "t.jsonl", [row(0, "Mai bạn rảnh không?")])
    append_jsonl(tmp_path / "r.jsonl", [{"id": "x-0", "output": "mai m rảnh kh"}])
    cfg = yaml.safe_load("""
exp: e
base_gguf: b.gguf
lora_gguf: l.gguf
sets: {chat: {test: t.jsonl, reference: r.jsonl}}
outputs_dir: out
report: report.md
""")
    append_jsonl(parity.out_path(cfg, tmp_path, "chat", "metal"), [out(0, "mai m rảnh kh")])
    text = parity.report(cfg, tmp_path).read_text(encoding="utf-8")
    assert "| chat | metal | 1 | 100% | 100.0 |" in text and "| chat | cpu |" not in text
    assert "rảnh" not in text  # aggregates only


def test_install_copies_once(tmp_path: Path):
    from stylesync import llama

    (tmp_path / "b.gguf").write_bytes(b"base")
    (tmp_path / "l.gguf").write_bytes(b"lora")
    dest = tmp_path / "cache"
    s = llama.Settings(url="http://127.0.0.1:1", model_dir=dest, base=dest / "base.gguf", lora=dest / "lora.gguf",
                       server_bin="llama-server")
    cfg = {"base_gguf": "b.gguf", "lora_gguf": "l.gguf"}
    assert parity.install(cfg, tmp_path, s) == [s.base, s.lora] and s.lora.read_bytes() == b"lora"
    mtime = s.base.stat().st_mtime_ns
    parity.install(cfg, tmp_path, s)
    assert s.base.stat().st_mtime_ns == mtime  # same size: left alone
