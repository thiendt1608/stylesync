"""Phase 5 eval: made-up examples only, no model, no network."""
import json
from pathlib import Path

from stylesync.clean.pipeline import write_jsonl
from stylesync.dataset.build import SYSTEM
from stylesync.eval import llm, metrics, owner_set
from stylesync.eval import run as ev
from stylesync.eval.generate import batches, messages

NEUTRAL = ["Tôi đã fix được bug login.", "Ngày mai bạn có rảnh không?", "Anh đã push code lên chưa?",
           "Tại sao lệnh `npm run build` cứ fail?", "Tôi sẽ mời cà phê.", "Em sẽ review cho anh.",
           "Bạn gửi cho tôi file config.yaml nhé.", "Server đang chạy bình thường.", "Tôi không biết nữa.",
           "Mọi người họp lúc 9 giờ nhé."]
MINE = ["fix bug login đc r =))", "mai m rảnh kh", "a push code chưa", "sao `npm run build` fail hoài z",
        "t bao cf", "e review cho", "gửi t file config.yaml nha", "server chạy bth mà", "t kh bt nữa",
        "mn họp 9h nha"]


def ex(i: int, variant: str = "formal", burst: str | None = None) -> dict:
    burst = burst or f"messenger-{i}"
    return {"id": f"{burst}-{variant[0]}", "source": "messenger", "variant": variant, "ts": "2026-01-01T00:00:00",
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": NEUTRAL[i]}],
            "completion": [{"role": "assistant", "content": MINE[i]}]}


def test_messages_swaps_only_the_system_prompt():
    e = ex(0)
    assert messages(e) == e["prompt"]
    swapped = messages(e, "khác")
    assert swapped[0] == {"role": "system", "content": "khác"} and swapped[1:] == e["prompt"][1:]
    assert e["prompt"][0]["content"] == SYSTEM  # the example itself is untouched


def test_batches_sort_by_length_and_keep_every_row():
    rows = ["ccc", "a", "bb", "dddd", "e"]
    out = batches(rows, 2, len)
    assert [len(b) for b in out] == [2, 2, 1]
    assert sorted(x for b in out for x in b) == sorted(rows)
    assert [len(x) for b in out for x in b] == sorted(len(x) for x in rows)


def test_retrieve_one_shot_per_burst_most_similar_last():
    train = [ex(i) for i in range(1, 10)] + [ex(1, "concise", "messenger-1")]
    test = [ex(0) | {"id": "messenger-x-f",
                     "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Mai bạn rảnh không?"}]}]
    shots = llm.retrieve(train, test, k=3)[test[0]["id"]]
    assert len(shots) == 3
    bursts = [s["id"].rsplit("-", 1)[0] for s in shots]
    assert len(set(bursts)) == 3
    assert shots[-1]["id"].startswith("messenger-1-")  # "Ngày mai bạn có rảnh không?" is the closest


def test_fewshot_input_and_keys():
    text = llm.fewshot_input([ex(1)], "Xin chào.")
    assert "<neutral>Ngày mai bạn có rảnh không?</neutral>" in text and "<real>mai m rảnh kh</real>" in text
    assert text.endswith("<neutral>Xin chào.</neutral>")
    cfg = {"model": "m", "prompt_version": "v1"}
    assert llm.judge_key(cfg, "a", "b") != llm.judge_key(cfg, "a", "c")
    assert llm.judge_key(cfg, "a", "b") != llm.fewshot_key(cfg, llm.judge_input("a", "b"))
    assert len(llm.fewshot_key(cfg, "x")) == 24


def test_decoders_reject_bad_shapes():
    assert llm.decode_any({"rewrite": " ok nha "}) == {"rewrite": "ok nha"}
    assert llm.decode_any({"score": 4, "added": False, "lost": True}) == {"score": 4, "added": False, "lost": True}
    assert llm.decode_any({"score": 7, "added": False, "lost": True}) is None
    assert llm.decode_any({"score": 4, "added": "no", "lost": True}) is None
    assert llm.decode_rewrite({"text": "x"}) is None


def test_body_is_strict_json_and_not_stored():
    b = llm.body({"model": "m", "effort": "low", "max_output_tokens": 100}, "sys", "in", "judge", llm.JUDGE_SCHEMA)
    assert b["store"] is False and b["text"]["format"]["strict"] is True and b["input"] == "in"


def test_checks_placeholder_in_source_is_not_a_leak():
    src = "<NAME>, bạn gửi file `config.yaml` cho tôi nhé."
    good = metrics.checks(src, "<NAME> gửi t file `config.yaml` nha")
    assert good == {"hard_tokens": True, "placeholder": True, "guard": True}
    bad = metrics.checks(src, "gửi t file config nha")
    assert bad == {"hard_tokens": False, "placeholder": False, "guard": False}


def test_check_rates_hard_tokens_only_over_inputs_that_have_them():
    r = metrics.check_rates([("Tôi không biết.", "t kh bt"), ("Chạy `make`.", "chạy đi")])
    assert r["hard_token_n"] == 1 and r["hard_tokens"] == 0.0 and r["placeholder"] == 1.0


def test_judge_summary():
    v = [{"score": 5, "added": False, "lost": False}, {"score": 2, "added": True, "lost": False}]
    assert metrics.judge_summary(v) == {"judged": 2, "meaning": 3.5, "added": 0.5, "lost": 0.0, "low": 0.5}
    assert metrics.judge_summary([])["meaning"] is None


def test_gate():
    g = {"style": 0.7, "meaning": 4.3, "added_over_reference": 0.03, "max_meaning_gap": 0.3}
    ft, ref = {"style": 0.8, "meaning": 4.4, "added": 0.1}, {"added": 0.2}
    assert metrics.gate(ft, {"style": 0.2}, {"style": 0.9, "meaning": 4.6}, ref, g)["pass"]
    assert metrics.gate(ft | {"added": 0.24}, {"style": 0.2}, {"style": 0.9, "meaning": 4.6}, ref, g) == {
        "beats_base_style": True, "style_floor": True, "meaning_floor": True, "added_vs_reference": False,
        "meaning_vs_fewshot": True, "pass": False, "loses_to_fewshot_on_both": True}
    lost = metrics.gate(ft, {"style": 0.2}, {"style": 0.9, "meaning": 4.8}, ref, g)
    assert not lost["meaning_vs_fewshot"] and not lost["pass"] and lost["loses_to_fewshot_on_both"]
    unjudged = metrics.gate(ft | {"meaning": None}, {}, {}, {}, g)
    assert not unjudged["pass"] and not unjudged["beats_base_style"] and unjudged["style_floor"]


def test_style_classifier_separates_voices():
    clf = metrics.fit_style(MINE * 3, NEUTRAL * 3)
    assert metrics.style_rate(clf, ["t kh bt z", "mai m rảnh kh"]) == 1.0
    assert metrics.style_rate(clf, ["Tôi không biết nữa.", "Ngày mai bạn có rảnh không?"]) == 0.0
    assert metrics.style_rate(clf, []) == 0.0


def cfg_for(tmp: Path) -> dict:
    cfg = ev.load_cfg(Path("."), "configs/eval.yaml")
    cfg["llm"]["out"] = {"cache": str(tmp / "cache.jsonl"), "batches": str(tmp / "batches.jsonl")}
    cfg |= {"train": str(tmp / "train.jsonl"), "test": str(tmp / "test.jsonl"), "outputs_dir": str(tmp / "outputs"),
            "report": str(tmp / "eval.md")}
    return cfg


def test_judge_requests_skip_neutral_and_dedupe(tmp_path):
    cfg = cfg_for(tmp_path)
    test = [ex(0), ex(1)]
    outs = {"neutral": {e["id"]: llm.source_text(e) for e in test},
            "reference": {e["id"]: llm.target_text(e) for e in test},
            "copy": {e["id"]: llm.target_text(e) for e in test}}  # same text as the reference: no new request
    bodies = ev.judge_requests(cfg, test, outs)
    assert len(bodies) == 2
    assert all(b["instructions"] == llm.JUDGE_SYSTEM for b in bodies.values())


def test_report_end_to_end_has_no_message_text(tmp_path):
    """Fake generated outputs and cached judge verdicts in, aggregate report out; examples stay private."""
    cfg = cfg_for(tmp_path)
    base = Path("/")  # every path in cfg is absolute under tmp_path
    train = [ex(i) for i in range(10)] * 2
    test = [ex(i) for i in range(6)]
    write_jsonl(tmp_path / "train.jsonl", train)
    write_jsonl(tmp_path / "test.jsonl", test)
    write_jsonl(tmp_path / "outputs" / "ft-qwen35-2b-r16-e1.jsonl",
                [{"id": e["id"], "output": llm.target_text(e)} for e in test])
    write_jsonl(tmp_path / "outputs" / "base-qwen35-2b.jsonl",
                [{"id": e["id"], "output": llm.source_text(e)} for e in test])
    cache = []
    for e in test:
        for out in (llm.target_text(e), llm.source_text(e)):
            cache.append({"key": llm.judge_key(cfg["llm"], llm.source_text(e), out), "stop_reason": "completed",
                          "out": {"score": 5, "added": False, "lost": False},
                          "usage": {"input_tokens": 100, "cached_tokens": 0, "output_tokens": 10,
                                    "reasoning_tokens": 0}})
    write_jsonl(tmp_path / "cache.jsonl", cache)

    path = ev.report(cfg, base, test)
    text = path.read_text(encoding="utf-8")
    assert "| ft-qwen35-2b-r16-e1 |" in text and "| base-qwen35-2b |" in text
    assert "Not run yet: base-gemma4-e2b" in text
    assert not any(m in text for m in MINE + NEUTRAL)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    ft = summary["rows"]["ft-qwen35-2b-r16-e1"]
    assert ft["meaning"] == 5.0 and ft["n"] == 6
    assert summary["rows"]["neutral"]["judged"] == 0
    assert summary["gate"]["ft-qwen35-2b-r16-e1"]["beats_base_style"]
    assert not summary["gate"]["ft-qwen35-2b-r16-e1"]["pass"]  # Baseline B has not run, so its rule fails
    assert (tmp_path / "examples.md").exists() and (tmp_path / "blind.md").exists()
    key = json.loads((tmp_path / "blind_key.json").read_text(encoding="utf-8"))
    assert key["system"] == "ft-qwen35-2b-r16-e1" and len(key["key"]) == 6


def test_blind_keeps_a_sheet_the_owner_started(tmp_path):
    cfg = cfg_for(tmp_path)
    test = [ex(i) for i in range(6)]
    outs = {"ft": {e["id"]: llm.source_text(e) for e in test}}
    path = ev.blind(cfg, Path("/"), test, outs, "ft")
    fresh = path.read_text(encoding="utf-8")
    assert ev.blind(cfg, Path("/"), test, outs, "ft").read_text(encoding="utf-8") == fresh  # unanswered: rewritten
    answered = fresh.replace("real: ", "real: B", 1)
    path.write_text(answered, encoding="utf-8")
    (tmp_path / "blind_key.json").write_text("{}", encoding="utf-8")
    ev.blind(cfg, Path("/"), test, outs | {"ft": {}}, "ft")
    assert path.read_text(encoding="utf-8") == answered
    assert (tmp_path / "blind_key.json").read_text(encoding="utf-8") == "{}"


SHEET = """# Owner set

## c01 commit

neutral:
fix(auth): Sửa lỗi null trong `get_user`.

Co-Authored-By: Dev <dev@example.com>

mine:
fix(auth): sửa null trong `get_user`

## r01 comment

neutral:
Anh đã kiểm tra lại, lỗi không còn.

mine:
a check r, hết lỗi r nha

## r02 comment

neutral:
Em xem giúp anh.

mine:
"""


def test_owner_sheet_bodies_only_and_unfilled_items_dropped():
    items = owner_set.parse(SHEET)
    assert [(i["id"], i["task"]) for i in items] == [("c01", "commit"), ("r01", "comment")]
    c, r = owner_set.examples(items)
    assert llm.source_text(c) == "Sửa lỗi null trong `get_user`." and llm.target_text(c) == "sửa null trong `get_user`"
    assert c["variant"] == "commit" and c["prompt"][0]["content"] == SYSTEM
    assert llm.source_text(r) == "Anh đã kiểm tra lại, lỗi không còn." and llm.target_text(r) == "a check r, hết lỗi r nha"


def test_for_set_owner_builds_rows_and_swaps_paths(tmp_path):
    cfg = ev.load_cfg(Path("."), "configs/eval.yaml")
    assert ev.for_set(cfg, tmp_path, "chat") is cfg
    (tmp_path / cfg["owner_set"]["sheet"]).parent.mkdir(parents=True)
    (tmp_path / cfg["owner_set"]["sheet"]).write_text(SHEET, encoding="utf-8")
    owner = ev.for_set(cfg, tmp_path, "owner")
    assert owner["test"] == cfg["owner_set"]["test"] and owner["report"] == cfg["owner_set"]["report"]
    assert owner["train"] == cfg["train"] and owner["set"] == "owner"
    masked = [n for n, s in owner["systems"].items() if s.get("mask")]
    assert masked and not any(s.get("mask") for s in cfg["systems"].values())
    assert set(cfg["systems"]) < set(owner["systems"])
    rows = [json.loads(x) for x in (tmp_path / owner["test"]).read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["c01", "r01"]


def test_table_puts_per_task_style_after_style():
    head, _, row = ev.table({"x": {"n": 2, "style": 0.5, "style_comment": 1.0, "style_commit": 0.0}})
    assert head.startswith("| system | n | style | style_comment | style_commit | mean_len")
    assert row.startswith("| x | 2 | 50% | 100% | 0% | –")


def test_calibration_line_only_when_reference_is_judged():
    assert ev.calibration({}) == [] and ev.calibration({"meaning": None}) == []
    line = ev.calibration({"meaning": 4.6, "added": 0.2, "lost": 0.06})[0]
    assert "4.6" in line and "20%" in line and "6%" in line
