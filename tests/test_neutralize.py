import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import yaml

from stylesync.clean.pipeline import read_jsonl, write_jsonl
from stylesync.neutralize import run
from stylesync.neutralize.prompt import SCHEMA, SYSTEM, VARIANTS, cache_key, request
from stylesync.neutralize.validate import check

CFG = yaml.safe_load(Path("configs/neutralize.yaml").read_text(encoding="utf-8"))
RATIO = tuple(CFG["validate"]["len_ratio"])

# Synthetic data only.
ROWS = [
    {"id": "m00000", "source": "messenger", "thread_id": "t1", "ts": "2025-01-01T00:00:00", "n_msgs": 1, "weight": 1.0,
     "text": "mai m rảnh kh, đi cf vs t"},
    {"id": "m00001", "source": "messenger", "thread_id": "t1", "ts": "2025-01-02T00:00:00", "n_msgs": 1, "weight": 1.0,
     "text": "fix `user_id` trong config.yaml đc r =))"},
    {"id": "d00000", "source": "discord", "thread_id": "t2", "ts": "2024-05-01T00:00:00", "n_msgs": 2, "weight": 0.5,
     "text": "<NAME> ơi deploy lúc 9h nha"},
]
OUT = {
    ROWS[0]["text"]: {"formal": "Ngày mai bạn có rảnh không? Bạn đi cà phê với tôi nhé.",
                      "concise": "Mai bạn rảnh không? Đi cà phê với tôi."},
    ROWS[1]["text"]: {"formal": "Tôi đã sửa `user_id` trong config.yaml.",
                      "concise": "Đã sửa `user_id` trong config.yaml."},
    ROWS[2]["text"]: {"formal": "<NAME>, chúng ta triển khai lúc 9 giờ nhé.",
                      "concise": "<NAME> ơi, deploy lúc 9h nhé k"},  # leftover abbreviation -> rejected
}


# --- validate ---------------------------------------------------------------

def test_check_keeps_good_rewrite():
    assert check("fix lỗi login đc r =))", "Đã sửa lỗi đăng nhập.", RATIO) == []


def test_check_reasons():
    assert check("ok đc r", "", RATIO) == ["empty"]
    assert "unchanged" in check("Đã xong rồi.", "đã xong rồi", RATIO)
    assert "len_ratio" in check("mai đi cf kh m ơi, t bao", "Đi.", RATIO)
    assert "placeholder" in check("<NAME> ơi đi chơi kh", "Bạn ơi, đi chơi không?", RATIO)
    assert "placeholder" in check("<NAME> ơi đi chơi kh", "<NAME>, <NAME> đi chơi không?", RATIO)
    assert "number" in check("mai 9h họp nha m", "Ngày mai họp nhé.", RATIO)
    assert "number" in check("t mua 25 cái r", "Tôi đã mua hai cái.", RATIO)  # only 0-10 may be spelled out
    assert "code" in check("sửa `foo_bar` r", "Đã sửa biến rồi.", RATIO)
    assert "code" in check("lỗi ở src/app/main.py á", "Lỗi nằm ở file main.", RATIO)
    assert "abbrev" in check("mai đi kh m", "Mai bạn đi kh?", RATIO)
    assert "emoticon" in check("xong r =))", "Đã xong =))", RATIO)
    assert "no_diacritics" in check("mai đi chơi kh m", "Are you going out tomorrow?", RATIO)


def test_check_ignores_allowed_tokens():
    # "ok", "v.v.", "v1", code in backticks are not leftover abbreviations.
    assert check("ok để t làm v1 r tính tiếp mấy cái kia", "Được, tôi làm bản v1 rồi tính tiếp các phần khác, v.v.",
                 RATIO) == []
    assert check("chạy `ls -k` đi m", "Bạn hãy chạy `ls -k`.", RATIO) == []
    # Small numbers spelled out keep the meaning.
    assert check("mai 9h họp nha m, mang 2 cái laptop", "Ngày mai họp lúc chín giờ, bạn mang hai cái laptop.",
                 RATIO) == []


# --- prompt -----------------------------------------------------------------

def test_cache_key_is_valid_custom_id_and_versioned():
    k = cache_key("m", "v1", "abc")
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", k)
    assert k == cache_key("m", "v1", "abc")
    assert k != cache_key("m", "v2", "abc") != cache_key("m2", "v1", "abc")


def test_request_shape():
    p = request(CFG, "xin chào")
    assert p["model"] == CFG["model"] and p["store"] is False
    assert p["input"] == "<message>\nxin chào\n</message>"
    fmt = p["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["schema"] is SCHEMA
    assert set(SCHEMA["required"]) == set(VARIANTS) and SCHEMA["additionalProperties"] is False


def test_prompt_examples_pass_validator():
    # The few-shot examples must show what the validator keeps, or the prompt teaches pairs that get rejected.
    examples = re.findall(r"<message>\n(.*?)\n</message>\nformal: (.*?)\nconcise: (.*?)(?=\n\n|$)", SYSTEM, re.S)
    assert len(examples) >= 3
    for original, formal, concise in examples:
        assert check(original, formal, RATIO) == [] and check(original, concise, RATIO) == []
    assert "push code" in SYSTEM and "đẩy mã nguồn" not in SYSTEM.split("Ví dụ:")[1]  # English terms stay English


# --- batch flow with a fake OpenAI client --------------------------------------

def body(text, status="completed", reason=None, refusal=False):
    content = [{"type": "refusal", "refusal": "no"}] if refusal else [{"type": "output_text", "text": text}]
    return {"status": status, "incomplete_details": {"reason": reason} if reason else None,
            "output": [{"type": "reasoning", "summary": []}, {"type": "message", "content": content}],
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400},
                      "output_tokens": 200, "output_tokens_details": {"reasoning_tokens": 50}}}


def ok_line(cid, b):
    return {"id": "r", "custom_id": cid, "response": {"status_code": 200, "request_id": "q", "body": b}, "error": None}


class FakeOpenAI:
    def __init__(self):
        self.files_store, self.batches_store, self.deleted, self.polls = {}, {}, [], 0
        self.files = NS(create=self.file_create, content=self.file_content, delete=self.deleted.append)
        self.batches = NS(create=self.batch_create, retrieve=self.batch_retrieve)

    def file_create(self, file, purpose):
        assert purpose == "batch"
        fid = f"file-{len(self.files_store)}"
        self.files_store[fid] = file.read().decode("utf-8")
        return NS(id=fid)

    def file_content(self, fid):
        return NS(text=self.files_store[fid])

    def batch_create(self, input_file_id, endpoint, completion_window):
        assert endpoint == "/v1/responses" and completion_window == "24h"
        bid = f"batch_{len(self.batches_store) + 1}"
        reqs = [json.loads(x) for x in self.files_store[input_file_id].splitlines()]
        assert all(r["url"] == "/v1/responses" and r["method"] == "POST" for r in reqs)
        out, err = [], []
        for r in reqs:
            text = r["body"]["input"].split("\n")[1]
            if text == ROWS[2]["text"] and bid == "batch_1":
                err.append({"id": "r", "custom_id": r["custom_id"], "response": None,
                            "error": {"code": "batch_expired", "message": "expired"}})  # retried by the next submit
            else:
                out.append(ok_line(r["custom_id"], body(json.dumps(OUT[text]))))
        self.files_store[f"out-{bid}"] = "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n"
        self.files_store[f"err-{bid}"] = "\n".join(json.dumps(x) for x in err)
        self.batches_store[bid] = (input_file_id, len(reqs))
        return NS(id=bid)

    def batch_retrieve(self, bid):
        self.polls += 1
        fid, n = self.batches_store[bid]
        status = "completed" if self.polls % 2 == 0 else "in_progress"
        return NS(status=status, request_counts=NS(completed=0, total=n), input_file_id=fid,
                  output_file_id=f"out-{bid}", error_file_id=f"err-{bid}" if self.files_store[f"err-{bid}"] else None)


def setup(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    write_jsonl(tmp_path / cfg["in"], ROWS)
    return cfg, FakeOpenAI()


def test_submit_collect_build_resume(tmp_path):
    cfg, client = setup(tmp_path)
    assert run.submit(client, cfg, tmp_path, ROWS) == "batch_1"
    assert not (tmp_path / cfg["out"]["batches"]).with_name("upload.jsonl").exists()
    assert run.submit(client, cfg, tmp_path, ROWS) is None  # all pending in batch_1
    assert run.collect(client, cfg, tmp_path, poll_s=0) == {"succeeded": 2, "batch_expired": 1}
    assert len(read_jsonl(tmp_path / cfg["out"]["cache"])) == 2
    assert client.deleted == ["file-0", "out-batch_1", "err-batch_1"]  # nothing left on OpenAI

    assert run.submit(client, cfg, tmp_path, ROWS) == "batch_2"  # only the expired one
    assert client.batches_store["batch_2"][1] == 1
    run.collect(client, cfg, tmp_path, poll_s=0)
    assert run.collect(client, cfg, tmp_path, poll_s=0) == {}  # done batches are not polled again
    assert run.submit(client, cfg, tmp_path, ROWS) is None  # everything cached

    stats = run.build(cfg, tmp_path, ROWS, n_all=30)
    pairs = read_jsonl(tmp_path / cfg["out"]["pairs"])
    assert stats["cached"] == 3 and stats["missing"] == 0
    assert stats["pairs"] == {"formal": 3, "concise": 2} and stats["reject_reasons"] == {"abbrev": 1}
    p = next(p for p in pairs if p["id"] == "d00000-f")
    assert p["input"] == OUT[ROWS[2]["text"]]["formal"] and p["output"] == ROWS[2]["text"]
    assert p["weight"] == 0.5 and p["thread_id"] == "t2" and p["task"] == "message"
    # 3 rows x (600 uncached x 0.05 + 400 cached x 0.005 + 200 out x 0.25) / 1e6
    assert stats["cost_usd"] == round(3 * (600 * 0.05 + 400 * 0.005 + 200 * 0.25) / 1e6, 4)
    assert stats["cost_usd_projected_all_rows"] == round(stats["cost_usd"] / 3 * 30, 2)
    assert stats["tokens"]["reasoning_tokens"] == 150


def test_parse_rejects_bad_output():
    good = json.dumps(OUT[ROWS[0]["text"]])
    assert run.parse(ok_line("k", body(good)))["out"] == OUT[ROWS[0]["text"]]
    assert run.parse(ok_line("k", body("not json")))["out"] is None
    assert run.parse(ok_line("k", body(json.dumps({"formal": "a"}))))["out"] is None
    cut = run.parse(ok_line("k", body(good[:20], status="incomplete", reason="max_output_tokens")))
    assert cut["out"] is None and cut["stop_reason"] == "max_output_tokens"
    refused = run.parse(ok_line("k", body("", refusal=True)))
    assert refused["out"] is None and refused["stop_reason"] == "refusal"
    assert refused["usage"] == {"input_tokens": 1000, "cached_tokens": 400, "output_tokens": 200,
                                "reasoning_tokens": 50}
    assert run.parse({"custom_id": "k", "response": {"status_code": 500, "body": {}}, "error": None}) is None
    assert run.parse({"custom_id": "k", "response": None, "error": {"code": "batch_expired"}}) is None


def test_build_marks_refusal_and_review(tmp_path):
    cfg, _ = setup(tmp_path)
    key = run.key_of(cfg, ROWS[0])
    run.append_jsonl(tmp_path / cfg["out"]["cache"], [
        {"key": key, "stop_reason": "refusal", "out": None, "usage": dict.fromkeys(run.USAGE, 0)},
        {"key": run.key_of(cfg, ROWS[1]), "stop_reason": "end_turn", "out": OUT[ROWS[1]["text"]],
         "usage": dict.fromkeys(run.USAGE, 0)},
    ])
    stats = run.build(cfg, tmp_path, ROWS, n_all=3)
    assert stats["reject_reasons"] == {"refusal": 1} and stats["missing"] == 1
    sheet = run.review(cfg, tmp_path, n=100).read_text(encoding="utf-8")
    assert sheet.count("## m00001-") == 2
