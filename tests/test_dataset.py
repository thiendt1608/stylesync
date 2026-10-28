from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from stylesync.clean.pipeline import read_jsonl, write_jsonl
from stylesync.dataset.build import SYSTEM, assign, build

CFG = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def burst(i, src="messenger", days=0.0, weight=1.0):
    return {"id": f"{src[0]}{i:05d}", "source": src, "ts": (T0 + timedelta(days=days)).isoformat(), "weight": weight}


def pairs_of(b):
    # Synthetic data only: two neutral variants per burst, distinct originals.
    return [{"id": f"{b['id']}-{v[0]}", "task": "message", "source": b["source"], "thread_id": "t1", "ts": b["ts"],
             "weight": b["weight"], "variant": v, "input": f"Bản trung tính {b['id']} {v}.",
             "output": f"tin gốc {b['id']} nha"} for v in ("formal", "concise")]


def test_assign_newest_is_test_then_val():
    bursts = [burst(i, days=i * 3) for i in range(20)]  # 3 days apart: embargo of 24h drops nothing
    split = assign(bursts, 0.1, 0.1, timedelta(hours=24))
    order = [split[b["id"]] for b in bursts]
    assert order == ["train"] * 16 + ["val"] * 2 + ["test"] * 2


def test_assign_embargo_drops_bursts_just_before_a_boundary():
    day = timedelta(hours=24)
    # val starts at 9.5: the train burst at 9 is 12h before it.
    bursts = [burst(i, days=d) for i, d in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 9, 9.5, 12])]
    assert [assign(bursts, 0.1, 0.1, day)[b["id"]] for b in bursts] == ["train"] * 8 + [None, "val", "test"]
    # test starts at 8.5: the only val burst (8) is 12h before it, and so is the train burst at 7.5.
    bursts = [burst(i, days=d) for i, d in enumerate([0, 1, 2, 3, 4, 5, 6, 7.5, 8, 8.5])]
    assert [assign(bursts, 0.1, 0.1, day)[b["id"]] for b in bursts] == ["train"] * 7 + [None, None, "test"]
    assert assign([], 0.1, 0.1, timedelta(hours=24)) == {}


def test_assign_keeps_same_timestamp_together():
    bursts = [burst(i, days=i) for i in range(8)] + [burst(100 + i, days=8) for i in range(3)]
    split = assign(bursts, 0.1, 0.1, timedelta(hours=0))
    assert {split[b["id"]] for b in bursts if b["ts"] == bursts[-1]["ts"]} == {"test"}


def test_build_splits_per_source_weights_and_format(tmp_path):
    bursts = ([burst(i, days=i * 2) for i in range(30)]
              + [burst(i, src="discord", days=i * 2, weight=0.5 if i < 4 else 1.0) for i in range(10)])
    write_jsonl(tmp_path / CFG["in"], [p for b in bursts for p in pairs_of(b)])
    stats = build(CFG, tmp_path)
    rows = {s: read_jsonl(tmp_path / CFG["out"]["dir"] / f"{s}.jsonl") for s in ("train", "val", "test")}

    # Each source gets its own newest 10%: discord is in test even though its timestamps are older.
    assert stats["splits"]["test"]["by_source"] == {"messenger": 6, "discord": 2}
    assert stats["splits"]["val"]["by_source"] == {"messenger": 6, "discord": 2}
    # Half-weight bursts keep 1 of 2 variants in train only.
    assert stats["splits"]["train"]["by_source"] == {"messenger": 48, "discord": 12}
    assert stats["dropped"] == {"weight": 4}
    # Both variants of a burst stay in the same split, and no original repeats across splits.
    seen = {}
    for s, ex in rows.items():
        for e in ex:
            assert seen.setdefault(e["id"].rsplit("-", 1)[0], s) == s
    assert stats["leak_same_output"] == {"train-val": 0, "train-test": 0}

    e = rows["test"][0]
    assert [m["role"] for m in e["prompt"]] == ["system", "user"] and e["prompt"][0]["content"] == SYSTEM
    assert e["completion"] == [{"role": "assistant", "content": f"tin gốc {e['id'].rsplit('-', 1)[0]} nha"}]
    assert e["prompt"][1]["content"].startswith("Bản trung tính")


def test_build_is_deterministic(tmp_path):
    bursts = [burst(i, src="discord", days=i, weight=0.5) for i in range(20)]
    write_jsonl(tmp_path / CFG["in"], [p for b in bursts for p in pairs_of(b)])
    build(CFG, tmp_path)
    first = (tmp_path / CFG["out"]["dir"] / "train.jsonl").read_text(encoding="utf-8")
    build(CFG, tmp_path)
    assert (tmp_path / CFG["out"]["dir"] / "train.jsonl").read_text(encoding="utf-8") == first
