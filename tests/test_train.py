from pathlib import Path

import pytest
import yaml

from stylesync.dataset.build import SYSTEM
from stylesync.train.sft import append_log, best_eval, experiment, render

CFG = yaml.safe_load(Path("configs/train.yaml").read_text(encoding="utf-8"))


class FakeTok:
    """ChatML-like template; `drift` mimics a template that rewrites earlier turns once a reply follows."""

    def __init__(self, drift=False):
        self.drift, self.calls = drift, []

    def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
        self.calls.append(kw)
        system = msgs[0]["content"] + (" (drift)" if self.drift and msgs[-1]["role"] == "assistant" else "")
        # Like Qwen non-thinking: the assistant turn opens with an empty think block.
        turn = {"user": "<user>{}</user>\n", "assistant": "<assistant><think></think>{}</assistant>\n"}
        text = f"<s>{system}</s>" + "".join(turn[m["role"]].format(m["content"]) for m in msgs[1:])
        return text + ("<assistant><think></think>" if add_generation_prompt else "")


# Synthetic example only.
EX = {"id": "m00001-c", "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Mai bạn rảnh không?"}],
      "completion": [{"role": "assistant", "content": "mai m rảnh kh"}]}


def test_experiment_merges_overrides_and_keeps_shared_settings():
    e = experiment(CFG, "qwen35-2b-r16-e1")
    assert e["model_id"] == CFG["models"]["qwen35-2b"]
    assert e["train"]["epochs"] == 1 and e["train"]["lr"] == CFG["train"]["lr"]
    assert e["lora"] == CFG["lora"] and e["seed"] == CFG["seed"]
    assert CFG["train"]["epochs"] == 2  # overrides do not leak into the shared config
    cfg = CFG | {"experiments": {"x": {"model": "gemma4-e2b", "lora": {"r": 32}}}}
    assert experiment(cfg, "x")["lora"] == CFG["lora"] | {"r": 32}
    with pytest.raises(SystemExit):
        experiment(CFG, "nope")


def test_no_compile_applies_per_model():
    assert experiment(CFG, "qwen35-2b-r16-e1")["compile"] is True
    assert experiment(CFG, "gemma4-e2b-r16-e1")["compile"] is False
    assert experiment(CFG | {"no_compile": []}, "gemma4-e2b-r16-e1")["compile"] is True


def test_every_experiment_names_a_known_model():
    for name in CFG["experiments"]:
        assert experiment(CFG, name)["model_id"]


def test_render_splits_on_the_generation_prompt_without_thinking():
    tok = FakeTok()
    r = render(tok, EX)
    assert tok.calls == [{"tokenize": False, "enable_thinking": False}] * 2
    assert r["prompt"].endswith("<user>Mai bạn rảnh không?</user>\n<assistant><think></think>")
    assert r["prompt"] + r["completion"] == tok.apply_chat_template(EX["prompt"] + EX["completion"]).rstrip()
    assert r["completion"] == "mai m rảnh kh</assistant>"  # no trailing newline after the end-of-turn token


def test_render_rejects_a_template_that_is_not_prefix_stable():
    with pytest.raises(ValueError, match="m00001-c"):
        render(FakeTok(drift=True), EX)


def test_log_and_best_eval(tmp_path):
    log = tmp_path / "log.jsonl"
    assert best_eval(log) == {"evals": 0, "best_eval_loss": None, "best_step": None}
    for step, logs in [(10, {"loss": 2.1}), (100, {"eval_loss": 1.5}), (200, {"eval_loss": 1.2}), (300, {"eval_loss": 1.3})]:
        append_log(log, step, logs)
    assert best_eval(log) == {"evals": 3, "best_eval_loss": 1.2, "best_step": 200}
