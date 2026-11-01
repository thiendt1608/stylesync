"""Phase 5: rewrite the test set with a local model, the base alone or base + LoRA adapter. Needs a CUDA GPU.

Greedy decoding, so a rerun gives the same text. Results are appended per batch, so a killed run resumes.
"""
from pathlib import Path

from stylesync.clean.pipeline import read_jsonl
from stylesync.mcp_server import mask, unmask
from stylesync.neutralize.run import append_jsonl
from stylesync.train.sft import TEMPLATE_KW, load_base, model_spec


def messages(ex: dict, system: str | None = None) -> list[dict]:
    """Prompt turns of one test example; `system` replaces the training system prompt (Baseline A)."""
    return [{"role": "system", "content": system}] + ex["prompt"][1:] if system else ex["prompt"]


def batches(rows: list, size: int, length) -> list[list]:
    """Rows sorted by length, so each batch pads little."""
    order = sorted(rows, key=length)
    return [order[i:i + size] for i in range(0, len(order), size)]


def generate(cfg: dict, train_cfg: dict, name: str, base: Path, limit: int | None = None) -> Path:
    system_cfg = cfg["systems"][name]
    out = base / cfg["outputs_dir"] / f"{name}.jsonl"
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = [r for r in read_jsonl(base / cfg["test"])[:limit] if r["id"] not in done]
    if not rows:
        return out

    model, tok = load_base(model_spec(train_cfg, system_cfg["model"]))
    import torch
    from unsloth import FastModel

    if "adapter" in system_cfg:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(base / system_cfg["adapter"]))
    FastModel.for_inference(model)
    tok.padding_side = "left"
    system = None if "adapter" in system_cfg else cfg["base_system"]
    # mask: code spans go around the model as <CODEn> placeholders, as rewrite_style sends them.
    spans = {r["id"]: mask(r["prompt"][-1]["content"]) if system_cfg.get("mask") else (None, []) for r in rows}
    text = {}
    for r in rows:
        turns = messages(r, system)
        if spans[r["id"]][0] is not None:
            turns = turns[:-1] + [{"role": "user", "content": spans[r["id"]][0]}]
        text[r["id"]] = tok.apply_chat_template(turns, add_generation_prompt=True, **TEMPLATE_KW)
    n = len(done)
    for batch in batches(rows, cfg["gen"]["batch_size"], lambda r: len(text[r["id"]])):
        # The template text already starts with BOS where the model wants one (Gemma): do not add a second.
        enc = tok([text[r["id"]] for r in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cfg["gen"]["max_new_tokens"], do_sample=False)
        new = gen[:, enc["input_ids"].shape[1]:]
        append_jsonl(out, [{"id": r["id"], "output": unmask(tok.decode(t, skip_special_tokens=True).strip(),
                                                              spans[r["id"]][1])}
                           for r, t in zip(batch, new)])
        n += len(batch)
        print(f"{name}: {n}/{len(done) + len(rows)}", flush=True)
    return out
