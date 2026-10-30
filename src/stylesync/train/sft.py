"""Phase 4: LoRA SFT of a base model on the Phase 3 splits (plan section 8). Needs a CUDA GPU (3060 or Colab).

  stylesync-train --exp qwen35-2b-r16-e1            train one experiment from configs/train.yaml
  stylesync-train --exp qwen35-2b-r16-e1 --resume   continue from the last checkpoint in runs/<exp>/
  stylesync-train --exp qwen35-2b-r16-e1 --max-steps 20 --eval-steps 10 --limit 400   smoke test

Each example is rendered to plain text with the base model's own chat template (non-thinking, the same
settings the MCP server sends to llama-server), so the trainer sees a standard prompt-completion pair and
computes loss on the completion only. Output in runs/<exp>/: adapter/, log.jsonl, metrics.json.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from stylesync.clean.pipeline import read_jsonl

TEMPLATE_KW = {"tokenize": False, "enable_thinking": False}


def experiment(cfg: dict, name: str) -> dict:
    """Merge one experiment's overrides into the shared settings."""
    if name not in cfg["experiments"]:
        raise SystemExit(f"unknown experiment {name!r}; known: {', '.join(cfg['experiments'])}")
    over = cfg["experiments"][name]
    return {"name": name, "model": over["model"], "model_id": cfg["models"][over["model"]], "seed": cfg["seed"],
            "max_seq_len": cfg["max_seq_len"], "compile": over["model"] not in cfg.get("no_compile", []),
            "lora": cfg["lora"] | over.get("lora", {}),
            "train": cfg["train"] | over.get("train", {})}


def render(tok, ex: dict) -> dict:
    """Prompt text ends with the assistant header; completion is the reply plus the template's end-of-turn."""
    prompt = tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, **TEMPLATE_KW)
    full = tok.apply_chat_template(ex["prompt"] + ex["completion"], **TEMPLATE_KW)
    if not full.startswith(prompt):
        # The template rewrites earlier turns once a reply follows, so prompt and full text would disagree.
        raise ValueError(f"chat template is not prefix-stable for {ex['id']}")
    # Drop the newline after the end-of-turn token: TRL appends eos when the completion does not end with it,
    # so for Qwen ("<|im_end|>" is eos) a trailing "\n" would teach a doubled end token.
    return {"prompt": prompt, "completion": full[len(prompt):].rstrip()}


def append_log(path: Path, step: int, logs: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"step": step} | logs, ensure_ascii=False) + "\n")


def best_eval(log: Path) -> dict:
    """Lowest val loss seen in log.jsonl, and how many evals ran."""
    evals = [r for r in read_jsonl(log) if "eval_loss" in r] if log.exists() else []
    best = min(evals, key=lambda r: r["eval_loss"], default=None)
    return {"evals": len(evals), "best_eval_loss": best and best["eval_loss"], "best_step": best and best["step"]}


def git_commit(base: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(base), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def train(exp: dict, cfg: dict, base: Path, resume: bool, max_steps: int, limit: int | None) -> dict:
    # Heavy imports stay here so the rest of the package (and its tests) runs without a GPU stack.
    if not exp["compile"]:
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # read when unsloth is imported
    import unsloth  # noqa: F401  (must be imported before transformers/trl to apply its patches)
    from unsloth import FastModel
    import importlib.metadata as md
    from datasets import Dataset
    from transformers import EarlyStoppingCallback, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    out = base / cfg["runs_dir"] / exp["name"]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.jsonl"
    if not resume:
        log.unlink(missing_ok=True)
    t, lora = exp["train"], exp["lora"]

    # use_exact_model_name: train the repo named in the config, not Unsloth's mirror of it (e.g. unsloth/gemma-4-E2B-it).
    model, proc = FastModel.from_pretrained(model_name=exp["model_id"], max_seq_length=exp["max_seq_len"],
                                            load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
                                            use_exact_model_name=True)
    tok = getattr(proc, "tokenizer", proc)  # VLM checkpoints return a processor; text-only training needs its tokenizer
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True,
                                     r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
                                     use_gradient_checkpointing="unsloth", random_state=exp["seed"])

    def load(split: str) -> Dataset:
        rows = read_jsonl(base / cfg["data_dir"] / f"{split}.jsonl")[:limit]
        return Dataset.from_list([render(tok, r) for r in rows])

    train_ds, val_ds = load("train"), load("val")

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            append_log(log, state.global_step, logs or {})

    args = SFTConfig(
        output_dir=str(out), seed=exp["seed"], data_seed=exp["seed"], max_length=exp["max_seq_len"],
        num_train_epochs=t["epochs"], max_steps=max_steps, learning_rate=t["lr"], lr_scheduler_type=t["scheduler"],
        warmup_steps=t["warmup_ratio"], per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"], gradient_accumulation_steps=t["grad_accum"],
        optim="adamw_8bit", bf16=True, logging_steps=t["logging_steps"],
        eval_strategy="steps", eval_steps=t["eval_steps"], save_strategy="steps", save_steps=t["eval_steps"],
        save_total_limit=t["save_total_limit"], load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, completion_only_loss=True, report_to="none")
    trainer = SFTTrainer(model=model, processing_class=tok, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         callbacks=[Log(), EarlyStoppingCallback(early_stopping_patience=t["early_stop_patience"])])
    start = time.time()
    result = trainer.train(resume_from_checkpoint=resume and any(out.glob("checkpoint-*")) or None)
    trainer.save_model(str(out / "adapter"))  # best checkpoint, restored by load_best_model_at_end
    tok.save_pretrained(str(out / "adapter"))

    metrics = {"exp": exp, "commit": git_commit(base), "n_train": len(train_ds), "n_val": len(val_ds),
               "steps": result.global_step, "planned_steps": trainer.state.max_steps,
               "stopped_early": result.global_step < trainer.state.max_steps, "train_loss": result.training_loss,
               "minutes": round((time.time() - start) / 60, 1), "smoke": bool(max_steps > 0 or limit),
               "versions": {p: md.version(p) for p in ("unsloth", "torch", "transformers", "trl", "peft")}}
    metrics |= best_eval(log)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="experiment name from the config")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--base", default=".", help="repo root that config paths are relative to")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N examples of each split")
    ap.add_argument("--eval-steps", type=int, default=None, help="override eval/checkpoint interval (smoke test)")
    args = ap.parse_args()
    base = Path(args.base)
    cfg = yaml.safe_load((base / args.config).read_text(encoding="utf-8"))
    exp = experiment(cfg, args.exp)
    if args.eval_steps:
        exp["train"]["eval_steps"] = args.eval_steps
    metrics = train(exp, cfg, base, args.resume, args.max_steps, args.limit)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
