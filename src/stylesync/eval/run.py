"""Phase 5: eval on the test set (plan section 9).

  gen --system NAME [--limit N]   rewrite the test set with one local system (CUDA GPU box)
  fewshot [--no-wait]             Baseline B through OpenAI Batch, then data/eval/outputs/<fewshot>.jsonl
  judge [--no-wait]               meaning judge on every system's outputs (and the reference) through OpenAI Batch
  report                          reports/eval.md (aggregates), data/eval/examples.md and blind sheet (private)

--set owner runs any command on the owner-written commit/comment set instead of the chat test set
(`owner_set` in the config): same systems and metrics, report in reports/eval_owner.md.

Systems: "neutral" (the input unchanged, the floor), "reference" (the real message, the ceiling), Baseline A
(base-*), the fine-tuned adapters (ft-*) and Baseline B (fewshot). Without --no-wait, fewshot and judge
block until their batches end (often minutes, up to 24 h); rerun with the same command to resume.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import yaml

from stylesync.clean.pipeline import read_jsonl, write_jsonl
from stylesync.eval import llm, metrics, owner_set
from stylesync.neutralize.run import USAGE, collect, cost, read_cache, submit_bodies


def load_cfg(base: Path, path: str) -> dict:
    return yaml.safe_load((base / path).read_text(encoding="utf-8"))


def for_set(cfg: dict, base: Path, name: str) -> dict:
    """The config for one eval set. For "owner", its test rows are rebuilt from the private sheet when it is here
    (the GPU box only gets the rows), its paths replace the chat set's and its extra systems are added."""
    if name == "chat":
        return cfg
    o = cfg["owner_set"]
    sheet = base / o["sheet"]
    if sheet.exists():
        write_jsonl(base / o["test"], owner_set.examples(owner_set.parse(sheet.read_text(encoding="utf-8"))))
    return (cfg | {k: o[k] for k in ("test", "outputs_dir", "report")}
            | {"systems": cfg["systems"] | o.get("systems", {}), "set": "owner"})


def system_names(cfg: dict) -> list[str]:
    return list(cfg["systems"]) + [cfg["fewshot"]["name"]]


def outputs(cfg: dict, base: Path, test: list[dict]) -> dict[str, dict[str, str]]:
    """{system: {test id: output}}, for the fixed systems and every output file present."""
    out = {"neutral": {r["id"]: llm.source_text(r) for r in test},
           "reference": {r["id"]: llm.target_text(r) for r in test}}
    for name in system_names(cfg):
        path = base / cfg["outputs_dir"] / f"{name}.jsonl"
        if path.exists():
            out[name] = {r["id"]: r["output"] for r in read_jsonl(path)}
    return out


def openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set in this shell")
    from openai import OpenAI

    return OpenAI()


def run_batch(cfg: dict, base: Path, bodies: dict[str, dict], wait: bool) -> None:
    client = openai_client()
    bid = submit_bodies(client, cfg["llm"], base, bodies)
    print(f"submitted {bid} ({len(bodies)} requests in scope)" if bid else "nothing to submit: all cached or pending")
    print(json.dumps(collect(client, cfg["llm"], base, decode=llm.decode_any, wait=wait)))


def fewshot_bodies(cfg: dict, base: Path, test: list[dict]) -> dict[str, tuple[str, dict]]:
    """{test id: (cache key, request body)}."""
    shots = llm.retrieve(read_jsonl(base / cfg["train"]), test, cfg["fewshot"]["k"])
    out = {}
    for r in test:
        text = llm.fewshot_input(shots[r["id"]], llm.source_text(r))
        out[r["id"]] = (llm.fewshot_key(cfg["llm"], text),
                        llm.body(cfg["llm"], llm.FEWSHOT_SYSTEM, text, "rewrite", llm.REWRITE_SCHEMA))
    return out


def fewshot(cfg: dict, base: Path, test: list[dict], wait: bool) -> Path:
    req = fewshot_bodies(cfg, base, test)
    run_batch(cfg, base, dict(req.values()), wait)
    cache = read_cache(cfg["llm"], base)
    # Refused or malformed replies become "" so they count as failures, not as missing rows.
    rows = [{"id": i, "output": (cache[k]["out"] or {}).get("rewrite", "")} for i, (k, _) in req.items() if k in cache]
    path = base / cfg["outputs_dir"] / f"{cfg['fewshot']['name']}.jsonl"
    write_jsonl(path, rows)
    print(f"{path}: {len(rows)}/{len(test)} rewritten")
    return path


def judge_requests(cfg: dict, test: list[dict], outs: dict) -> dict[str, dict]:
    """One judge request per distinct (source, rewrite); the neutral floor needs none (it is the source)."""
    bodies = {}
    for name, by_id in outs.items():
        if name == "neutral":
            continue
        for r in test:
            if r["id"] in by_id:
                src = llm.source_text(r)
                key = llm.judge_key(cfg["llm"], src, by_id[r["id"]])
                bodies[key] = llm.body(cfg["llm"], llm.JUDGE_SYSTEM, llm.judge_input(src, by_id[r["id"]]),
                                       "judge", llm.JUDGE_SCHEMA)
    return bodies


def verdicts(cfg: dict, cache: dict, test: list[dict], by_id: dict[str, str]) -> list[dict]:
    """Cached judge verdicts for one system, with the test id; malformed or refused verdicts are left out."""
    out = []
    for r in test:
        if r["id"] in by_id:
            e = cache.get(llm.judge_key(cfg["llm"], llm.source_text(r), by_id[r["id"]]))
            if e and e["out"] and "score" in e["out"]:
                out.append(e["out"] | {"id": r["id"]})
    return out


def summarize(cfg: dict, base: Path, test: list[dict], outs: dict) -> dict[str, dict]:
    """One row of aggregate numbers per system."""
    train = read_jsonl(base / cfg["train"])
    clf = metrics.fit_style([llm.target_text(r) for r in train], [llm.source_text(r) for r in train])
    cache = read_cache(cfg["llm"], base)
    variants = sorted({r["variant"] for r in test})
    rows = {}
    for name, by_id in outs.items():
        done = [r for r in test if r["id"] in by_id]
        texts = [by_id[r["id"]] for r in done]
        v = verdicts(cfg, cache, test, by_id) if name != "neutral" else []
        rows[name] = ({"n": len(done), "style": metrics.style_rate(clf, texts)}
                      | {f"style_{x}": metrics.style_rate(clf, [by_id[r["id"]] for r in done if r["variant"] == x])
                         for x in variants}
                      | metrics.stylometry(texts)
                      | metrics.check_rates([(llm.source_text(r), by_id[r["id"]]) for r in done])
                      | metrics.judge_summary(v))
    return rows


def gates(cfg: dict, rows: dict) -> dict[str, dict]:
    few, ref = rows.get(cfg["fewshot"]["name"], {}), rows.get("reference", {})
    out = {}
    for name, sys_cfg in cfg["systems"].items():
        if "adapter" in sys_cfg and name in rows:
            ft, base_row = rows[name], rows.get(f"base-{sys_cfg['model']}", {})
            out[name] = metrics.gate(ft, base_row, few, ref, cfg["gate"])
    return out


PERCENT = {"style", "abbrev", "diacritics", "emoticon", "codemix", "hard_tokens",
           "placeholder", "guard", "added", "lost", "low"}
COLUMNS = ("n", "style", "mean_len", "abbrev", "diacritics", "emoticon", "codemix",
           "hard_tokens", "hard_token_n", "placeholder", "guard", "meaning", "added", "lost", "low")


def columns(rows: dict) -> tuple[str, ...]:
    """COLUMNS with the per-variant style rates (concise/formal, or commit/comment) right after `style`."""
    split = sorted({c for r in rows.values() for c in r if c.startswith("style_")})
    return COLUMNS[:2] + tuple(split) + COLUMNS[2:]


def table(rows: dict, cols: tuple[str, ...] | None = None) -> list[str]:
    def cell(x):
        return "–" if x is None else str(x)

    cols = cols or columns(rows)
    lines = ["| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(cell(r.get(c)) if not (c in PERCENT or c.startswith("style_")) or r.get(c) is None
                                                       else f"{r[c]:.0%}" for c in cols) + " |")
    return lines


def llm_cost(cfg: dict, base: Path) -> dict:
    entries = list(read_cache(cfg["llm"], base).values())
    return {"requests": len(entries), "tokens": {u: sum(e["usage"][u] for e in entries) for u in USAGE},
            "cost_usd": round(cost(entries, cfg["llm"]["price"]), 4)}


def best_ft(cfg: dict, rows: dict) -> str | None:
    """Fine-tuned system with the most style among those judged, else among all generated."""
    ft = [n for n, s in cfg["systems"].items() if "adapter" in s and n in rows]
    return max(ft, key=lambda n: (rows[n]["style"], rows[n]["meaning"] or 0), default=None)


def examples_md(cfg: dict, base: Path, test: list[dict], outs: dict, name: str, n: int = 10) -> Path:
    """Private sheet: the n best and n worst rewrites of one system by judge score (ties: shorter first)."""
    v = {x["id"]: x for x in verdicts(cfg, read_cache(cfg["llm"], base), test, outs[name])}
    by = {r["id"]: r for r in test if r["id"] in v}
    ranked = sorted(by, key=lambda i: (v[i]["score"], -len(outs[name][i])))
    lines = [f"# {name}: best and worst by judge score (private, never commit)\n"]
    for title, ids in (("Best", ranked[::-1][:n]), ("Worst", ranked[:n])):
        lines.append(f"## {title}\n")
        for i in ids:
            j = v[i]
            lines += [f"### {i} score {j['score']} added {j['added']} lost {j['lost']}", "",
                      "input:", "", llm.source_text(by[i]), "", "output:", "", outs[name][i], "",
                      "real:", "", llm.target_text(by[i]), ""]
    path = (base / cfg["outputs_dir"]).parent / "examples.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def blind(cfg: dict, base: Path, test: list[dict], outs: dict, name: str) -> Path:
    """Private sheet of n (real, model) pairs in random order; the key goes to a separate file.
    A sheet the owner has started answering is kept, with its key, so a re-run never wipes the answers."""
    out_dir = (base / cfg["outputs_dir"]).parent
    path = out_dir / "blind.md"
    if path.exists() and re.search(r"^real:[ \t]*[ABab]", path.read_text(encoding="utf-8"), re.M):
        return path
    rng = random.Random(cfg["blind"]["seed"])
    pool = [r for r in test if outs[name].get(r["id"])]
    picked = rng.sample(pool, min(cfg["blind"]["n"], len(pool)))
    lines = ["# Blind test: which reply did the owner really write, A or B? Write the letter after each pair.\n"]
    key = {}
    for k, r in enumerate(picked, 1):
        real_first = rng.random() < 0.5
        a, b = (llm.target_text(r), outs[name][r["id"]]) if real_first else (outs[name][r["id"]], llm.target_text(r))
        key[k] = {"id": r["id"], "real": "A" if real_first else "B"}
        lines += [f"## {k}", "", "A:", "", a, "", "B:", "", b, "", "real: ", ""]
    (out_dir / "blind_key.json").write_text(json.dumps({"system": name, "key": key}, indent=1), encoding="utf-8")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def calibration(ref: dict, what: str = "real message") -> list[str]:
    """What the judge says about the owner's real message against its own neutral input: the best a rewrite
    can look under this judge, since the neutral text may itself drop details the real message has."""
    if ref.get("meaning") is None:
        return []
    return [f"Judge calibration: the `reference` row is the owner's {what} judged against its neutral input."
            f" It scores {ref['meaning']}, with {ref['added']:.0%} flagged as adding and {ref['lost']:.0%} as losing"
            " information, so flags at that level come from the neutral inputs and the judge, not the model.", ""]


def report(cfg: dict, base: Path, test: list[dict]) -> Path:
    outs = outputs(cfg, base, test)
    rows = summarize(cfg, base, test, outs)
    g = gates(cfg, rows)
    spent = llm_cost(cfg, base)
    missing = [n for n in system_names(cfg) if n not in outs]
    owner = cfg.get("set") == "owner"
    intro = ([
        "# Phase 5 eval: owner commit/comment set\n",
        f"{len(test)} items the owner wrote by hand (plan section 9): a neutral draft and the owner's own version, split"
        f" by task. As in `rewrite_style`, only a commit's body is rewritten and compared; the conventional prefix"
        f" and trailers are kept by code. With n = {len(test)}, one item moves a rate by {1 / max(len(test), 1):.1%}."
        " Generated by `stylesync-eval report --set owner`; aggregates only, no message text. Good/bad examples"
        f" stay in `{Path(cfg['outputs_dir']).parent}/` (private).\n",
    ] if owner else [
        "# Phase 5 eval\n",
        f"Test set: {len(test)} examples (plan section 9). Generated by `stylesync-eval report`; aggregates only,",
        "no message text. Good/bad examples and the blind sheet stay in `data/eval/` (private).\n",
    ])
    lines = [
        *intro,
        "## Systems\n",
        "- `neutral`: the input unchanged (floor). `reference`: the owner's " + ("own version" if owner else "real message")
        + " (ceiling).",
        "- `base-*`: Baseline A, base model zero-shot with the descriptive MCP prompt, greedy.",
        "- `ft-*`: base model + LoRA adapter from Phase 4, training prompt, greedy.",
        f"- `{cfg['fewshot']['name']}`: Baseline B, {cfg['llm']['model']} with the {cfg['fewshot']['k']} most similar "
        "train pairs (char n-gram TF-IDF, one per burst) as examples.",
        f"- Not run yet: {', '.join(missing)}." if missing else "- Every configured system has outputs.",
        "",
        "## Results\n",
        "- `style`: share classified as the owner's (char n-gram TF-IDF + logistic regression, train outputs vs train"
        " inputs). `style_<variant>`: same, split by the neutral input's variant"
        + (" (here the task; the classifier only saw chat, so read it as a rough signal)." if owner else "."),
        "- `mean_len` … `codemix`: stylometry as in `reports/data.md` (word shares; emoticon is a share of messages).",
        "- `hard_tokens`: the text of backtick spans and identifiers copied verbatim, over the `hard_token_n` inputs"
        " that have one (few in chat; the owner's commit/comment set is where this matters);"
        " `placeholder`: same placeholders; `guard`: MCP guard passes (it also needs each span's backticks).",
        f"- `meaning` … `low`: {cfg['llm']['model']} judge, mean score 1–5, share with added / lost information,"
        " share scored ≤ 2. The judge is the same model as Baseline B, so it may favour B: read B's meaning"
        " score as an upper bound.",
        "",
        *table(rows),
        "",
        *calibration(rows.get("reference", {}), "own version" if owner else "real message"),
        "## Gate\n",
        f"Rules (plan section 9): beats its own base on style; style ≥ {cfg['gate']['style']:.0%}; meaning ≥"
        f" {cfg['gate']['meaning']}; added at most {cfg['gate']['added_over_reference']:.0%} above the reference's rate; meaning at most"
        f" {cfg['gate']['max_meaning_gap']} below Baseline B. A missing number fails its rule.\n",
    ]
    rules = ("beats_base_style", "style_floor", "meaning_floor", "added_vs_reference", "meaning_vs_fewshot", "pass",
             "loses_to_fewshot_on_both")
    lines += ["| system | " + " | ".join(rules) + " |", "|---" * (len(rules) + 1) + "|"]
    lines += [f"| {n} | " + " | ".join("yes" if r[k] else "no" for k in rules) + " |" for n, r in g.items()]
    if not owner:
        lines += ["", "## Not covered here\n",
                  "- Commit/comment tasks: owner-written set, `reports/eval_owner.md` (`--set owner`).",
                  "- Blind test: sheet written to `data/eval/blind.md`; results go here once filled in.",
                  "- Latency: measured on the GGUF build in Phase 6."]
    lines += ["", "## OpenAI cost\n",
              f"{spent['requests']} cached requests (Baseline B + judge, both sets), {spent['tokens']['input_tokens']:,} input /"
              f" {spent['tokens']['output_tokens']:,} output tokens, ${spent['cost_usd']} at Batch prices"
              f" in `configs/eval.yaml`.", ""]
    path = base / cfg["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    (base / cfg["outputs_dir"]).parent.joinpath("summary.json").write_text(
        json.dumps({"rows": rows, "gate": g, "cost": spent}, ensure_ascii=False, indent=1), encoding="utf-8")
    best = best_ft(cfg, rows)
    if best:
        print(examples_md(cfg, base, test, outs, best))
        if not owner:
            print(blind(cfg, base, test, outs, best))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("gen", "fewshot", "judge", "report"))
    ap.add_argument("--system", help="gen: system name from the config")
    ap.add_argument("--limit", type=int, default=None, help="gen/fewshot/judge: only the first N test examples")
    ap.add_argument("--no-wait", action="store_true",
                    help="submit, collect only the batches that have already ended, and return")
    ap.add_argument("--set", default="chat", choices=("chat", "owner"),
                    help="chat test set, or the owner-written commit/comment set")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--base", default=".", help="repo root that config paths are relative to")
    args = ap.parse_args()
    base = Path(args.base)
    cfg = for_set(load_cfg(base, args.config), base, args.set)

    if args.cmd == "gen":
        from stylesync.eval.generate import generate

        if args.system not in cfg["systems"]:
            sys.exit(f"--system must be one of: {', '.join(cfg['systems'])}")
        print(generate(cfg, load_cfg(base, cfg["train_config"]), args.system, base, args.limit))
        return
    test = read_jsonl(base / cfg["test"])[:args.limit]
    if args.cmd == "fewshot":
        fewshot(cfg, base, test, not args.no_wait)
    elif args.cmd == "judge":
        run_batch(cfg, base, judge_requests(cfg, test, outputs(cfg, base, test)), not args.no_wait)
    else:
        print(report(cfg, base, test))


if __name__ == "__main__":
    main()
