"""Phase 5 metrics (plan section 9): style classifier, stylometry, hard-token / placeholder / guard checks,
judge summary and the pass gate. Pure functions over texts, no model or network."""
from collections import Counter

from stylesync.clean.drift import rates, text_metrics
from stylesync.clean.filters import PLACEHOLDER
from stylesync.mcp_server import guard
from stylesync.neutralize.validate import hard_tokens


def fit_style(mine: list[str], neutral: list[str]):
    """Classifier for "sounds like the owner": char n-gram TF-IDF + logistic regression, the owner's real
    messages against the neutral inputs of the same pairs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, min_df=2),
                        LogisticRegression(max_iter=2000, C=4.0))
    return clf.fit(mine + neutral, [1] * len(mine) + [0] * len(neutral))


def style_rate(clf, texts: list[str]) -> float:
    """Share of texts the classifier calls the owner's."""
    return round(float(clf.predict(texts).mean()), 3) if texts else 0.0


def stylometry(texts: list[str]) -> dict:
    return rates([text_metrics(t) for t in texts])


def checks(source: str, output: str) -> dict:
    """Per-example booleans: hard tokens and placeholders copied, MCP guard passes."""
    # The MCP guard flags any <NAME> in a reply because MCP inputs never hold one; test inputs do,
    # so a copied placeholder is not a leak here (placeholder preservation is its own column).
    problems = [p for p in guard(source, output) if not (p == "placeholder leaked" and PLACEHOLDER.search(source))]
    return {"hard_tokens": all(t in output for t in hard_tokens(source)),
            "placeholder": Counter(PLACEHOLDER.findall(output)) == Counter(PLACEHOLDER.findall(source)),
            "guard": not problems}


def check_rates(pairs: list[tuple[str, str]]) -> dict:
    """Pass rates of checks() over (source, output) pairs; hard_tokens only over sources that have any."""
    rows = [(checks(s, o), bool(hard_tokens(s))) for s, o in pairs]
    with_tokens = [c for c, has in rows if has]
    n = max(len(rows), 1)
    return {"hard_tokens": round(sum(c["hard_tokens"] for c in with_tokens) / max(len(with_tokens), 1), 3),
            "hard_token_n": len(with_tokens),
            "placeholder": round(sum(c["placeholder"] for c, _ in rows) / n, 3),
            "guard": round(sum(c["guard"] for c, _ in rows) / n, 3)}


def judge_summary(verdicts: list[dict]) -> dict:
    n = len(verdicts)
    if not n:
        return {"judged": 0, "meaning": None, "added": None, "lost": None, "low": None}
    return {"judged": n,
            "meaning": round(sum(v["score"] for v in verdicts) / n, 2),
            "added": round(sum(v["added"] for v in verdicts) / n, 3),
            "lost": round(sum(v["lost"] for v in verdicts) / n, 3),
            "low": round(sum(v["score"] <= 2 for v in verdicts) / n, 3)}


def gate(ft: dict, base: dict, fewshot: dict, ref: dict, g: dict) -> dict:
    """Plan section 9 for one fine-tuned system: each rule's result, and the overall pass.
    Rules whose numbers are missing (system not judged yet) count as failed."""
    def ok(cond):
        try:
            return bool(cond())
        except (KeyError, TypeError):  # a missing system or a None in the comparison
            return False

    rules = {
        "beats_base_style": ok(lambda: ft["style"] > base["style"]),
        "style_floor": ok(lambda: ft["style"] >= g["style"]),
        "meaning_floor": ok(lambda: ft["meaning"] >= g["meaning"]),
        # Relative to the owner's real messages, which the judge also flags: the neutral inputs drop details.
        "added_vs_reference": ok(lambda: ft["added"] <= ref["added"] + g["added_over_reference"]),
        "meaning_vs_fewshot": ok(lambda: ft["meaning"] >= fewshot["meaning"] - g["max_meaning_gap"]),
    }
    # Plan: losing to Baseline B on both style and meaning means the local model is not worth shipping.
    loses = ok(lambda: ft["style"] < fewshot["style"] and ft["meaning"] < fewshot["meaning"])
    return rules | {"pass": all(rules.values()), "loses_to_fewshot_on_both": loses}
