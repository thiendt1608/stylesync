"""Phase 0: run the test set through onnxruntime-genai on CPU and record latency, memory and output checks.

Usage: python spike/phase0/bench_onnx.py <onnx_model_dir> <name>
"""
import json
import resource
import sys
import time

import onnxruntime_genai as og

from common import check, load_cases, messages, save, summarize


def main() -> None:
    model_dir, name = sys.argv[1], sys.argv[2]
    t0 = time.perf_counter()
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    load_s = time.perf_counter() - t0

    rows = []
    for case in load_cases():
        prompt = tokenizer.apply_chat_template(
            json.dumps(messages(case), ensure_ascii=False), add_generation_prompt=True)
        tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(tokens) + 160)
        s = time.perf_counter()
        gen = og.Generator(model, params)
        gen.append_tokens(tokens)
        prefill_end = time.perf_counter()
        out_ids = []
        while not gen.is_done():
            gen.generate_next_token()
            out_ids.append(gen.get_next_tokens()[0])
        end = time.perf_counter()
        out = tokenizer.decode(out_ids).strip()
        decode_s = end - prefill_end
        rows.append({
            **case, "output": out, "wall_s": round(end - s, 3),
            "prompt_n": len(tokens), "prompt_ms": round((prefill_end - s) * 1000, 1),
            "predicted_n": len(out_ids), "predicted_ms": round(decode_s * 1000, 1),
            "decode_tok_s": len(out_ids) / decode_s if decode_s else None,
            **check(case["text"], out),
        })

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    if sys.platform != "darwin":
        peak_rss *= 1024
    save(name, "onnx-cpu", rows, summarize(name, "onnx-cpu", rows, load_s, peak_rss))


if __name__ == "__main__":
    main()
