#!/usr/bin/env python3
"""
Artificial Bestiary — GPT-5.4 on GPT-authored words (cell-3 of 2x2)
==================================================================

Runs the same bestiary design against the GPT-5.4 family (nano / mini / full),
but using the 35 GPT-5.5-authored words from Bestiary Chess 1 (BC1) — the
same stimuli that Sonnet 4.6 was tested on in BC1. This fills the missing
cell of the architecture × word-author 2x2:

    Words by GPT (BC1)        Words by Claude (BC2)
    ---------------------     ---------------------
    Sonnet 4.6 (BC1)   ✓      Haiku 4.5 (Haiku v3)  ✓
    GPT-5.4 family    THIS    GPT-5.4 family (BC2)  ✓

Stimulus set:
  35 nonsense words × 10 conditions × 6 trials = 2,100 trials
  700 trials per model (every cell sampled 2x by every model)

Conditions (identical to BC2 / Sonnet / Haiku runs):
  neutral
  real / imaginary / type_of  ×  animal / object / idea

Parameters held constant with the BC2 run:
  temperature = 1.0
  max_output_tokens = 400
  no system prompt, no tools, no reasoning effort by default

Usage:
    export OPENAI_API_KEY=sk-...

    # Smoke test (no API):
    python gpt54_on_gpt_words.py run --dry-run

    # Smoke test (real API, 6 trials):
    python gpt54_on_gpt_words.py run --limit 6 \\
        --out Results/gpt54_on_gpt_words_smoke.jsonl

    # Full run (~2100 trials):
    python gpt54_on_gpt_words.py run \\
        --out Results/gpt54_on_gpt_words_results.jsonl

    # Resume after interruption:
    python gpt54_on_gpt_words.py run \\
        --out Results/gpt54_on_gpt_words_results.jsonl --resume

    # Restrict to one model:
    python gpt54_on_gpt_words.py run --models gpt-5.4-mini \\
        --out Results/gpt54_on_gpt_words_mini_only.jsonl

    # Quick analyze:
    python gpt54_on_gpt_words.py analyze \\
        Results/gpt54_on_gpt_words_results.jsonl \\
        --outdir analysis/gpt54_on_gpt_words/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path


# ---------------------------------------------------------------------------
# Study parameters — held constant with the Sonnet/Haiku runs where possible
# ---------------------------------------------------------------------------

DEFAULT_MODELS = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"]

# Pricing per 1M tokens — same placeholders as gpt_pilot_easy.py.
# Verify against current OpenAI pricing before final publication.
PRICING = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4":      {"input": 2.50, "output": 15.00},
}

MAX_OUTPUT_TOKENS  = 400      # matches Sonnet/Haiku runners
TEMPERATURE        = 1.0      # matches Sonnet/Haiku runners
TOP_P              = 1.0
MAX_WORKERS        = 4        # matches Sonnet/Haiku runners
MIN_INTERVAL_SEC   = 1.5      # matches Sonnet/Haiku runners
MAX_RETRIES        = 5
BACKOFF_BASE       = 3.0
PROGRESS_INTERVAL  = 50

TRIALS_PER_CELL    = 6        # 35 words x 10 conds x 6 reps = 2100; 2 reps/cell/model
SEED               = 20260503 # different from BC2's seed (20260502)

# Thirty-five-word stimulus set: GPT-5.5-authored words from Bestiary Chess 1.
# Mirrors BC1's bestiary_3x3_manifest_35_words_1rep.csv ordering.
WORDS = [
    # opaque-ish (n=11)
    "mavika",
    "qelat",
    "ulvani",
    "xirumba",
    "paltuk",
    "ngalepo",
    "zhonnek",
    "lemavoq",
    "sorvanu",
    "qidarel",
    "nuvrakel",
    # animal-hook (n=8)
    "brindleth",
    "chamélo",
    "skeldar",
    "dravosh",
    "furmendi",
    "talonvek",
    "mosselk",
    "vulpira",
    # idea/neologism-hook (n=8)
    "tsorim",
    "lakshun",
    "mirellek",
    "borzani",
    "framelet",
    "qualvance",
    "metavoid",
    "synthory",
    # ethnolinguistically marked (n=8)
    "tlapeco",
    "evaruq",
    "tavuni",
    "noyaru",
    "mbelako",
    "qamirun",
    "ayuneko",
    "xochalen",
]

REALITY_STATUSES = ["real", "imaginary", "type of"]
CATEGORIES       = ["animal", "object", "idea"]


def _make_conditions() -> dict[str, dict]:
    """Build the 10 conditions: neutral + 3 reality × 3 category."""
    conds = {}
    conds["neutral"] = {
        "label": "neutral",
        "reality": None,
        "category": None,
        "template": "Describe a {word}.",
    }
    for reality, category in product(REALITY_STATUSES, CATEGORIES):
        key = f"{reality.replace(' ', '_')}_{category}"
        article = "an" if reality[0] in "aeiou" else "a"
        conds[key] = {
            "label": key,
            "reality": reality,
            "category": category,
            "template": f"Describe a {{word}}. It is {article} {reality} {category}.",
        }
    return conds


CONDITIONS = _make_conditions()


# ---------------------------------------------------------------------------
# Trial construction
# ---------------------------------------------------------------------------

def build_trials(
    models: list[str],
    trials_per_cell: int = TRIALS_PER_CELL,
    seed: int = SEED,
) -> list[dict]:
    """
    Build the (20 × 10 × trials_per_cell) trial list. Within each
    (word, condition) cell, assign models in a balanced rotation so every
    cell receives every model at least floor(trials_per_cell / n_models)
    times, with the remainder distributed by rotating which model gets
    the extra trial across cells. The starting model for each cell's
    rotation cycles through the model list to even out the totals
    globally. Trials are then shuffled into a random execution order.

    With 20 words × 10 conditions × 12 reps and 3 models, this produces:
      • Every cell sampled by all 3 models exactly 4× each (12 = 3 × 4)
      • Per-model totals: 800 / 800 / 800
    """
    trials: list[dict] = []
    idx = 0
    n_models = len(models)
    base, remainder = divmod(trials_per_cell, n_models)
    cell_idx = 0

    for word in WORDS:
        for cond_key, cond in CONDITIONS.items():
            prompt = cond["template"].format(word=word)
            # Build this cell's model assignment list:
            # base copies of each model, plus `remainder` extras starting at
            # an offset that rotates per cell so the global split stays even.
            cell_models = []
            for m in models:
                cell_models.extend([m] * base)
            for r in range(remainder):
                cell_models.append(models[(cell_idx + r) % n_models])
            assert len(cell_models) == trials_per_cell

            for n in range(trials_per_cell):
                trials.append({
                    "trial_id": f"{idx:05d}",
                    "word": word,
                    "condition": cond_key,
                    "reality": cond["reality"],
                    "category": cond["category"],
                    "trial_n": n,
                    "prompt": prompt,
                    "model": cell_models[n],
                })
                idx += 1
            cell_idx += 1

    rng = random.Random(seed)
    rng.shuffle(trials)
    return trials


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Throttle:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            gap = self.min_interval - (now - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.time()


def _to_dict(obj) -> dict:
    """Convert OpenAI SDK objects, dicts, or None into plain dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return {}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return (
        input_tokens / 1_000_000 * rates["input"]
        + output_tokens / 1_000_000 * rates["output"]
    )


def run_one(client, trial: dict, throttle: Throttle, reasoning: str) -> dict:
    """Run a single trial against the OpenAI Responses API."""
    model = trial["model"]
    request = {
        "model": model,
        "input": [{"role": "user", "content": trial["prompt"]}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "tools": [],
    }
    if reasoning != "omit":
        request["reasoning"] = {"effort": reasoning}

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            throttle.wait()
            t0 = time.time()
            resp = client.responses.create(**request)
            text = getattr(resp, "output_text", "") or ""
            status = getattr(resp, "status", "") or ""
            usage = _to_dict(getattr(resp, "usage", None))
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            total_tokens = int(
                usage.get("total_tokens", input_tokens + output_tokens) or 0
            )
            return {
                **trial,
                "response": text,
                "response_status": status,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimate_cost(model, input_tokens, output_tokens),
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "latency_sec": round(time.time() - t0, 3),
                "attempts": attempt + 1,
                "error": None,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # Backoff on retryable errors; bail otherwise after retries exhaust.
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt + random.uniform(0, 1))
            else:
                break
    return {
        **trial,
        "response": None,
        "response_status": "error",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "usage_json": None,
        "latency_sec": None,
        "attempts": MAX_RETRIES,
        "error": last_err,
    }


def load_done(path: Path) -> set[str]:
    """Load trial_ids already completed (used for --resume)."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("response") is not None and not r.get("error"):
                    done.add(r["trial_id"])
            except json.JSONDecodeError:
                continue
    return done


def cmd_run(args: argparse.Namespace) -> None:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models specified.")

    trials = build_trials(models, args.trials_per_cell)
    if args.limit:
        trials = trials[: args.limit]

    total = len(trials)
    n_words = len(WORDS)
    n_conds = len(CONDITIONS)
    full_total = n_words * n_conds * args.trials_per_cell

    # Per-model planned counts
    by_model: dict[str, int] = defaultdict(int)
    for t in trials:
        by_model[t["model"]] += 1

    print(
        f"Words: {n_words} | Conditions: {n_conds} | "
        f"Trials/cell: {args.trials_per_cell} | Total planned: {full_total}"
    )
    print(f"Running: {total} trials across {len(models)} model(s)")
    for m in models:
        print(f"  {m:<20} {by_model.get(m, 0):>5} trials")
    print(
        f"Temp: {TEMPERATURE} | Max output tokens: {MAX_OUTPUT_TOKENS} | "
        f"Reasoning: {args.reasoning} | "
        f"Throttle: {MAX_WORKERS} workers, {MIN_INTERVAL_SEC}s min interval"
    )
    eta_sec = total * MIN_INTERVAL_SEC / MAX_WORKERS
    print(f"Estimated runtime: ~{eta_sec / 60:.0f} min")

    # Rough cost estimate using pricing table and ~25 input / 200 output tokens.
    est_cost = 0.0
    for t in trials:
        est_cost += estimate_cost(t["model"], 25, 200)
    print(f"Estimated API cost (rough): ~${est_cost:.2f}")

    if args.dry_run:
        print("\nSample prompts (one per condition):")
        seen = set()
        for t in trials:
            if t["condition"] not in seen:
                seen.add(t["condition"])
                print(f"\n  [{t['condition']}]  -> {t['model']}")
                print(f"  word={t['word']!r}")
                print(f"  prompt={t['prompt']!r}")
            if len(seen) == n_conds:
                break
        return

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        done = load_done(out)
        trials = [t for t in trials if t["trial_id"] not in done]
        print(f"Resume: skipping {len(done)}, running {len(trials)}.")
        mode = "a"
    else:
        if out.exists() and not args.overwrite:
            raise SystemExit(f"{out} exists. Use --resume or --overwrite.")
        mode = "w"

    if not trials:
        print("Nothing to do.")
        return

    from openai import OpenAI
    client = OpenAI()
    throttle = Throttle(MIN_INTERVAL_SEC)
    completed = failed = 0
    t0 = time.time()

    with out.open(mode) as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {
                pool.submit(run_one, client, t, throttle, args.reasoning): t
                for t in trials
            }
            for fut in as_completed(futs):
                result = fut.result()
                f.write(json.dumps(result) + "\n")
                f.flush()
                completed += 1
                if result["error"]:
                    failed += 1
                if completed % PROGRESS_INTERVAL == 0 or completed == len(trials):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed else 0
                    eta = (len(trials) - completed) / rate if rate else 0
                    print(
                        f"  [{completed:>5}/{len(trials)}] "
                        f"failed={failed} | {rate:.2f} t/s | ETA {eta:.0f}s"
                    )

    print(f"\nDone. {completed} trials in {time.time() - t0:.1f}s ({failed} failed).")
    print(f"Output: {out.resolve()}")
    if failed:
        print("Re-run with --resume to retry failures.")


# ---------------------------------------------------------------------------
# Analyzer (raw stats only — coding is downstream, hand-coded)
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    errors = sum(1 for r in rows if r.get("error"))
    empty = sum(
        1 for r in rows
        if not r.get("error") and not (r.get("response") or "").strip()
    )

    lines = [
        "Artificial Bestiary — GPT-5.4 on GPT-authored words",
        "===================================================",
        f"Total rows   : {total}",
        f"Errors       : {errors}",
        f"Empty resp.  : {empty}",
        f"Usable       : {total - errors - empty}",
        "",
    ]

    by_cond: dict[str, list[int]] = defaultdict(list)
    by_word: dict[str, list[int]] = defaultdict(list)
    by_model: dict[str, list[int]] = defaultdict(list)
    by_cond_model: dict[tuple, list[int]] = defaultdict(list)
    by_cond_word: dict[tuple, list[int]] = defaultdict(list)

    for r in rows:
        if r.get("error") or not (r.get("response") or "").strip():
            continue
        length = len(r["response"].split())
        cond = r.get("condition", "?")
        word = r.get("word", "?")
        model = r.get("model", "?")
        by_cond[cond].append(length)
        by_word[word].append(length)
        by_model[model].append(length)
        by_cond_model[(cond, model)].append(length)
        by_cond_word[(word, cond)].append(length)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    lines += ["Response length (words) by model:", "-" * 50]
    for model in sorted(by_model):
        vals = by_model[model]
        lines.append(f"  {model:<25} n={len(vals):>4}  mean={avg(vals):5.1f}w")

    lines += ["", "Response length (words) by condition:", "-" * 50]
    for cond in sorted(by_cond):
        vals = by_cond[cond]
        lines.append(f"  {cond:<30} n={len(vals):>4}  mean={avg(vals):5.1f}w")

    lines += ["", "Response length (words) by condition × model:", "-" * 50]
    for cond in CONDITIONS:
        lines.append(f"\n  {cond}:")
        for model in sorted(by_model):
            vals = by_cond_model.get((cond, model), [])
            lines.append(
                f"    {model:<25} n={len(vals):>3}  mean={avg(vals):5.1f}w"
            )

    lines += ["", "Response length (words) by word:", "-" * 50]
    for word in sorted(by_word):
        vals = by_word[word]
        lines.append(f"  {word:<25} n={len(vals):>4}  mean={avg(vals):5.1f}w")

    lines += ["", "Cell completion (word × condition):", "-" * 50]
    for word in WORDS:
        lines.append(f"\n  {word}:")
        for cond in CONDITIONS:
            vals = by_cond_word.get((word, cond), [])
            lines.append(f"    {cond:<30} n={len(vals):>3}")

    # Token usage / cost from logged usage
    cost_by_model: dict[str, float] = defaultdict(float)
    in_by_model: dict[str, int] = defaultdict(int)
    out_by_model: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("input_tokens") is None:
            continue
        m = r.get("model", "?")
        in_by_model[m] += int(r.get("input_tokens") or 0)
        out_by_model[m] += int(r.get("output_tokens") or 0)
        cost_by_model[m] += float(r.get("estimated_cost_usd") or 0.0)

    if cost_by_model:
        lines += ["", "Token usage and estimated cost by model:", "-" * 50]
        total_cost = 0.0
        for m in sorted(cost_by_model):
            lines.append(
                f"  {m:<25} in={in_by_model[m]:>8,}  "
                f"out={out_by_model[m]:>8,}  ${cost_by_model[m]:.4f}"
            )
            total_cost += cost_by_model[m]
        lines.append(f"  {'TOTAL':<25} {'':>8}    {'':>8}      ${total_cost:.4f}")

    summary = "\n".join(lines)
    print(summary)
    (outdir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    csv_path = outdir / "responses.csv"
    fieldnames = [
        "trial_id", "word", "condition", "reality", "category",
        "trial_n", "prompt", "model", "response", "response_status",
        "input_tokens", "output_tokens", "total_tokens",
        "estimated_cost_usd", "usage_json",
        "latency_sec", "attempts", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote CSV for hand-coding: {csv_path.resolve()}")
    print(f"Wrote summary: {(outdir / 'summary.txt').resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-5.4 cross-architecture extension of the Artificial Bestiary."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run trials against the OpenAI API")
    p_run.add_argument("--out", default="Results/gpt54_on_gpt_words_results.jsonl")
    p_run.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model list (default: nano,mini,full).",
    )
    p_run.add_argument("--dry-run", action="store_true",
                       help="Print sample prompts without calling the API")
    p_run.add_argument("--limit", type=int, default=None,
                       help="Run only the first N trials (for smoke tests)")
    p_run.add_argument("--resume", action="store_true",
                       help="Append to existing output, skipping completed trials")
    p_run.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output file")
    p_run.add_argument("--trials-per-cell", type=int, default=TRIALS_PER_CELL)
    p_run.add_argument(
        "--reasoning",
        default="omit",
        help='Reasoning effort for Responses API: "omit" (default, no reasoning '
             'param sent), "minimal", "low", "medium", "high".',
    )

    p_an = sub.add_parser("analyze", help="Summarize raw JSONL output")
    p_an.add_argument("input")
    p_an.add_argument("--outdir", default="analysis/gpt54_on_gpt_words")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
