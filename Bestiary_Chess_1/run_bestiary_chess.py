#!/usr/bin/env python3
"""
Run the 3x3 Bestiary trials from a CSV manifest, sequentially, with live progress.

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python run_bestiary.py \
      --manifest bestiary_3x3_manifest_35_words_1rep.csv \
      --out results.jsonl \
      --model claude-sonnet-4-5 \
      --replicates 1

Resume is automatic: re-run the same command after an interruption and it picks up
where it left off (anything already in results.jsonl is skipped).
"""

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path

import anthropic


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def expand_replicates(rows: list[dict], replicates: int) -> list[dict]:
    """Return rows × replicates, with a fresh trial_id on each."""
    out = []
    for rep in range(1, replicates + 1):
        for r in rows:
            r2 = dict(r)
            r2["replicate"] = rep
            r2["trial_id"] = f"{r['word']}::{r['condition']}::{rep:03d}"
            out.append(r2)
    return out


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                tid = json.loads(line).get("trial_id")
                if tid:
                    done.add(tid)
            except json.JSONDecodeError:
                pass
    return done


def call_with_retries(client, params: dict, max_tries: int = 3) -> str:
    """Return the text of the response, or raise after exhausting retries."""
    for attempt in range(1, max_tries + 1):
        try:
            msg = client.messages.create(**params)
            return "".join(b.text for b in msg.content if b.type == "text")
        except (anthropic.APIConnectionError, anthropic.APITimeoutError,
                anthropic.RateLimitError, anthropic.InternalServerError) as e:
            if attempt == max_tries:
                raise
            sleep_s = 2 ** attempt + random.random()
            print(f"  ! {type(e).__name__}, retrying in {sleep_s:.1f}s", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                    help="CSV with columns: word, condition, prompt, ...")
    ap.add_argument("--out", required=True, type=Path, help="Output JSONL.")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--replicates", type=int, default=1,
                    help="How many times to run each row in the manifest.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle trial order so partial runs cover all conditions.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to wait between trials.")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic()

    rows = expand_replicates(load_manifest(args.manifest), args.replicates)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    done = already_done(args.out)
    todo = [r for r in rows if r["trial_id"] not in done]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {len(rows)} trials total. "
          f"Already done: {len(done)}. To run: {len(todo)}.")
    print(f"Model: {args.model}  |  Output: {args.out}\n")

    n_ok = 0
    n_err = 0
    t0 = time.time()

    with args.out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            params = {
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "messages": [{"role": "user", "content": row["prompt"]}],
            }
            started = now_iso()
            try:
                text = call_with_retries(client, params)
                rec = {**row, "model": args.model, "temperature": args.temperature,
                       "max_tokens": args.max_tokens, "started_at": started,
                       "ended_at": now_iso(), "ok": True, "response_text": text}
                n_ok += 1
                preview = text.replace("\n", " ")[:70]
                status = f"ok    | {preview}…"
            except Exception as e:
                rec = {**row, "model": args.model, "temperature": args.temperature,
                       "max_tokens": args.max_tokens, "started_at": started,
                       "ended_at": now_iso(), "ok": False,
                       "error_type": type(e).__name__, "error": str(e)}
                n_err += 1
                status = f"ERROR | {type(e).__name__}: {e}"

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (len(todo) - i) / rate if rate else 0
            print(f"[{i:>4}/{len(todo)}] {row['word']:<10} {row['condition']:<16} "
                  f"({rate:.1f}/s, ETA {eta/60:.1f}m)  {status}")

            if args.sleep:
                time.sleep(args.sleep)

    print(f"\nDone. ok={n_ok}  err={n_err}  total_written={n_ok + n_err}")


if __name__ == "__main__":
    main()
