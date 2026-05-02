#!/usr/bin/env python3
"""
Run the original 3x3 Artificial Bestiary framing over the 35-word approved set.

Design:
  word × reality_status × ontology × replicate

Reality statuses:
  real, imaginary, type_of

Ontologies:
  animal, object, idea

Prompts:
  Describe a {word}. It is a real {ontology}.
  Describe a {word}. It is an imaginary {ontology}.
  Describe a {word}. It is a type of {ontology}.

Key methodological choices:
  - No system prompt.
  - One fresh single-turn request per trial.
  - No warning that the word is nonce, fictional, invented, or part of a test.
  - JSONL output, one response per line.
  - Unique trial_id/custom_id includes word + condition + replicate.
  - Resume support for sync runs.

Install:
  pip install anthropic

Auth:
  export ANTHROPIC_API_KEY="..."

Examples:
  # Dry-run manifest only
  python claude_bestiary_3x3_runner.py manifest --trials 10 --out manifest_3x3_t10.csv

  # Sequential run, resumable
  python claude_bestiary_3x3_runner.py sync \
    --model claude-sonnet-4-6 \
    --trials 10 \
    --out results_sonnet46_t10.jsonl \
    --shuffle

  # Submit as a Message Batch
  python claude_bestiary_3x3_runner.py submit-batch \
    --model claude-sonnet-4-6 \
    --trials 10 \
    --batch-meta batch_sonnet46_t10.json

  # Poll/retrieve a finished Message Batch
  python claude_bestiary_3x3_runner.py retrieve-batch \
    --batch-meta batch_sonnet46_t10.json \
    --out results_sonnet46_t10.jsonl
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import anthropic
except ImportError:
    anthropic = None


# -----------------------------
# Stimuli
# -----------------------------

WORDS: List[Dict[str, str]] = [
    # Opaque-ish
    {"word": "mavika", "word_set": "opaque_ish", "source": "user_provided_fresh_gpt", "note": "Smooth, name-like; comparatively low obvious English morphology."},
    {"word": "qelat", "word_set": "opaque_ish", "source": "user_provided_fresh_gpt", "note": "Compact Semitic-shaped form; relatively low semantic transparency."},
    {"word": "ulvani", "word_set": "opaque_ish", "source": "user_provided_fresh_gpt", "note": "Soft, name-like; Finnish-ish texture without an obvious English hook."},
    {"word": "xirumba", "word_set": "opaque_ish", "source": "user_provided_fresh_gpt", "note": "Creature/place-sounding; textured but not obviously decomposable in English."},
    {"word": "paltuk", "word_set": "opaque_ish", "source": "user_provided_fresh_gpt", "note": "Tool/object-ish compact form; Turkic-sounding."},
    {"word": "ngalepo", "word_set": "opaque_ish", "source": "user_provided_fresh_gpt", "note": "Austronesian/Bantu-ish texture; creature/place-like."},
    {"word": "zhonnek", "word_set": "opaque_ish", "source": "user_provided_fresh_gpt", "note": "Sci-fi/proper-noun texture; low obvious dictionary pull."},
    {"word": "lemavoq", "word_set": "opaque_ish", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Built to feel pronounceable but not strongly rooted; mild object/creature neutrality."},
    {"word": "sorvanu", "word_set": "opaque_ish", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Soft three-syllable form; name-like without a clear morphemic hook."},
    {"word": "qidarel", "word_set": "opaque_ish", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Compact but not too English; slight term/object feel."},
    {"word": "nuvrakel", "word_set": "opaque_ish", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Slightly technical/proper-noun shaped; intended moderate opacity."},

    # Animal-hook
    {"word": "brindleth", "word_set": "animal_hook", "source": "user_provided_fresh_gpt", "note": "Contains 'brindle'; likely animal/color cue."},
    {"word": "chamélo", "word_set": "animal_hook", "source": "user_provided_fresh_gpt", "note": "Close to chameleon; obvious animal-hook."},
    {"word": "skeldar", "word_set": "animal_hook", "source": "user_provided_fresh_gpt", "note": "Norse/fantasy creature or beast-name texture."},
    {"word": "dravosh", "word_set": "animal_hook", "source": "user_provided_fresh_gpt", "note": "Slavic/fantasy creature-villain texture."},
    {"word": "furmendi", "word_set": "animal_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Contains 'fur' and soft mammal rhythm; designed animal-biased."},
    {"word": "talonvek", "word_set": "animal_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Contains 'talon'; likely bird/reptile/predator cue."},
    {"word": "mosselk", "word_set": "animal_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Contains 'moss' + 'elk'; forest animal/ecology cue."},
    {"word": "vulpira", "word_set": "animal_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Echoes vulpine/vulpes; foxlike animal cue."},

    # Idea/neologism-hook
    {"word": "tsorim", "word_set": "idea_neologism_hook", "source": "user_provided_fresh_gpt", "note": "Hebrew-plural-looking; term/category feel."},
    {"word": "lakshun", "word_set": "idea_neologism_hook", "source": "user_provided_fresh_gpt", "note": "South Asian/English resemblance; concept/name hook."},
    {"word": "mirellek", "word_set": "idea_neologism_hook", "source": "user_provided_fresh_gpt", "note": "French/name-like; aesthetic or school-of-thought feel."},
    {"word": "borzani", "word_set": "idea_neologism_hook", "source": "user_provided_fresh_gpt", "note": "Persian/Italian/name-like; plausible surname/place/concept."},
    {"word": "framelet", "word_set": "idea_neologism_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Diminutive of 'frame'; likely to invite conceptual definition."},
    {"word": "qualvance", "word_set": "idea_neologism_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Echoes quality/valence/advance; abstract-concept hook."},
    {"word": "metavoid", "word_set": "idea_neologism_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Overt meta- + void; strongly idea/philosophy hook."},
    {"word": "synthory", "word_set": "idea_neologism_hook", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Echoes synthesis/theory; neologism/concept hook."},

    # Ethnolinguistically marked
    {"word": "tlapeco", "word_set": "ethnolinguistically_marked", "source": "user_provided_fresh_gpt", "note": "Nahuatl/Mesoamerican place/object texture."},
    {"word": "evaruq", "word_set": "ethnolinguistically_marked", "source": "user_provided_fresh_gpt", "note": "Inuit-like orthographic texture; marked and should be handled carefully."},
    {"word": "tavuni", "word_set": "ethnolinguistically_marked", "source": "user_provided_fresh_gpt", "note": "Polynesian-ish place/name texture."},
    {"word": "noyaru", "word_set": "ethnolinguistically_marked", "source": "user_provided_fresh_gpt", "note": "Japanese-ish segmentation risk; marked but usable with care."},
    {"word": "mbelako", "word_set": "ethnolinguistically_marked", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Bantu-ish phonotactic texture; marked, with nasal onset."},
    {"word": "qamirun", "word_set": "ethnolinguistically_marked", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Semitic/Arabic-ish texture; marked but avoids obvious charged roots."},
    {"word": "ayuneko", "word_set": "ethnolinguistically_marked", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Japanese/Austronesian-ish texture; soft name/place feel."},
    {"word": "xochalen", "word_set": "ethnolinguistically_marked", "source": "gpt_5_5_thinking_context_bo_chesterton", "note": "Mesoamerican/Nahuatl-ish texture; marked, likely place/ritual-object cue."},
]

REALITY_STATUSES = ["real", "imaginary", "type_of"]
ONTOLOGIES = ["animal", "object", "idea"]


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slug_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() or "x"


def make_prompt(word: str, status: str, ontology: str) -> str:
    # Keep "Describe a {word}" even for vowel-initial words to preserve comparability
    # with the original Bestiary prompt shape.
    if status == "real":
        return f"Describe a {word}. It is a real {ontology}."
    if status == "imaginary":
        return f"Describe a {word}. It is an imaginary {ontology}."
    if status == "type_of":
        return f"Describe a {word}. It is a type of {ontology}."
    raise ValueError(f"Unknown status: {status}")


def make_trial_id(word: str, word_set: str, status: str, ontology: str, replicate: int) -> str:
    return f"{word_set}::{word}::{status}_{ontology}::{replicate:03d}"


def make_custom_id(word: str, word_set: str, status: str, ontology: str, replicate: int) -> str:
    """
    Anthropic batch custom_id must match ^[a-zA-Z0-9_-]{1,64}$.
    Include a short hash to protect uniqueness after ASCII folding.
    """
    basis = f"{word_set}|{word}|{status}|{ontology}|{replicate:03d}"
    h = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:8]
    raw = f"{slug_ascii(word_set)[:8]}_{slug_ascii(word)[:18]}_{status[:3]}_{ontology[:3]}_{replicate:03d}_{h}"
    return raw[:64]


def build_manifest(trials: int, model: str, temperature: float, max_tokens: int, shuffle: bool = False, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in WORDS:
        for status in REALITY_STATUSES:
            for ontology in ONTOLOGIES:
                for rep in range(1, trials + 1):
                    prompt = make_prompt(item["word"], status, ontology)
                    rows.append({
                        "trial_id": make_trial_id(item["word"], item["word_set"], status, ontology, rep),
                        "custom_id": make_custom_id(item["word"], item["word_set"], status, ontology, rep),
                        "word": item["word"],
                        "word_set": item["word_set"],
                        "word_source": item["source"],
                        "word_note": item["note"],
                        "status": status,
                        "ontology": ontology,
                        "condition": f"{status}_{ontology}",
                        "replicate": rep,
                        "prompt": prompt,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    })
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def write_manifest_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id", "custom_id", "word", "word_set", "word_source", "word_note",
        "status", "ontology", "condition", "replicate", "prompt",
        "model", "temperature", "max_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_completed_trial_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                tid = obj.get("trial_id")
                if tid:
                    done.add(tid)
            except json.JSONDecodeError:
                continue
    return done


def obj_to_plain(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, list, dict)):
        if isinstance(obj, list):
            return [obj_to_plain(x) for x in obj]
        if isinstance(obj, dict):
            return {k: obj_to_plain(v) for k, v in obj.items()}
        return obj
    for method in ("model_dump", "to_dict", "dict"):
        if hasattr(obj, method):
            try:
                return getattr(obj, method)()
            except TypeError:
                pass
    if hasattr(obj, "to_json"):
        try:
            return json.loads(obj.to_json())
        except Exception:
            pass
    # Last resort for SDK objects.
    if hasattr(obj, "__dict__"):
        return {k: obj_to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def extract_text_from_message_plain(message: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for block in message.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(block.get("text", ""))
    return "".join(chunks)


def require_anthropic() -> None:
    if anthropic is None:
        raise SystemExit("Missing dependency: pip install anthropic")


def make_client() -> Any:
    require_anthropic()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def call_with_retries(client: Any, params: Dict[str, Any], retries: int, base_sleep: float) -> Any:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return client.messages.create(**params)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = base_sleep * (2 ** attempt) + random.random()
            print(f"[warn] API error on attempt {attempt + 1}; retrying in {sleep_s:.1f}s: {exc}", file=sys.stderr)
            time.sleep(sleep_s)
    raise last_exc


# -----------------------------
# Commands
# -----------------------------

def cmd_manifest(args: argparse.Namespace) -> None:
    rows = build_manifest(
        trials=args.trials,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    write_manifest_csv(rows, Path(args.out))
    print(f"Wrote {len(rows)} trials to {args.out}")


def cmd_sync(args: argparse.Namespace) -> None:
    client = make_client()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_manifest(
        trials=args.trials,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    done = read_completed_trial_ids(out_path) if args.resume else set()
    if done:
        print(f"Resume: found {len(done)} completed trial_ids in {out_path}")

    total = len(rows)
    remaining = [r for r in rows if r["trial_id"] not in done]
    print(f"Running {len(remaining)} remaining trials out of {total}")

    with out_path.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(remaining, start=1):
            params = {
                "model": row["model"],
                "max_tokens": int(row["max_tokens"]),
                "temperature": float(row["temperature"]),
                "messages": [{"role": "user", "content": row["prompt"]}],
            }
            started_at = utc_now_iso()
            try:
                message = call_with_retries(client, params, retries=args.retries, base_sleep=args.retry_sleep)
                ended_at = utc_now_iso()
                plain = obj_to_plain(message)
                result = {
                    **row,
                    "run_mode": "sync",
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "ok": True,
                    "response_text": extract_text_from_message_plain(plain),
                    "response_raw": plain,
                    "request_id": getattr(message, "_request_id", None),
                }
            except Exception as exc:
                ended_at = utc_now_iso()
                result = {
                    **row,
                    "run_mode": "sync",
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{idx}/{len(remaining)}] {row['custom_id']} ok={result.get('ok')}")
            if args.sleep > 0:
                time.sleep(args.sleep)


def cmd_submit_batch(args: argparse.Namespace) -> None:
    client = make_client()
    rows = build_manifest(
        trials=args.trials,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    requests = []
    for row in rows:
        requests.append({
            "custom_id": row["custom_id"],
            "params": {
                "model": row["model"],
                "max_tokens": int(row["max_tokens"]),
                "temperature": float(row["temperature"]),
                "messages": [{"role": "user", "content": row["prompt"]}],
            }
        })

    # Optional single request validation before batch submit.
    if args.validate_one:
        first = requests[0]["params"]
        print("Validating request shape with one synchronous Messages API call...")
        test_message = client.messages.create(**first)
        print(f"Validation response stop_reason={getattr(test_message, 'stop_reason', None)}")

    message_batch = client.messages.batches.create(requests=requests)
    batch_plain = obj_to_plain(message_batch)

    meta = {
        "created_at": utc_now_iso(),
        "batch": batch_plain,
        "batch_id": batch_plain.get("id"),
        "trial_count": len(rows),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "manifest": rows,
    }
    meta_path = Path(args.batch_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Submitted batch {meta.get('batch_id')} with {len(rows)} requests")
    print(f"Wrote batch metadata to {meta_path}")


def cmd_retrieve_batch(args: argparse.Namespace) -> None:
    client = make_client()
    meta_path = Path(args.batch_meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    batch_id = args.batch_id or meta.get("batch_id")
    if not batch_id:
        raise SystemExit("No batch_id supplied and none found in batch_meta.")

    if args.wait:
        while True:
            mb = client.messages.batches.retrieve(batch_id)
            plain = obj_to_plain(mb)
            status = plain.get("processing_status")
            counts = plain.get("request_counts")
            print(f"Batch {batch_id}: status={status}, counts={counts}")
            if status == "ended":
                break
            time.sleep(args.poll_seconds)

    manifest_by_custom_id = {row["custom_id"]: row for row in meta.get("manifest", [])}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in client.messages.batches.results(batch_id):
            n += 1
            plain_item = obj_to_plain(item)
            custom_id = plain_item.get("custom_id")
            row_meta = manifest_by_custom_id.get(custom_id, {})
            result = plain_item.get("result", {}) or {}
            out: Dict[str, Any] = {
                **row_meta,
                "run_mode": "batch",
                "retrieved_at": utc_now_iso(),
                "batch_id": batch_id,
                "batch_result_type": result.get("type"),
                "batch_raw": plain_item,
            }
            if result.get("type") == "succeeded":
                message = result.get("message", {}) or {}
                out["ok"] = True
                out["response_text"] = extract_text_from_message_plain(message)
                out["response_raw"] = message
            else:
                out["ok"] = False
                out["error"] = result.get("error")
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {n} batch results to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run 3x3 Artificial Bestiary trials over the 35-word approved set.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
                        help="Claude model name. Default: env CLAUDE_MODEL or claude-sonnet-4-6.")
        sp.add_argument("--trials", type=int, default=10, help="Replicates per word × condition. Default: 10.")
        sp.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Default: 1.0.")
        sp.add_argument("--max-tokens", type=int, default=700, help="Max output tokens per trial. Default: 700.")
        sp.add_argument("--shuffle", action="store_true", help="Shuffle trial order.")
        sp.add_argument("--seed", type=int, default=None, help="Random seed for shuffling.")

    m = sub.add_parser("manifest", help="Write a CSV manifest without calling the API.")
    add_common(m)
    m.add_argument("--out", default="manifest_3x3.csv")
    m.set_defaults(func=cmd_manifest)

    s = sub.add_parser("sync", help="Run trials sequentially with the Messages API.")
    add_common(s)
    s.add_argument("--out", default="results_3x3.jsonl")
    s.add_argument("--resume", action="store_true", default=True, help="Skip trial_ids already present in output. Default: true.")
    s.add_argument("--no-resume", dest="resume", action="store_false")
    s.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    s.add_argument("--retries", type=int, default=5, help="Retries per request. Default: 5.")
    s.add_argument("--retry-sleep", type=float, default=2.0, help="Base retry sleep seconds. Default: 2.")
    s.set_defaults(func=cmd_sync)

    b = sub.add_parser("submit-batch", help="Submit all trials as an Anthropic Message Batch.")
    add_common(b)
    b.add_argument("--batch-meta", default="batch_meta_3x3.json", help="Where to save batch id and manifest.")
    b.add_argument("--validate-one", action="store_true", help="Run one synchronous request first to validate request shape.")
    b.set_defaults(func=cmd_submit_batch)

    r = sub.add_parser("retrieve-batch", help="Retrieve a completed Message Batch into JSONL.")
    r.add_argument("--batch-meta", default="batch_meta_3x3.json")
    r.add_argument("--batch-id", default=None)
    r.add_argument("--out", default="results_3x3_batch.jsonl")
    r.add_argument("--wait", action="store_true", help="Poll until batch processing_status is ended.")
    r.add_argument("--poll-seconds", type=int, default=60)
    r.set_defaults(func=cmd_retrieve_batch)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
