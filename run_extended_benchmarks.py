"""
run_extended_benchmarks.py
==========================
Task 4: Run GSM8K, ARC-Challenge, and HellaSwag on the 5 models
that have reliable real MMLU data.

REQUIRES: Eagle WiFi (or any network without the 407 proxy block)
          API keys in .env:  GROQ_API_KEY, MISTRAL_API_KEY

Benchmarks:
  GSM8K          - Grade-school math reasoning (1,319 test questions)
  ARC-Challenge  - Advanced science MCQ (1,172 questions)
  HellaSwag      - Commonsense NLI MCQ (10,042 val; we sample N)

Usage:
  # Dry run - checks keys + downloads, no API calls
  .venv\\Scripts\\python run_extended_benchmarks.py --dry-run

  # Full run (recommended: 200 questions per benchmark, ~45min)
  .venv\\Scripts\\python run_extended_benchmarks.py --n 200 --run

  # Quick smoke test
  .venv\\Scripts\\python run_extended_benchmarks.py --n 20 --run

  # Specific benchmark only
  .venv\\Scripts\\python run_extended_benchmarks.py --benchmarks gsm8k --n 200 --run

Output: results/extended/  (EvalCard JSON per model per benchmark)
        results/extended_summary.json  (aggregate table, LaTeX-ready)

Author: James Bond 🐶  (Satish Namballa, Walmart Global Tech, June 2025)
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── Bootstrap: load .env before importing API clients ──────────────────────────────────
def _bootstrap() -> None:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    # Inject Windows cert store FIRST — handles Walmart proxy SSL intercept
    try:
        import truststore
        truststore.inject_into_ssl()
        print("  [truststore] Windows cert store injected", flush=True)
    except ImportError:
        pass

_bootstrap()
# ──────────────────────────────────────────────────────────────────────────────

# Walmart proxy (loaded from .env)
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "extended"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry (only models with reliable option1_submission EvalCards) ────
MODELS = {
    "mistral-small": {
        "client":   "mistral",
        "model_id": "mistral-small-latest",
        "label":    "Mistral-Small-3.1",
        "api_key":  os.environ.get("MISTRAL_API_KEY", ""),
        "base_url": "https://api.mistral.ai/v1",
        "reasoning": False,
    },
    "llama-3.3-70b": {
        "client":   "groq",
        "model_id": "llama-3.3-70b-versatile",
        "label":    "Llama-3.3-70B",
        "api_key":  os.environ.get("GROQ_API_KEY", ""),
        "base_url": "https://api.groq.com/openai/v1",
        "reasoning": False,
    },
    "llama-4-scout": {
        "client":   "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "label":    "Llama-4-Scout-17B",
        "api_key":  os.environ.get("GROQ_API_KEY", ""),
        "base_url": "https://api.groq.com/openai/v1",
        "reasoning": True,
    },
    "llama-3.1-8b": {
        "client":   "groq",
        "model_id": "llama-3.1-8b-instant",
        "label":    "Llama-3.1-8B",
        "api_key":  os.environ.get("GROQ_API_KEY", ""),
        "base_url": "https://api.groq.com/openai/v1",
        "reasoning": False,
    },
    "gpt-oss-120b": {
        "client":   "groq",
        "model_id": "openai/gpt-oss-120b",
        "label":    "GPT-OSS-120B",
        "api_key":  os.environ.get("GROQ_API_KEY", ""),
        "base_url": "https://api.groq.com/openai/v1",
        "reasoning": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK LOADERS  (local cache files — no HuggingFace dependency)
# ═══════════════════════════════════════════════════════════════════════════

CACHE_DIR = HERE / "benchmark_cache"


def load_gsm8k(n: int) -> list[dict]:
    """Load GSM8K from local cache (run download_benchmarks.py first)."""
    path = CACHE_DIR / "gsm8k_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: .venv\\Scripts\\python download_benchmarks.py")
    print(f"  Loading GSM8K from {path}...", end=" ", flush=True)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            ans = m.group(1).replace(",", "").strip() if m else \
                  row["answer"].split("\n")[-1].strip()
            items.append({"id": f"gsm8k_{len(items)}",
                          "question": row["question"],
                          "answer": ans, "type": "open_ended_math"})
    rng = random.Random(SEED)
    rng.shuffle(items)
    items = items[:n]
    print(f"loaded {len(items)} questions.")
    return items


def load_arc_challenge(n: int) -> list[dict]:
    """Load ARC-Challenge from local cache."""
    path = CACHE_DIR / "arc_challenge_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: .venv\\Scripts\\python download_benchmarks.py")
    print(f"  Loading ARC-Challenge from {path}...", end=" ", flush=True)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            choices  = row["choices"]
            labels   = choices.get("label",  choices.get("labels", []))
            texts    = choices.get("text",   choices.get("texts",  []))
            choice_str = "\n".join(f"{lbl}. {txt}"
                                    for lbl, txt in zip(labels, texts))
            items.append({"id": row.get("id", f"arc_{len(items)}"),
                          "question": f"{row['question']}\n{choice_str}",
                          "answer": row["answerKey"],
                          "type": "mcq"})
    rng = random.Random(SEED)
    rng.shuffle(items)
    items = items[:n]
    print(f"loaded {len(items)} questions.")
    return items


def load_hellaswag(n: int) -> list[dict]:
    """Load HellaSwag from local cache."""
    path = CACHE_DIR / "hellaswag_val.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: .venv\\Scripts\\python download_benchmarks.py")
    print(f"  Loading HellaSwag from {path}...", end=" ", flush=True)
    label_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            endings    = row["endings"]
            choice_str = "\n".join(f"{chr(65+i)}. {e}"
                                    for i, e in enumerate(endings))
            items.append({"id": f"hellaswag_{len(items)}",
                          "question": (f"Context: {row['activity_label']}: "
                                       f"{row['ctx']}\n"
                                       f"Which ending best completes the activity?\n"
                                       f"{choice_str}"),
                          "answer": label_map.get(str(row["label"]), "A"),
                          "type": "mcq"})
    rng = random.Random(SEED)
    rng.shuffle(items)
    items = items[:n]
    print(f"loaded {len(items)} questions.")
    return items


BENCHMARK_LOADERS = {
    "gsm8k":         load_gsm8k,
    "arc-challenge":  load_arc_challenge,
    "hellaswag":     load_hellaswag,
}


# ═══════════════════════════════════════════════════════════════════════════════
# API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def make_client(model_cfg: dict):
    """Create an OpenAI-compatible client with proxy + truststore."""
    from openai import OpenAI  # type: ignore
    import httpx
    http_client = httpx.Client(proxy=PROXY) if PROXY else None
    return OpenAI(
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        http_client=http_client,
    )


def call_model(client, model_cfg: dict, prompt: str,
               system_content: str = None,
               max_tokens: int = 256, retries: int = 3) -> tuple[str, float]:
    """
    Call the model and return (response_text, latency_ms).
    Handles 429 rate-limit retries with exponential backoff.
    """
    if system_content is None:
        system_content = ("Answer the question. For multiple choice, "
                          "respond with ONLY the letter (A/B/C/D). "
                          "For math problems, respond with ONLY the "
                          "final numeric answer.")
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model_cfg["model_id"],
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            text = resp.choices[0].message.content or ""
            return text.strip(), latency_ms
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"\n    [429 rate limit] waiting {wait}s...",
                      end="", flush=True)
                time.sleep(wait)
            else:
                return f"ERROR: {e}", 0.0
    return "ERROR: max retries", 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

_LETTER_PAT = re.compile(r"\b([A-D])\b")

def score_answer(predicted: str, gold: str, q_type: str) -> float:
    """Return 1.0 if correct, 0.0 otherwise."""
    predicted = predicted.strip()
    if q_type == "mcq":
        # Extract first letter A-D
        m = _LETTER_PAT.search(predicted)
        pred_letter = m.group(1).upper() if m else predicted[:1].upper()
        return 1.0 if pred_letter == gold.upper() else 0.0
    elif q_type == "open_ended_math":
        # Clean both predicted and gold from currency symbols and spaces
        gold_num = re.sub(r"[$,\s]", "", gold)
        
        # 1. Try to find the "final answer is: [number]" anchor
        m = re.search(r"(?:final answer is|answer is)[:\s]*\$?([\-\d,\.]+)", predicted, re.IGNORECASE)
        if m:
            pred_num = re.sub(r"[$,\s]", "", m.group(1))
        else:
            # 2. Fall back to the last line's last number
            last_line = predicted.split("\n")[-1].strip()
            num_matches = re.findall(r"[\-\d,\.]+", last_line)
            if num_matches:
                pred_num = re.sub(r"[$,\s]", "", num_matches[-1])
            else:
                all_nums = re.findall(r"[\-\d,\.]+", predicted)
                pred_num = re.sub(r"[$,\s]", "", all_nums[-1]) if all_nums else predicted
                
        # Clean trailing periods or final punctuation if any
        if pred_num.endswith("."):
            pred_num = pred_num[:-1]

        # Exact match first
        if pred_num == gold_num:
            return 1.0
        # Float comparison (handles 3.0 == 3)
        try:
            return 1.0 if abs(float(pred_num) - float(gold_num)) < 0.01 else 0.0
        except ValueError:
            # Last resort: check if gold appears anywhere in prediction
            return 1.0 if gold_num in pred_num else 0.0
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(model_key: str, model_cfg: dict,
                  benchmark: str, questions: list[dict],
                  dry_run: bool = False) -> dict:
    """Run one model against one benchmark. Returns EvalCard dict."""
    label     = model_cfg["label"]
    n         = len(questions)
    out_path  = RESULTS_DIR / f"{label}_{benchmark}.json"

    if out_path.exists():
        print(f"  [SKIP] {label} / {benchmark} — already done, loading cache.")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n  {'[DRY RUN] ' if dry_run else ''}Running {label} on {benchmark} "
          f"({n} questions)...")

    if dry_run:
        return {"model": label, "benchmark": benchmark,
                "n_evaluated": 0, "accuracy": 0.0,
                "dry_run": True}

    client    = make_client(model_cfg)
    latencies = []
    n_correct = 0
    n_errors  = 0
    
    # GSM8K CoT needs headroom for everyone
    max_tokens = 512 if (model_cfg["reasoning"] or benchmark == "gsm8k") else 128
    
    system_content = None
    if benchmark == "gsm8k":
        system_content = ("Solve the math problem step-by-step. Show your work clearly. "
                          "At the very end of your response, write 'The final answer is: ' "
                          "followed by the numeric answer.")

    for qi, q in enumerate(questions):
        # Avoid hitting Groq's TPM limits by sleeping 2.5s (8.0s for the massive GPT-OSS)
        if model_cfg["client"] == "groq" and qi > 0:
            sleep_time = 8.0 if model_key == "gpt-oss-120b" else 2.5
            time.sleep(sleep_time)

        text, lat_ms = call_model(client, model_cfg, q["question"],
                                   system_content=system_content,
                                   max_tokens=max_tokens)
        if text.startswith("ERROR"):
            n_errors += 1
            if qi % 20 == 0:
                print(f"    q{qi+1}/{n} err", end="", flush=True)
            continue

        score = score_answer(text, q["answer"], q["type"])
        n_correct += score
        latencies.append(lat_ms)

        if qi % 20 == 0:
            running_acc = n_correct / max(1, qi + 1 - n_errors)
            print(f"\n    q{qi+1}/{n} | running_acc={running_acc:.3f}",
                  end="", flush=True)

    n_evaluated = n - n_errors
    accuracy    = n_correct / n_evaluated if n_evaluated > 0 else 0.0
    latencies.sort()
    p95_ms = latencies[int(0.95 * len(latencies))] if latencies else 0.0
    mean_ms = sum(latencies) / len(latencies) if latencies else 0.0

    card = {
        "model":           label,
        "benchmark":       benchmark,
        "n_total":         n,
        "n_evaluated":     n_evaluated,
        "n_errors":        n_errors,
        "accuracy":        round(accuracy, 6),
        "p95_latency_ms":  round(p95_ms, 3),
        "mean_latency_ms": round(mean_ms, 3),
        "total_cost_usd":  0.0,
        "seed":            SEED,
    }

    with open(out_path, "w") as f:
        json.dump(card, f, indent=2)

    print(f"\n    DONE: acc={accuracy:.3f}  p95={p95_ms:.0f}ms  "
          f"errors={n_errors}/{n}")
    return card


def p95_of(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    s = sorted(latencies)
    return s[int(0.95 * len(s))]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GSM8K / ARC-Challenge / HellaSwag on 4 free-tier models."
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Questions per benchmark (default: 200)")
    parser.add_argument("--benchmarks", nargs="+",
                        choices=list(BENCHMARK_LOADERS.keys()),
                        default=list(BENCHMARK_LOADERS.keys()),
                        help="Benchmarks to run (default: all 3)")
    parser.add_argument("--models", nargs="+",
                        choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Models to evaluate (default: all 4)")
    parser.add_argument("--model-sleep", type=int, default=0,
                        help="Seconds to sleep between model runs (default: 0)")
    parser.add_argument("--run", action="store_true",
                        help="Actually run inference (default: dry-run only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check keys + load datasets, no API calls")
    args = parser.parse_args()

    dry_run = not args.run or args.dry_run

    print("\n" + "="*68)
    print("  EvalForge Extended Benchmarks — Task 4")
    print("="*68)
    print(f"  Benchmarks : {', '.join(args.benchmarks)}")
    print(f"  Models     : {', '.join(args.models)}")
    print(f"  N per bench: {args.n}")
    print(f"  Mode       : {'DRY RUN' if dry_run else 'LIVE INFERENCE'}")
    print()

    # Check API keys
    missing_keys = []
    for mk in args.models:
        m = MODELS[mk]
        if not m["api_key"]:
            missing_keys.append(f"{m['label']} ({m['client'].upper()}_API_KEY)")
    if missing_keys:
        print("  [WARN] Missing API keys (set in .env):")
        for k in missing_keys:
            print(f"    - {k}")
        if not dry_run:
            print("\n  ERROR: Cannot run inference without API keys.")
            print("  Switch to Eagle WiFi, verify keys with test_keys_quick.py,")
            print("  then run with --run flag.")
            sys.exit(1)

    # Load all benchmarks
    all_questions = {}
    for bname in args.benchmarks:
        loader = BENCHMARK_LOADERS[bname]
        try:
            all_questions[bname] = loader(args.n)
        except Exception as e:
            print(f"  [ERROR] Failed to load {bname}: {e}")
            print("  Are you on Eagle WiFi / internet? HuggingFace required.")
            if not dry_run:
                sys.exit(1)
            all_questions[bname] = []

    # Run evaluations
    all_cards = []
    for i, mk in enumerate(args.models):
        if i > 0 and args.model_sleep > 0 and not dry_run:
            print(f"\n  [sleeping {args.model_sleep}s between models...]", flush=True)
            time.sleep(args.model_sleep)
        model_cfg = MODELS[mk]
        for bname in args.benchmarks:
            questions = all_questions.get(bname, [])
            if not questions:
                continue
            card = run_benchmark(mk, model_cfg, bname, questions,
                                 dry_run=dry_run)
            all_cards.append(card)

    if dry_run:
        print("\n  [DRY RUN COMPLETE] Switch to Eagle WiFi + use --run flag.")
        return

    # Summary
    print("\n" + "="*68)
    print("  RESULTS SUMMARY")
    print("="*68)
    print(f"  {'Model':<25} {'Benchmark':<15} {'Acc':>6} {'P95ms':>7} {'n':>5}")
    print("  " + "-"*60)

    summary = {}
    for card in all_cards:
        if card.get("dry_run"):
            continue
        label = card["model"]
        bench = card["benchmark"]
        acc   = card.get("accuracy", 0.0)
        p95   = card.get("p95_latency_ms", 0.0)
        n_ev  = card.get("n_evaluated", 0)
        print(f"  {label:<25} {bench:<15} {acc:>6.3f} {p95:>7.0f} {n_ev:>5}")
        summary.setdefault(label, {})[bench] = card

    # Save aggregate
    out_summary = HERE / "results" / "extended_summary.json"
    with open(out_summary, "w") as f:
        json.dump({"seed": SEED, "n_per_bench": args.n,
                   "results": all_cards}, f, indent=2)
    print(f"\n  [OK] Saved: {out_summary}")
    print(f"  [OK] Individual EvalCards in: {RESULTS_DIR}")
    print()
    print("  Next: Task 5 — full paper rewrite with all new data")
    print("="*68)


if __name__ == "__main__":
    main()
