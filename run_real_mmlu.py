"""
run_real_mmlu.py
Runs the FULL real MMLU benchmark (cais/mmlu test split, 14k questions)
against all available models using the downloaded mmlu_test.jsonl.

BEFORE running:
  1. Download mmlu_test.jsonl off-VPN (see HOW_TO_RUN_REAL_MMLU.sh)
  2. Place it at: mmlu_cache/mmlu_test.jsonl
  3. Set API keys in .env
  4. Run: python run_real_mmlu.py [--n 500] [--models groq]

Expected real scores (0-shot, from published benchmarks):
  Llama-3.1-8B      ~0.65-0.68  (our embedded bank gave 0.924 -- FAKE)
  Llama-3.3-70B     ~0.84-0.87
  Llama-4-Scout-17B ~0.79-0.82
  Qwen3-32B         ~0.85-0.87
  Mistral-Small-3.1 ~0.72-0.75
  GPT-4o-mini       ~0.82-0.85
"""
import os, sys, json, time, re, argparse
from pathlib import Path

# ── Bootstrap: load .env + Walmart proxy + SSL trust (MUST be first) ─────────
def _bootstrap() -> None:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    try:
        import truststore
        truststore.inject_into_ssl()
        print("  [truststore] Windows cert store injected", flush=True)
    except ImportError:
        pass

_bootstrap()
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
MMLU_PATH = os.path.join(os.path.dirname(__file__), "mmlu_cache", "mmlu_test.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "real_mmlu")
os.makedirs(RESULTS_DIR, exist_ok=True)

CHOICE_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

MODELS = {
    "llama-3.1-8b": {
        "client_key": "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "model_id":   "llama-3.1-8b-instant",
        "label":      "Llama-3.1-8B",
        "reasoning":  False,
    },
    "llama-3.3-70b": {
        "client_key": "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "model_id":   "llama-3.3-70b-versatile",
        "label":      "Llama-3.3-70B",
        "reasoning":  False,
    },
    "llama-4-scout": {
        "client_key": "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "model_id":   "meta-llama/llama-4-scout-17b-16e-instruct",
        "label":      "Llama-4-Scout-17B",
        "reasoning":  True,
    },
    "qwen3-32b": {
        "client_key": "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "model_id":   "qwen/qwen3-32b",
        "label":      "Qwen3-32B",
        "reasoning":  True,
        "extra_body": {},
    },
    "mistral-small": {
        "client_key": "MISTRAL_API_KEY",
        "base_url":   "https://api.mistral.ai/v1",
        "model_id":   "mistral-small-2503",
        "label":      "Mistral-Small-3.1",
        "reasoning":  False,
    },
    "gpt-oss-120b": {
        "client_key": "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "model_id":   "openai/gpt-oss-120b",
        "label":      "GPT-OSS-120B",
        "reasoning":  True,
    },
    "gpt-oss-120b-cerebras": {
        "client_key": "CEREBRAS_API_KEY",
        "base_url":   "https://api.cerebras.ai/v1",
        "model_id":   "gpt-oss-120b",
        "label":      "GPT-OSS-120B",
        "reasoning":  True,
    },
    "zai-glm": {
        "client_key": "CEREBRAS_API_KEY",
        "base_url":   "https://api.cerebras.ai/v1",
        "model_id":   "zai-glm-4.7",
        "label":      "ZAI-GLM-4.7",
        "reasoning":  True,
    },
    "gpt-4o-mini": {
        "client_key": "GITHUB_TOKEN",
        "base_url":   "https://models.inference.ai.azure.com",
        "model_id":   "gpt-4o-mini",
        "label":      "GPT-4o-mini",
        "reasoning":  False,
    },
}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer multiple-choice questions by "
    "responding with ONLY the letter A, B, C, or D. No explanation."
)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_mmlu(path, n, seed):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("Download it off-VPN using HOW_TO_RUN_REAL_MMLU.sh first.")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    sample = [rows[i] for i in idxs]
    print(f"Loaded {len(sample)} questions from {len(rows)} total  (seed={seed})")
    return sample


# ── Inference ─────────────────────────────────────────────────────────────────
def _make_httpx_client() -> httpx.Client:
    """Build an httpx client that respects the Walmart corporate proxy.
    OpenAI SDK uses httpx internally; we pass our own client so the proxy
    env-var (HTTPS_PROXY / HTTP_PROXY) is always honoured.
    """
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )
    if proxy:
        return httpx.Client(proxy=proxy, timeout=120)
    return httpx.Client(timeout=120)


def make_client(cfg):
    key = os.environ.get(cfg["client_key"], "").strip()
    if not key:
        raise RuntimeError(f"Missing env var: {cfg['client_key']}")
    return OpenAI(api_key=key, base_url=cfg["base_url"],
                  http_client=_make_httpx_client())


def extract_answer(text: str, reasoning: bool = False) -> str:
    """Extract A/B/C/D from model output.
    For reasoning models: look at the LAST occurrence — answer comes after thinking.
    For standard models: look at the first occurrence.
    """
    text = (text or "").strip()
    # Strip <think>...</think> blocks if present
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text_clean:
        text_clean = text  # fallback if whole output was thinking

    if reasoning:
        # Take the last 300 chars — the answer letter comes at the very end
        tail = text_clean[-300:] if len(text_clean) > 300 else text_clean
        # Look for patterns like "answer is B", "The answer: C", or lone letter
        m = re.search(r"(?:answer(?:\s+is)?[:\s]+|therefore[,\s]+)([ABCD])\b",
                      tail, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Fallback: last standalone letter A/B/C/D in tail
        matches = re.findall(r"\b([ABCD])\b", tail.upper())
        return matches[-1] if matches else ""
    else:
        if text_clean and text_clean[0].upper() in "ABCD":
            return text_clean[0].upper()
        m = re.search(r"\b([ABCD])\b", text_clean.upper())
        return m.group(1) if m else ""


def eval_model(cfg, examples, max_tokens):
    client  = make_client(cfg)
    correct = 0
    latencies, costs = [], []
    errors = 0
    # Cerebras free tier: ~30 req/min cap -- pace to 2s between requests
    inter_req_delay = 2.0 if "cerebras" in cfg["base_url"] else 0.0

    for i, row in enumerate(examples):
        choices = "\n".join(
            f"{CHOICE_MAP[j]}. {c}" for j, c in enumerate(row["choices"])
        )
        prompt = (
            f"Question: {row['question']}\n{choices}\n\n"
            f"Answer with only the letter A, B, C, or D:"
        )
        ref = CHOICE_MAP[row["answer"]]

        _answered = False
        for attempt in range(3):
            try:
                t0  = time.perf_counter()  # reset per attempt -- excludes retry waits
                resp = client.chat.completions.create(
                    model=cfg["model_id"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                    extra_body=cfg.get("extra_body", {}),
                )
                ms   = (time.perf_counter() - t0) * 1000
                text = (resp.choices[0].message.content or "")
                # Reasoning models: check reasoning_content fallback
                if not text.strip():
                    rc = getattr(resp.choices[0].message, "reasoning_content", None)
                    if rc:
                        m = re.search(r"\b([ABCD])\b", rc.upper()[-200:])
                        text = m.group(1) if m else ""

                pred = extract_answer(text, reasoning=cfg.get("reasoning", False))
                hit  = int(pred == ref)
                correct  += hit
                latencies.append(ms)
                costs.append(0.0)
                if inter_req_delay > 0:
                    time.sleep(inter_req_delay)
                _answered = True
                break
            except Exception as e:
                errmsg = str(e)
                if "429" in errmsg:
                    wait = 10 * (2 ** attempt)
                    print(f"    [429 - waiting {wait}s]", flush=True)
                    time.sleep(wait)
                    # NOTE: if attempt == 2, loop ends without break -- handled below
                else:
                    print(f"    [ERR q{i}] {errmsg[:80]}", flush=True)
                    errors += 1
                    break
        # Bug fix: 429s that exhaust all 3 retries exit the loop silently.
        # Explicitly count them as errors so n_evaluated + n_errors == n_total.
        if not _answered:
            errors += 1
            print(f"    [DROPPED q{i}] all retries exhausted (429 or unhandled)",
                  flush=True)

        if (i + 1) % 50 == 0:
            acc_so_far = correct / len(latencies) if latencies else 0
            print(f"    {i+1}/{len(examples)}  acc={acc_so_far:.3f}", flush=True)

    n_v = len(latencies)
    return {
        "accuracy":        correct / n_v if n_v else 0.0,
        "p95_latency_ms":  float(np.percentile(latencies, 95)) if latencies else 0.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "n_evaluated":     n_v,
        "n_errors":        errors,
        "total_cost_usd":  0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int, default=500,
                    help="Questions per model (default 500, full=14042)")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--models", type=str, default="all",
                    help="Comma-separated model keys or 'all' or 'groq'")
    args = ap.parse_args()

    # Select models
    if args.models == "all":
        chosen = list(MODELS.keys())
    elif args.models == "groq":
        chosen = [k for k, v in MODELS.items() if "groq" in v["base_url"]]
    else:
        chosen = [m.strip() for m in args.models.split(",")]

    print("=" * 65)
    print(f"  REAL MMLU Benchmark  |  n={args.n}  seed={args.seed}")
    print(f"  Models: {chosen}")
    print("=" * 65)

    examples = load_mmlu(MMLU_PATH, args.n, args.seed)
    all_results = {}

    for key in chosen:
        if key not in MODELS:
            print(f"Unknown model key: {key}. Options: {list(MODELS.keys())}")
            continue
        cfg = MODELS[key]
        max_tokens = 4096 if cfg.get("reasoning") else 16

        print(f"\n>> {cfg['label']}  [{cfg['base_url'].split('/')[2]}]", flush=True)
        t_start = time.time()
        try:
            res = eval_model(cfg, examples, max_tokens)
        except RuntimeError as e:
            print(f"   SKIPPED: {e}")
            continue
        elapsed = time.time() - t_start

        all_results[key] = res
        res["model"]   = cfg["label"]
        res["n_total"] = args.n

        # Save EvalCard
        card_path = os.path.join(RESULTS_DIR, f"{cfg['label']}_real_mmlu.json")
        with open(card_path, "w") as f:
            json.dump(res, f, indent=2)

        print(f"   accuracy = {res['accuracy']:.4f}  ({res['accuracy']*100:.1f}%)")
        print(f"   P95      = {res['p95_latency_ms']:.0f} ms")
        print(f"   elapsed  = {elapsed:.0f}s")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  REAL MMLU RESULTS  (n={args.n}, seed={args.seed})")
    print("=" * 65)
    print(f"  {'Model':<25} {'Acc':>8} {'P95(ms)':>10} {'N':>6}")
    print("  " + "-" * 55)
    for key, res in sorted(all_results.items(),
                           key=lambda x: -x[1]["accuracy"]):
        cfg = MODELS[key]
        print(f"  {cfg['label']:<25} {res['accuracy']:>7.3f}  "
              f"{res['p95_latency_ms']:>9.0f}  {res['n_evaluated']:>5}")
    print("=" * 65)
    print("  Total cost: $0.00")
    print(f"  EvalCards saved to: {RESULTS_DIR}")
    print()
    print("  NOTE: Compare these numbers against published 0-shot MMLU scores.")
    print("  Llama-3.1-8B should land ~0.65-0.68, NOT 0.92.")
    print("  When you see that, the data is real and the paper is publishable.")


if __name__ == "__main__":
    main()
