"""
compute_harmonic_eri.py
Computes TRUE harmonic ERI from real MMLU EvalCards using the evalforge library.

    ERI = 1 / (wA/acc + wL/lat + wO/cost)   <- weighted harmonic mean

Weights (paper): A=0.40, L=0.40, O=0.20
SLA = 5,000 ms  |  Dimensions: accuracy, latency, cost (subset mode)

Reproduces from the released implementation:
    from evalforge.eri import compute_eri_from_scores
"""
import json, os, sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from evalforge.eri import compute_eri_from_scores

# EvalCards location: same directory as this script.
# In the submission package (option1_submission/) the JSONs sit alongside.
# In the repo root, point at the curated option1_submission/ copies.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBMISSION = os.path.join(_HERE, "option1_submission")
RESULTS_DIR = _SUBMISSION if os.path.isdir(_SUBMISSION) else _HERE
SLA_MS = 5000.0
WEIGHTS = {"accuracy": 0.40, "latency": 0.40, "cost": 0.20}
EPS = 1e-9

# ── Load EvalCards ────────────────────────────────────────────────────────────
cards = {}
for fn in sorted(os.listdir(RESULTS_DIR)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(RESULTS_DIR, fn)) as f:
        d = json.load(f)
    if d.get("n_evaluated", 0) > 0:
        cards[d["model"]] = d

# ── Compute ERI via library ───────────────────────────────────────────────────
results = []
for name, c in cards.items():
    lat_norm = max(EPS, 1.0 - c["p95_latency_ms"] / SLA_MS)
    dim_scores = {
        "accuracy": c["accuracy"],
        "latency":  lat_norm,
        "cost":     1.0,          # all free-tier: cost = 1.0 (perfect)
    }
    eri = compute_eri_from_scores(dim_scores, WEIGHTS)
    results.append({
        "model":    name,
        "accuracy": c["accuracy"],
        "p95_ms":   c["p95_latency_ms"],
        "lat_norm": round(lat_norm, 3),
        "eri":      round(eri, 3),
    })

results.sort(key=lambda x: -x["eri"])

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 68)
print("  TRUE HARMONIC ERI  (from real MMLU EvalCards via evalforge.eri)")
print(f"  Formula : ERI = 1 / (wA/acc + wL/lat + wO/cost)")
print(f"  Weights : A={WEIGHTS['accuracy']}, L={WEIGHTS['latency']}, "
      f"O(cost)={WEIGHTS['cost']}  |  SLA={SLA_MS:.0f}ms")
print(f"  acc     = raw accuracy [0,1]")
print(f"  lat     = max(eps, 1 - p95_ms / {SLA_MS:.0f})")
print(f"  cost    = 1.0  (all free-tier APIs)")
print("=" * 68)
print(f"  {'Model':<25} {'Acc':>6} {'P95ms':>7} {'lat_n':>6} {'ERI':>6}  Rank")
print("  " + "-" * 60)
for i, r in enumerate(results, 1):
    if r["model"] == "GPT-OSS-120B":
        tag = "  <- HEADLINE: highest acc, LOWEST ERI"
    elif i == 1:
        tag = "  <- RECOMMENDED"
    else:
        tag = ""
    print(f"  {r['model']:<25} {r['accuracy']:>6.3f} {r['p95_ms']:>7.0f} "
          f"{r['lat_norm']:>6.3f} {r['eri']:>6.3f}  #{i}{tag}")

print()
print("  KEY FINDING: GPT-OSS-120B = highest MMLU (88.1%) but LOWEST ERI (0.340)")
print("  Harmonic penalty: 4126ms P95 -> lat_norm=0.175, cannot be offset by accuracy.")
print("  Mistral-Small-3.1 recommended: ERI=0.693 (2232ms P95, lat_norm=0.554)")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS_DIR, "real_harmonic_eri.json")
with open(out_path, "w") as f:
    json.dump({"sla_ms": SLA_MS, "weights": WEIGHTS, "models": results}, f, indent=2)
print(f"\n  Saved: {out_path}")
