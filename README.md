# EvalForge

**A modular, open-source framework for holistic LLM evaluation with the Enterprise Readiness Index (ERI)**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)

---

## What Is EvalForge?

EvalForge is an open-source evaluation framework that answers the question practitioners face in production: **"Which LLM should I deploy?"** — not just "which scores highest on a benchmark?"

It measures operational dimensions simultaneously to construct the **Enterprise Readiness Index (ERI)** — a weighted harmonic mean grounded in multi-criteria decision theory. Because the harmonic mean is non-compensatory, it penalizes weakness: one bad dimension (such as catastrophic latency or safety failures) cannot be masked by strong accuracy elsewhere.

---

## Repository Structure

```text
evalforge/
├── README.md                       # Documentation & reproduction instructions
├── LICENSE                         # Apache 2.0 License
├── .gitignore                      # Safe file filters (excludes secrets, local caches)
├── pyproject.toml                  # PEP 517 build metadata & dependencies
│
├── run_real_mmlu.py                # MMLU benchmark runner (with 429-exhaustion fix)
├── run_extended_benchmarks.py      # GSM8K benchmark runner (with CoT & per-model throttle)
├── compute_harmonic_eri.py         # reads results, prints the ERI tables
│
├── evalforge/                      # Core framework package
│   ├── __init__.py                 # Version 0.1.0
│   ├── contamination.py            # Bloom-filter n-gram detector
│   ├── core.py                     # Evaluation harness
│   ├── eri.py                      # ERI metrics (harmonic, arithmetic, subsets)
│   └── reporter.py                 # EvalCard JSON generator
│
├── tests/                          # Automated testing suite
│   ├── test_contamination.py
│   ├── test_core.py
│   └── test_eri.py
│
└── results/                        # Locked reproducibility artifacts
    ├── mmlu/                       # 6 MMLU EvalCards (5 valid + Qwen3 excluded)
    │   ├── GPT-OSS-120B_real_mmlu.json
    │   ├── Llama-3.3-70B_real_mmlu.json
    │   ├── Llama-4-Scout-17B_real_mmlu.json
    │   ├── Mistral-Small-3.1_real_mmlu.json
    │   ├── Llama-3.1-8B_real_mmlu.json
    │   └── Qwen3-32B_real_mmlu.json
    ├── gsm8k/                      # 4 GSM8K EvalCards (GPT-OSS excluded)
    │   ├── Llama-4-Scout-17B_gsm8k.json
    │   ├── Llama-3.3-70B_gsm8k.json
    │   ├── Mistral-Small-3.1_gsm8k.json
    │   └── Llama-3.1-8B_gsm8k.json
    ├── real_harmonic_eri.json      # ERI reference calculations
    └── figures/                    # 8 PNG figures at 300 DPI (corrected fig1/2/4/7)
```

---

## Installation

To clone and install EvalForge in editable mode:

```bash
git clone https://github.com/satish888/evalforge
cd evalforge
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## Reproducing the Paper Results

To guarantee complete scientific transparency, we provide the exact scripts, seeds, and configurations used to produce the figures and tables in the paper.

### 1. Run Benchmark 1: MMLU (Zero-Shot Direct)
This runs the MMLU evaluation (500 test questions, seed 42) across our models. Ensure your API keys are configured in a local `.env` file first:
```bash
python run_real_mmlu.py --n 500 --seed 42
```
*Note: Qwen3-32B is excluded due to API stability limits, and its raw responses are available for inspection in the local results directory.*

### 2. Run Benchmark 2: GSM8K (Zero-Shot Chain-of-Thought)
This evaluates the models on grade-school math reasoning using Chain-of-Thought (CoT) prompts. To prevent API rate-limiting artifacts, we implement spacing between requests:
```bash
python run_extended_benchmarks.py --benchmarks gsm8k --n 200 --run
```
*Note: GPT-OSS-120B is excluded due to free-tier rate-limiting constraints on massive reasoning token streams.*

### 3. Generate ERI Tables
To process the generated EvalCards, compute the ERI rankings, and verify the boundary conditions and rank inversions:
```bash
python compute_harmonic_eri.py
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
