"""
EvalForge — EvalCard Reporter
==============================
Produces machine-readable YAML EvalCards and Markdown summaries.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # PyYAML

from .core import TaskResult
from .eri import ERIScore

EVALCARD_VERSION = "1.2"


# ─────────────────────────────────────────────────────────────────────────────
# EvalCard data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelIdentity:
    model_name: str
    model_id: str
    checkpoint_sha256: str = "api"
    adapter: str = "openai"


@dataclass
class EnvironmentInfo:
    evalforge_version: str
    python_version: str = field(
        default_factory=lambda: sys.version.split()[0]
    )
    os_info: str = field(
        default_factory=lambda: f"{platform.system()} {platform.release()}"
    )
    cuda_version: str | None = None
    gpu_model: str | None = None
    gpu_count: int | None = None
    random_seed: int = 42
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class TaskConfig:
    task_name: str
    split: str = "test"
    few_shot_count: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.0
    example_count: int = 0
    contamination_flagged: bool = False
    contamination_score: float | None = None


@dataclass
class EvalCard:
    """
    Structured, machine-readable evaluation report.

    Serialises to YAML for reproducibility and archival.
    """
    identity: ModelIdentity
    environment: EnvironmentInfo
    task_configs: list[TaskConfig]
    results: list[TaskResult]
    eri: ERIScore | None = None

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "evalcard_version": EVALCARD_VERSION,
            "identity": asdict(self.identity),
            "environment": asdict(self.environment),
            "task_configs": [asdict(c) for c in self.task_configs],
            "results": [
                {
                    "task_name": r.task_name,
                    "scores": r.scores,
                    "ci": {k: list(v) for k, v in r.ci.items()},
                    "mean_latency_ms": r.mean_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "throughput_tokens_per_s": r.throughput_tokens_per_s,
                    "cost_usd_per_1k": r.cost_usd_per_1k,
                    "contamination_flagged": r.contamination_flagged,
                    "contamination_score": r.contamination_score,
                }
                for r in self.results
            ],
            "eri": (
                {
                    "eri_score": self.eri.eri,
                    "ci_lower_95": self.eri.ci_lower,
                    "ci_upper_95": self.eri.ci_upper,
                    "weight_profile": self.eri.weight_profile,
                    "weights": self.eri.weights,
                    "dimension_scores": self.eri.dimension_scores,
                }
                if self.eri
                else None
            ),
        }

    def save_yaml(self, path: Path | str) -> None:
        """Write EvalCard to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def save_json(self, path: Path | str) -> None:
        """Write EvalCard to a JSON file (for API consumers)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # ── Markdown summary ───────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Render a human-readable Markdown summary."""
        lines: list[str] = []
        m = self.identity
        e = self.environment

        lines += [
            f"# EvalCard — {m.model_name}",
            "",
            f"**EvalForge version:** `{e.evalforge_version}`  ",
            f"**Timestamp (UTC):** `{e.timestamp_utc}`  ",
            f"**Model ID:** `{m.model_id}`  ",
            f"**Checkpoint SHA-256:** `{m.checkpoint_sha256}`  ",
            f"**Adapter:** `{m.adapter}`  ",
            f"**Python:** `{e.python_version}`  ",
            f"**OS:** `{e.os_info}`  ",
            "",
        ]

        if self.eri:
            lines += [
                "## Enterprise Readiness Index (ERI)",
                "",
                f"| Metric | Score |",
                f"|--------|-------|",
                f"| **ERI ({self.eri.weight_profile})** "
                f"| **{self.eri.eri:.3f}** "
                f"({self.eri.ci_lower:.3f}–{self.eri.ci_upper:.3f}) |",
            ]
            for dim, score in self.eri.dimension_scores.items():
                w = self.eri.weights[dim]
                lines.append(f"| {dim.capitalize()} (w={w:.2f}) | {score:.3f} |")
            lines.append("")

        lines += [
            "## Task Results",
            "",
            "| Task | Score | Latency P95 (ms) | Cost/1k (USD) | Contaminated |",
            "|------|-------|-----------------|---------------|--------------|",
        ]
        for r in self.results:
            primary = next(iter(r.scores))
            score = r.scores[primary]
            ci = r.ci.get(primary, (0.0, 0.0))
            flag = "⚠️ Yes" if r.contamination_flagged else "No"
            lines.append(
                f"| {r.task_name} | {score:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) "
                f"| {r.p95_latency_ms:.0f} | {r.cost_usd_per_1k:.4f} | {flag} |"
            )

        return "\n".join(lines)

    def save_markdown(self, path: Path | str) -> None:
        """Write Markdown summary to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
