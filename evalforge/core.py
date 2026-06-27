"""
EvalForge — Holistic LLM Evaluation Framework
==============================================
Core abstractions: Task, Example, Prediction, and the task registry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Example:
    """A single evaluation example."""
    id: str
    input: str | dict[str, Any]
    reference: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Model output for one Example."""
    text: str
    raw_logprobs: list[float] | None = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class TaskResult:
    """Aggregated scores for one Task over all its examples."""
    task_name: str
    scores: dict[str, float]          # metric_name -> mean score
    ci: dict[str, tuple[float, float]] # metric_name -> (lower, upper)
    example_scores: list[dict[str, float]]
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_tokens_per_s: float
    cost_usd_per_1k: float
    contamination_score: float | None = None
    contamination_flagged: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Task base class
# ─────────────────────────────────────────────────────────────────────────────

class Task(ABC):
    """
    Abstract base class for all EvalForge tasks.

    Subclasses must implement:
      - load_examples()
      - build_prompt()
      - score()

    The `dimension` and `display_name` class attributes must be set.
    """

    #: Capability dimension: one of reasoning|knowledge|coding|math|
    #:   long_context|safety|rag
    dimension: str = "general"

    #: Human-readable name for reports and EvalCards.
    display_name: str = ""

    #: Primary metric name returned by score(); used in leaderboard.
    primary_metric: str = "accuracy"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.display_name:
            cls.display_name = cls.__name__

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def load_examples(self, split: str = "test") -> list[Example]:
        """Return evaluation examples for the given split."""

    @abstractmethod
    def build_prompt(
        self,
        example: Example,
        few_shot: list[Example] | None = None,
    ) -> str:
        """Convert an example (and optional few-shot context) to a prompt."""

    @abstractmethod
    def score(
        self,
        prediction: Prediction,
        example: Example,
    ) -> dict[str, float]:
        """
        Return metric_name -> score (all scores in [0.0, 1.0]).
        Must include self.primary_metric as a key.
        """

    # ── Optional hooks ────────────────────────────────────────────────────

    def postprocess(self, text: str) -> str:
        """Optional text postprocessing before scoring."""
        return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────────────────────────────────────

class TaskRegistry:
    """
    Central registry mapping task names to Task classes.

    Usage::

        registry = TaskRegistry()

        @registry.register("myorg/my_task")
        class MyTask(Task):
            ...

        task = registry.build("myorg/my_task")
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Task]] = {}

    def register(
        self,
        name: str,
        *,
        override: bool = False,
    ):
        """Decorator that registers a Task class under `name`."""
        def decorator(cls: type[Task]) -> type[Task]:
            if name in self._registry and not override:
                raise KeyError(
                    f"Task '{name}' is already registered. "
                    "Use override=True to replace it."
                )
            if not issubclass(cls, Task):
                raise TypeError(f"{cls} must be a subclass of Task.")
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs: Any) -> Task:
        """Instantiate a registered task by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(
                f"Unknown task '{name}'. Available tasks: {available}"
            )
        return self._registry[name](**kwargs)

    def list_tasks(
        self,
        dimension: str | None = None,
    ) -> list[str]:
        """Return registered task names, optionally filtered by dimension."""
        names = sorted(self._registry)
        if dimension is None:
            return names
        return [
            n for n in names
            if self._registry[n].dimension == dimension
        ]

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# ── Module-level singleton ────────────────────────────────────────────────────
task_registry = TaskRegistry()
