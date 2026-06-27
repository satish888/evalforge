"""
Tests for EvalForge core module.
"""
import pytest
from evalforge.core import (
    Example, Prediction, Task, TaskRegistry, task_registry
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class DummyTask(Task):
    dimension = "reasoning"
    display_name = "Dummy Reasoning Task"
    primary_metric = "accuracy"

    def load_examples(self, split="test"):
        return [
            Example(id="1", input="What is 2+2?", reference="4"),
            Example(id="2", input="What is 3+3?", reference="6"),
        ]

    def build_prompt(self, example, few_shot=None):
        return f"Q: {example.input}\nA:"

    def score(self, prediction, example):
        correct = prediction.text.strip() == str(example.reference)
        return {"accuracy": float(correct)}


# ─────────────────────────────────────────────────────────────────────────────
# Task tests
# ─────────────────────────────────────────────────────────────────────────────

def test_task_load_examples():
    task = DummyTask()
    examples = task.load_examples()
    assert len(examples) == 2
    assert all(isinstance(e, Example) for e in examples)


def test_task_build_prompt():
    task = DummyTask()
    example = Example(id="1", input="Test?", reference="Yes")
    prompt = task.build_prompt(example)
    assert "Test?" in prompt


def test_task_score_correct():
    task = DummyTask()
    example = Example(id="1", input="2+2?", reference="4")
    pred = Prediction(text="4")
    scores = task.score(pred, example)
    assert scores["accuracy"] == 1.0


def test_task_score_wrong():
    task = DummyTask()
    example = Example(id="1", input="2+2?", reference="4")
    pred = Prediction(text="5")
    scores = task.score(pred, example)
    assert scores["accuracy"] == 0.0


def test_task_postprocess_strips_whitespace():
    task = DummyTask()
    assert task.postprocess("  hello  ") == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_register_and_build():
    registry = TaskRegistry()

    @registry.register("test/dummy")
    class _T(DummyTask):
        pass

    assert "test/dummy" in registry
    task = registry.build("test/dummy")
    assert isinstance(task, _T)


def test_registry_raises_on_duplicate():
    registry = TaskRegistry()

    @registry.register("test/dup")
    class _A(DummyTask):
        pass

    with pytest.raises(KeyError, match="already registered"):
        @registry.register("test/dup")
        class _B(DummyTask):
            pass


def test_registry_override():
    registry = TaskRegistry()

    @registry.register("test/over")
    class _A(DummyTask):
        display_name = "A"

    @registry.register("test/over", override=True)
    class _B(DummyTask):
        display_name = "B"

    assert registry.build("test/over").display_name == "B"


def test_registry_unknown_task_raises():
    registry = TaskRegistry()
    with pytest.raises(KeyError, match="Unknown task"):
        registry.build("nonexistent/task")


def test_registry_list_by_dimension():
    registry = TaskRegistry()

    @registry.register("dim/reason")
    class _R(DummyTask):
        dimension = "reasoning"

    @registry.register("dim/safety")
    class _S(DummyTask):
        dimension = "safety"

    reason_tasks = registry.list_tasks(dimension="reasoning")
    assert "dim/reason" in reason_tasks
    assert "dim/safety" not in reason_tasks


def test_registry_len():
    registry = TaskRegistry()
    assert len(registry) == 0

    @registry.register("len/t1")
    class _T(DummyTask):
        pass

    assert len(registry) == 1


def test_registry_non_task_subclass_raises():
    registry = TaskRegistry()
    with pytest.raises(TypeError):
        @registry.register("bad/type")
        class _NotATask:
            pass
