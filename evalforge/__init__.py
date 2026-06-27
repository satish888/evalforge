"""EvalForge — Public package API."""
from .contamination import ContaminationConfig, ContaminationDetector, ContaminationReport
from .core import Example, Prediction, Task, TaskRegistry, TaskResult, task_registry
from .eri import ERIConfig, ERIScore, WEIGHT_PROFILES, compute_eri
from .reporter import EvalCard, EnvironmentInfo, ModelIdentity, TaskConfig

__version__ = "0.1.0"
__all__ = [
    # Core
    "Example", "Prediction", "Task", "TaskRegistry", "TaskResult",
    "task_registry",
    # ERI
    "ERIConfig", "ERIScore", "WEIGHT_PROFILES", "compute_eri",
    # Contamination
    "ContaminationConfig", "ContaminationDetector", "ContaminationReport",
    # Reporter
    "EvalCard", "EnvironmentInfo", "ModelIdentity", "TaskConfig",
]
