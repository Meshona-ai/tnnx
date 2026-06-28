from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CompilePass = Literal["prune", "normalize", "shape_prop"]
LatencyPriority = Literal["balanced", "low_latency", "low_memory", "auditability"]

DEFAULT_PASSES: tuple[CompilePass, ...] = ("prune", "normalize", "shape_prop")


@dataclass(slots=True)
class ResourceBudget:
    """User-supplied target constraints recorded with each generated artifact.

    The current compiler does not auto-search optimization plans from these
    constraints. They make the target and adaptation intent explicit so
    experiments can compare pass configurations under the same declared budget.
    """

    target_hardware: str = "unspecified"
    preferred_dtype: str | None = None
    memory_budget_mb: int | None = None
    latency_priority: LatencyPriority = "balanced"
    notes: str = ""

    def to_metadata(self) -> dict[str, int | str | None]:
        return {
            "target_hardware": self.target_hardware,
            "preferred_dtype": self.preferred_dtype,
            "memory_budget_mb": self.memory_budget_mb,
            "latency_priority": self.latency_priority,
            "notes": self.notes,
        }


@dataclass(slots=True)
class CompileConfig:
    deterministic: bool = True
    infer_shapes: bool = True
    emit_shape_asserts: bool = True
    emit_graph_ir: bool = True
    opset: int = 18
    entrypoint: str = "forward"
    weights_filename: str = "weights.npz"
    enabled_passes: tuple[CompilePass, ...] = DEFAULT_PASSES
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)

    def to_metadata(self) -> dict[str, object]:
        return {
            "deterministic": self.deterministic,
            "infer_shapes": self.infer_shapes,
            "emit_shape_asserts": self.emit_shape_asserts,
            "emit_graph_ir": self.emit_graph_ir,
            "opset": self.opset,
            "entrypoint": self.entrypoint,
            "weights_filename": self.weights_filename,
            "enabled_passes": list(self.enabled_passes),
            "resource_budget": self.resource_budget.to_metadata(),
        }


@dataclass(slots=True)
class ArtifactManifest:
    target: str
    files: list[Path] = field(default_factory=list)
    weights_file: Path | None = None
    entrypoint: str = "forward"
