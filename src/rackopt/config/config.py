"""Configuration management with Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class RewardConfig(BaseModel):
    """Reward function configuration.

    Attributes:
        w_completed: Weight for completed tasks
        w_rejected: Weight (penalty) for rejected tasks
        w_sla_violations: Weight (penalty) for SLA violations
        w_action_penalty: Weight for invalid action penalties
    """

    w_completed: float = Field(default=1.0, ge=0.0)
    w_rejected: float = Field(default=10.0, ge=0.0)
    w_sla_violations: float = Field(default=5.0, ge=0.0)
    w_action_penalty: float = Field(default=1.0, ge=0.0)


class WorkloadConfig(BaseModel):
    """Workload generation configuration.

    Attributes:
        type: Type of workload generation ('synthetic' or 'trace')
        arrival_distribution: Distribution for task arrivals (e.g., 'poisson')
        arrival_rate: Mean arrival rate (tasks per time unit)
        duration_distribution: Distribution for task duration (e.g., 'exponential', 'uniform')
        duration_params: Parameters for duration distribution
        resource_demands: Resource demand specifications
        num_tasks: Number of tasks to generate (for synthetic workloads)
        trace_path: Path to trace file (for trace-based workloads)
        deadline_probability: Probability that a task has a deadline
        deadline_slack_factor: Factor for deadline slack (deadline = arrival + duration * factor)
    """

    type: Literal["synthetic", "trace"] = "synthetic"
    arrival_distribution: str = "poisson"
    arrival_rate: float = Field(default=1.0, gt=0.0)
    duration_distribution: str = "exponential"
    duration_params: dict[str, float] = Field(default_factory=lambda: {"mean": 10.0})
    resource_demands: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "cpu": {"min": 1.0, "max": 8.0},
            "ram": {"min": 2.0, "max": 16.0},
        }
    )
    num_tasks: int = Field(default=100, ge=1)
    trace_path: str | None = None
    deadline_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    deadline_slack_factor: float = Field(default=2.0, gt=1.0)
    preemptible_probability: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("trace_path")
    @classmethod
    def validate_trace_path(cls, v: str | None, info: Any) -> str | None:
        """Validate trace path exists if type is 'trace'."""
        if info.data.get("type") == "trace" and v is None:
            raise ValueError("trace_path is required when type='trace'")
        if v is not None and not Path(v).exists():
            raise ValueError(f"trace_path does not exist: {v}")
        return v


class ClusterConfig(BaseModel):
    """Complete cluster simulation configuration.

    Attributes:
        num_nodes: Number of nodes in cluster (1-50)
        node_resources: Resources per node (e.g., {'cpu': 32, 'ram': 128})
        heterogeneous: If True, add variation to node resources
        heterogeneity_factor: Factor for resource variation (0.0-1.0)
        workload: Workload generation configuration
        reward: Reward function configuration
        max_simulation_time: Maximum simulation time
        deadline_warning_threshold: Threshold for deadline warning (0.0-1.0)
        seed: Random seed for reproducibility
        policy_name: Name of scheduling policy (for tracking)
    """

    num_nodes: int = Field(default=4, ge=1, le=50)
    node_resources: dict[str, float] = Field(
        default_factory=lambda: {"cpu": 32.0, "ram": 128.0}
    )
    heterogeneous: bool = False
    heterogeneity_factor: float = Field(default=0.2, ge=0.0, le=1.0)

    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    max_simulation_time: float = Field(default=1000.0, gt=0.0)
    deadline_warning_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int | None = None
    policy_name: str = "unknown"

    @field_validator("node_resources")
    @classmethod
    def validate_node_resources(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate node resources are positive."""
        for resource, capacity in v.items():
            if capacity <= 0:
                raise ValueError(f"Resource {resource} must be positive, got {capacity}")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClusterConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            ClusterConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> ClusterConfig:
        """Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            ClusterConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            ClusterConfig instance
        """
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return self.model_dump()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output path
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def copy(self, **kwargs: Any) -> ClusterConfig:
        """Create a copy with optional overrides.

        Args:
            **kwargs: Fields to override

        Returns:
            New ClusterConfig instance
        """
        data = self.to_dict()
        data.update(kwargs)
        return ClusterConfig(**data)


# Predefined configuration profiles
class ConfigProfiles:
    """Collection of predefined configuration profiles."""

    @staticmethod
    def small_cluster() -> ClusterConfig:
        """Small cluster for testing (4 nodes)."""
        return ClusterConfig(
            num_nodes=4,
            node_resources={"cpu": 16.0, "ram": 64.0},
            workload=WorkloadConfig(
                arrival_rate=0.5,
                num_tasks=50,
            ),
            max_simulation_time=500.0,
        )

    @staticmethod
    def medium_cluster() -> ClusterConfig:
        """Medium cluster (16 nodes)."""
        return ClusterConfig(
            num_nodes=16,
            node_resources={"cpu": 32.0, "ram": 128.0},
            workload=WorkloadConfig(
                arrival_rate=2.0,
                num_tasks=200,
            ),
            max_simulation_time=1000.0,
        )

    @staticmethod
    def large_cluster() -> ClusterConfig:
        """Large cluster (50 nodes - maximum)."""
        return ClusterConfig(
            num_nodes=50,
            node_resources={"cpu": 64.0, "ram": 256.0},
            workload=WorkloadConfig(
                arrival_rate=5.0,
                num_tasks=500,
            ),
            max_simulation_time=2000.0,
        )

    @staticmethod
    def heterogeneous_cluster() -> ClusterConfig:
        """Heterogeneous cluster with varied resources."""
        return ClusterConfig(
            num_nodes=20,
            node_resources={"cpu": 32.0, "ram": 128.0, "gpu": 4.0},
            heterogeneous=True,
            heterogeneity_factor=0.3,
            workload=WorkloadConfig(
                arrival_rate=1.5,
                num_tasks=300,
                resource_demands={
                    "cpu": {"min": 2.0, "max": 16.0},
                    "ram": {"min": 4.0, "max": 64.0},
                    "gpu": {"min": 0.0, "max": 2.0},
                },
            ),
            max_simulation_time=1500.0,
        )

    @staticmethod
    def batch_workload() -> ClusterConfig:
        """Configuration for batch processing workload."""
        return ClusterConfig(
            num_nodes=10,
            node_resources={"cpu": 32.0, "ram": 128.0},
            workload=WorkloadConfig(
                arrival_distribution="poisson",
                arrival_rate=0.8,
                duration_distribution="uniform",
                duration_params={"min": 50.0, "max": 150.0},
                num_tasks=100,
                deadline_probability=0.1,  # Few deadlines for batch
            ),
            max_simulation_time=2000.0,
        )

    @staticmethod
    def interactive_workload() -> ClusterConfig:
        """Configuration for interactive workload."""
        return ClusterConfig(
            num_nodes=16,
            node_resources={"cpu": 16.0, "ram": 64.0},
            workload=WorkloadConfig(
                arrival_distribution="poisson",
                arrival_rate=3.0,
                duration_distribution="exponential",
                duration_params={"mean": 10.0},
                num_tasks=300,
                deadline_probability=0.7,  # Many deadlines for interactive
                deadline_slack_factor=1.5,
            ),
            max_simulation_time=500.0,
        )
