"""Type-safe observation class for cluster state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rackopt.core.models import Task, Node, TaskState


@dataclass
class NodeState:
    """Snapshot of a node's state.

    Attributes:
        node_id: Node identifier
        name: Node name
        capacity: Total resource capacity
        usage: Current resource usage
        utilization: Utilization ratio per resource (0.0 to 1.0)
        num_tasks: Number of tasks running on this node
        task_ids: List of task IDs running on this node
    """

    node_id: int
    name: str | None
    capacity: dict[str, float]
    usage: dict[str, float]
    utilization: dict[str, float]
    num_tasks: int
    task_ids: list[int]

    @classmethod
    def from_node(cls, node: Node) -> NodeState:
        """Create NodeState from a Node object.

        Args:
            node: Node to snapshot

        Returns:
            NodeState representation
        """
        utilization = {
            resource: node.get_utilization(resource) for resource in node.resources
        }

        return cls(
            node_id=node.node_id,
            name=node.name,
            capacity=node.resources.copy(),
            usage=node.usage,
            utilization=utilization,
            num_tasks=len(node.tasks),
            task_ids=node.tasks.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.node_id,
            "name": self.name,
            "capacity": self.capacity,
            "usage": self.usage,
            "utilization": self.utilization,
            "num_tasks": self.num_tasks,
            "task_ids": self.task_ids,
        }


@dataclass
class TaskInfo:
    """Snapshot of a task's state.

    Attributes:
        task_id: Task identifier
        state: Current task state
        demands: Resource demands
        duration: Total duration
        arrival_time: Arrival time
        progress: Execution progress (0.0 to 1.0)
        deadline: Optional deadline
        preemptible: Whether task can be preempted
        priority: Task priority
        task_type: Task type/category
        allocated_node_id: Node where task is allocated (-1 if not allocated)
    """

    task_id: int
    state: TaskState
    demands: dict[str, float]
    duration: float
    arrival_time: float
    progress: float
    deadline: float | None
    preemptible: bool
    priority: int
    task_type: str
    allocated_node_id: int

    @classmethod
    def from_task(cls, task: Task) -> TaskInfo:
        """Create TaskInfo from a Task object.

        Args:
            task: Task to snapshot

        Returns:
            TaskInfo representation
        """
        return cls(
            task_id=task.task_id,
            state=task.state,
            demands=task.demands.copy(),
            duration=task.duration,
            arrival_time=task.arrival_time,
            progress=task.progress,
            deadline=task.deadline,
            preemptible=task.preemptible,
            priority=task.priority,
            task_type=task.task_type,
            allocated_node_id=task.allocated_node_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "demands": self.demands,
            "duration": self.duration,
            "arrival_time": self.arrival_time,
            "progress": self.progress,
            "deadline": self.deadline,
            "preemptible": self.preemptible,
            "priority": self.priority,
            "task_type": self.task_type,
            "allocated_node_id": self.allocated_node_id,
        }


@dataclass
class Observation:
    """Type-safe observation of cluster state.

    This class provides a snapshot of the cluster state at a specific point in time.
    It can be converted to dict or numpy arrays for compatibility with different
    frameworks.

    Attributes:
        nodes: State of all nodes in the cluster
        pending_tasks: Tasks waiting to be allocated
        running_tasks: Tasks currently running
        current_time: Current simulation time
        metrics: Additional metrics (e.g., tasks_completed, rejection_rate)
    """

    nodes: list[NodeState]
    pending_tasks: list[TaskInfo]
    running_tasks: list[TaskInfo]
    current_time: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert observation to dictionary format.

        Returns:
            Dictionary representation compatible with JSON serialization
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "pending_tasks": [task.to_dict() for task in self.pending_tasks],
            "running_tasks": [task.to_dict() for task in self.running_tasks],
            "current_time": self.current_time,
            "metrics": self.metrics.copy(),
        }

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert observation to numpy arrays for ML frameworks.

        Returns:
            Dictionary of numpy arrays with standardized shapes
        """
        # Node features: [num_nodes, num_resources * 2]
        # (capacity and usage for each resource)
        node_features = []
        resource_names = sorted(self.nodes[0].capacity.keys()) if self.nodes else []

        for node in self.nodes:
            features = []
            for resource in resource_names:
                features.append(node.capacity.get(resource, 0.0))
                features.append(node.usage.get(resource, 0.0))
            node_features.append(features)

        nodes_array = np.array(node_features, dtype=np.float32) if node_features else np.array(
            [], dtype=np.float32
        )

        # Task features: [num_tasks, feature_dim]
        def task_to_features(task: TaskInfo) -> list[float]:
            features = [
                float(task.task_id),
                float(task.state.value == "waiting"),
                float(task.state.value == "running"),
                task.duration,
                task.arrival_time,
                task.progress,
                task.deadline if task.deadline is not None else -1.0,
                float(task.preemptible),
                float(task.priority),
            ]
            # Add resource demands in sorted order
            for resource in resource_names:
                features.append(task.demands.get(resource, 0.0))
            return features

        pending_array = np.array(
            [task_to_features(task) for task in self.pending_tasks], dtype=np.float32
        ) if self.pending_tasks else np.array([], dtype=np.float32)

        running_array = np.array(
            [task_to_features(task) for task in self.running_tasks], dtype=np.float32
        ) if self.running_tasks else np.array([], dtype=np.float32)

        return {
            "nodes": nodes_array,
            "pending_tasks": pending_array,
            "running_tasks": running_array,
            "current_time": np.array([self.current_time], dtype=np.float32),
            "resource_names": np.array(resource_names),
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to observation attributes.

        Args:
            key: Attribute name

        Returns:
            Attribute value
        """
        if key == "nodes":
            return self.nodes
        elif key == "pending_tasks":
            return self.pending_tasks
        elif key == "running_tasks":
            return self.running_tasks
        elif key == "current_time":
            return self.current_time
        elif key == "metrics":
            return self.metrics
        else:
            raise KeyError(f"Unknown observation key: {key}")

    def get_num_nodes(self) -> int:
        """Get number of nodes in observation."""
        return len(self.nodes)

    def get_num_pending_tasks(self) -> int:
        """Get number of pending tasks."""
        return len(self.pending_tasks)

    def get_num_running_tasks(self) -> int:
        """Get number of running tasks."""
        return len(self.running_tasks)

    def get_total_capacity(self) -> dict[str, float]:
        """Get total resource capacity across all nodes."""
        total: dict[str, float] = {}
        for node in self.nodes:
            for resource, capacity in node.capacity.items():
                total[resource] = total.get(resource, 0.0) + capacity
        return total

    def get_total_usage(self) -> dict[str, float]:
        """Get total resource usage across all nodes."""
        total: dict[str, float] = {}
        for node in self.nodes:
            for resource, usage in node.usage.items():
                total[resource] = total.get(resource, 0.0) + usage
        return total

    def get_average_utilization(self) -> float:
        """Get average utilization across all nodes."""
        if not self.nodes:
            return 0.0
        total_util = sum(
            sum(node.utilization.values()) / len(node.utilization)
            if node.utilization else 0.0
            for node in self.nodes
        )
        return total_util / len(self.nodes)
