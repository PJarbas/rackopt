"""Core data models for cluster simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TaskState(Enum):
    """Task execution state."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    REJECTED = "rejected"
    PREEMPTED = "preempted"


@dataclass
class Task:
    """Represents a task/job to be scheduled.

    Attributes:
        task_id: Unique identifier for the task
        demands: Resource demands (e.g., {'cpu': 4.0, 'ram': 8.0, 'gpu': 1.0})
        duration: Task execution duration
        arrival_time: Time when task arrived in the system
        state: Current task state
        deadline: Optional deadline for task completion
        progress: Execution progress (0.0 to 1.0)
        preemptible: Whether the task can be preempted
        priority: Task priority (higher = more important)
        task_type: Task classification/category
        allocated_node_id: ID of node where task is allocated (-1 if not allocated)
        start_time: Time when task started execution (None if not started)
    """

    task_id: int
    demands: dict[str, float]
    duration: float
    arrival_time: float
    state: TaskState = TaskState.WAITING
    deadline: float | None = None
    progress: float = 0.0
    preemptible: bool = False
    priority: int = 0
    task_type: str = "default"
    allocated_node_id: int = -1
    start_time: float | None = None

    def get_deadline_status(self, current_time: float, warning_threshold: float = 0.2) -> str:
        """Calculate deadline status based on remaining time.

        Args:
            current_time: Current simulation time
            warning_threshold: Threshold ratio for warning status (default: 0.2 = 20%)

        Returns:
            "ok", "warning", "late", or "" if no deadline
        """
        if self.deadline is None:
            return ""

        if current_time >= self.deadline:
            return "late"

        total_time = self.deadline - self.arrival_time
        remaining_time = self.deadline - current_time

        if total_time <= 0:
            return "late"

        remaining_ratio = remaining_time / total_time

        if remaining_ratio < warning_threshold:
            return "warning"

        return "ok"

    def get_remaining_duration(self) -> float:
        """Get remaining execution duration."""
        return self.duration * (1.0 - self.progress)

    def update_progress(self, elapsed_time: float) -> None:
        """Update task progress based on elapsed time.

        Args:
            elapsed_time: Time elapsed since last update
        """
        if self.duration > 0:
            self.progress = min(1.0, self.progress + elapsed_time / self.duration)


@dataclass
class Node:
    """Represents a compute node in the cluster.

    Attributes:
        node_id: Unique identifier for the node
        resources: Total available resources (e.g., {'cpu': 32.0, 'ram': 128.0})
        name: Optional human-readable name
        tasks: List of task IDs currently running on this node
        _usage: Current resource usage (tracked internally)
    """

    node_id: int
    resources: dict[str, float]
    name: str | None = None
    tasks: list[int] = field(default_factory=list)
    _usage: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize usage tracking."""
        self._usage = {resource: 0.0 for resource in self.resources}

    @property
    def usage(self) -> dict[str, float]:
        """Get current resource usage."""
        return self._usage.copy()

    def get_available_resources(self) -> dict[str, float]:
        """Get available resources (capacity - usage)."""
        return {
            resource: self.resources[resource] - self._usage.get(resource, 0.0)
            for resource in self.resources
        }

    def can_allocate(self, demands: dict[str, float]) -> bool:
        """Check if node can allocate resources for given demands.

        Args:
            demands: Resource demands to check

        Returns:
            True if all demanded resources are available
        """
        available = self.get_available_resources()

        for resource, demand in demands.items():
            if resource not in available:
                return False
            if demand > available[resource]:
                return False

        return True

    def allocate(self, task_id: int, demands: dict[str, float]) -> bool:
        """Allocate resources for a task.

        Args:
            task_id: ID of task to allocate
            demands: Resource demands

        Returns:
            True if allocation successful, False otherwise
        """
        if not self.can_allocate(demands):
            return False

        # Update usage
        for resource, demand in demands.items():
            self._usage[resource] = self._usage.get(resource, 0.0) + demand

        # Add task to node
        if task_id not in self.tasks:
            self.tasks.append(task_id)

        return True

    def deallocate(self, task_id: int, demands: dict[str, float]) -> None:
        """Deallocate resources from a task.

        Args:
            task_id: ID of task to deallocate
            demands: Resource demands to release
        """
        # Release resources
        for resource, demand in demands.items():
            self._usage[resource] = max(0.0, self._usage.get(resource, 0.0) - demand)

        # Remove task from node
        if task_id in self.tasks:
            self.tasks.remove(task_id)

    def get_utilization(self, resource: str | None = None) -> float:
        """Get resource utilization ratio (0.0 to 1.0).

        Args:
            resource: Specific resource to check, or None for average across all

        Returns:
            Utilization ratio
        """
        if resource is not None:
            if resource not in self.resources or self.resources[resource] == 0:
                return 0.0
            return self._usage.get(resource, 0.0) / self.resources[resource]

        # Average utilization across all resources
        if not self.resources:
            return 0.0

        total_util = sum(
            self._usage.get(res, 0.0) / cap if cap > 0 else 0.0
            for res, cap in self.resources.items()
        )
        return total_util / len(self.resources)


@dataclass
class Cluster:
    """Represents a cluster of compute nodes.

    Attributes:
        nodes: List of nodes in the cluster
        tasks: Dictionary mapping task_id to Task objects
    """

    nodes: list[Node]
    tasks: dict[int, Task] = field(default_factory=dict)

    def get_node(self, node_id: int) -> Node | None:
        """Get node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Node object or None if not found
        """
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_task(self, task_id: int) -> Task | None:
        """Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        return self.tasks.get(task_id)

    def add_task(self, task: Task) -> None:
        """Add a task to the cluster.

        Args:
            task: Task to add
        """
        self.tasks[task.task_id] = task

    def allocate_task(self, task_id: int, node_id: int) -> bool:
        """Allocate a task to a node.

        Args:
            task_id: ID of task to allocate
            node_id: ID of target node

        Returns:
            True if allocation successful, False otherwise
        """
        task = self.get_task(task_id)
        node = self.get_node(node_id)

        if task is None or node is None:
            return False

        if task.state != TaskState.WAITING and task.state != TaskState.PREEMPTED:
            return False

        if node.allocate(task_id, task.demands):
            task.state = TaskState.RUNNING
            task.allocated_node_id = node_id
            return True

        return False

    def deallocate_task(self, task_id: int) -> bool:
        """Deallocate a task from its node.

        Args:
            task_id: ID of task to deallocate

        Returns:
            True if deallocation successful, False otherwise
        """
        task = self.get_task(task_id)
        if task is None or task.allocated_node_id < 0:
            return False

        node = self.get_node(task.allocated_node_id)
        if node is None:
            return False

        node.deallocate(task_id, task.demands)
        task.allocated_node_id = -1

        return True

    def get_total_resources(self) -> dict[str, float]:
        """Get total resources across all nodes."""
        total: dict[str, float] = {}
        for node in self.nodes:
            for resource, capacity in node.resources.items():
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
        """Get average utilization across all nodes and resources."""
        if not self.nodes:
            return 0.0

        total_util = sum(node.get_utilization() for node in self.nodes)
        return total_util / len(self.nodes)
