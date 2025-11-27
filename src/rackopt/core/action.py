"""Action representation with soft validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rackopt.core.observation import Observation

# Constant for rejecting a task
REJECT = -1


@dataclass
class AllocationDecision:
    """Single allocation decision.

    Attributes:
        task_id: ID of task to allocate
        node_id: ID of target node (use REJECT=-1 to reject the task)
        preempt_task_id: Optional ID of task to preempt on target node
    """

    task_id: int
    node_id: int
    preempt_task_id: int | None = None

    def is_rejection(self) -> bool:
        """Check if this decision represents a rejection.

        Returns:
            True if node_id is REJECT
        """
        return self.node_id == REJECT

    def is_preemption(self) -> bool:
        """Check if this decision includes preemption.

        Returns:
            True if preempt_task_id is set
        """
        return self.preempt_task_id is not None


@dataclass
class Action:
    """Represents scheduling decisions for one or more tasks.

    The Action class supports batch allocation decisions with soft validation.
    Invalid actions result in penalties rather than exceptions, making it
    compatible with reinforcement learning frameworks.

    Attributes:
        decisions: List of allocation decisions to execute
    """

    decisions: list[AllocationDecision]

    def __init__(self, decisions: list[AllocationDecision] | None = None):
        """Initialize action.

        Args:
            decisions: List of allocation decisions (default: empty list)
        """
        self.decisions = decisions if decisions is not None else []

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        """Create Action from dictionary.

        Args:
            data: Dictionary with 'decisions' key containing list of dicts

        Returns:
            Action object
        """
        decisions = [
            AllocationDecision(
                task_id=d["task_id"],
                node_id=d["node_id"],
                preempt_task_id=d.get("preempt_task_id"),
            )
            for d in data.get("decisions", [])
        ]
        return cls(decisions=decisions)

    @classmethod
    def from_list(cls, allocations: list[tuple[int, int]]) -> Action:
        """Create Action from list of (task_id, node_id) tuples.

        Args:
            allocations: List of (task_id, node_id) pairs

        Returns:
            Action object
        """
        decisions = [
            AllocationDecision(task_id=task_id, node_id=node_id)
            for task_id, node_id in allocations
        ]
        return cls(decisions=decisions)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "decisions": [
                {
                    "task_id": d.task_id,
                    "node_id": d.node_id,
                    "preempt_task_id": d.preempt_task_id,
                }
                for d in self.decisions
            ]
        }

    def validate(self, observation: Observation) -> tuple[bool, float]:
        """Validate action against current observation (soft validation).

        This method checks if the action is valid and returns a penalty score
        for invalid actions. This approach is compatible with RL frameworks
        that learn from negative rewards.

        Args:
            observation: Current cluster state

        Returns:
            Tuple of (is_valid, penalty):
                - is_valid: True if action is completely valid
                - penalty: Penalty score for invalid parts (0.0 if valid)
        """
        if not self.decisions:
            return True, 0.0

        penalty = 0.0
        is_valid = True

        # Build lookup structures
        node_states = {node.node_id: node for node in observation.nodes}
        pending_task_ids = {task.task_id for task in observation.pending_tasks}
        running_task_ids = {task.task_id for task in observation.running_tasks}
        pending_tasks = {task.task_id: task for task in observation.pending_tasks}

        # Track resources within this action (for batch validation)
        node_available = {
            node_id: node.capacity.copy()
            for node_id, node in node_states.items()
        }

        # Deduct current usage
        for node_id, node in node_states.items():
            for resource, usage in node.usage.items():
                node_available[node_id][resource] -= usage

        for decision in self.decisions:
            # Check if task exists in pending queue
            if decision.task_id not in pending_task_ids:
                penalty += 5.0  # High penalty for invalid task
                is_valid = False
                continue

            # Rejection is always valid
            if decision.is_rejection():
                continue

            # Check if node exists
            if decision.node_id not in node_states:
                penalty += 10.0  # High penalty for non-existent node
                is_valid = False
                continue

            task = pending_tasks[decision.task_id]

            # Check preemption validity
            if decision.is_preemption():
                if decision.preempt_task_id not in running_task_ids:
                    penalty += 5.0  # Invalid preemption target
                    is_valid = False
                    continue

                # Check if preemption target is on the target node
                preempt_task = next(
                    (t for t in observation.running_tasks if t.task_id == decision.preempt_task_id),
                    None
                )
                if preempt_task and preempt_task.allocated_node_id != decision.node_id:
                    penalty += 5.0  # Preemption target not on target node
                    is_valid = False
                    continue

            # Check resource availability
            available = node_available[decision.node_id]
            can_allocate = True

            for resource, demand in task.demands.items():
                if resource not in available:
                    penalty += 3.0  # Unknown resource type
                    is_valid = False
                    can_allocate = False
                    break

                if demand > available[resource]:
                    penalty += 2.0  # Over-allocation penalty
                    is_valid = False
                    can_allocate = False
                    break

            # Update available resources for next decision in batch
            if can_allocate:
                for resource, demand in task.demands.items():
                    node_available[decision.node_id][resource] -= demand

        return is_valid, penalty

    def add_decision(
        self,
        task_id: int,
        node_id: int,
        preempt_task_id: int | None = None
    ) -> None:
        """Add an allocation decision to this action.

        Args:
            task_id: ID of task to allocate
            node_id: ID of target node (or REJECT)
            preempt_task_id: Optional task to preempt
        """
        self.decisions.append(
            AllocationDecision(
                task_id=task_id,
                node_id=node_id,
                preempt_task_id=preempt_task_id
            )
        )

    def get_rejections(self) -> list[int]:
        """Get list of rejected task IDs.

        Returns:
            List of task IDs that are rejected
        """
        return [d.task_id for d in self.decisions if d.is_rejection()]

    def get_allocations(self) -> list[tuple[int, int]]:
        """Get list of (task_id, node_id) allocations (excluding rejections).

        Returns:
            List of allocation pairs
        """
        return [
            (d.task_id, d.node_id)
            for d in self.decisions
            if not d.is_rejection()
        ]

    def get_preemptions(self) -> list[tuple[int, int]]:
        """Get list of (task_id, preempt_task_id) preemption pairs.

        Returns:
            List of preemption pairs
        """
        return [
            (d.task_id, d.preempt_task_id)
            for d in self.decisions
            if d.is_preemption()
        ]

    def is_empty(self) -> bool:
        """Check if action has no decisions.

        Returns:
            True if no decisions
        """
        return len(self.decisions) == 0

    def __len__(self) -> int:
        """Get number of decisions in action."""
        return len(self.decisions)

    def __repr__(self) -> str:
        """String representation of action."""
        if not self.decisions:
            return "Action([])"

        parts = []
        for d in self.decisions:
            if d.is_rejection():
                parts.append(f"reject({d.task_id})")
            elif d.is_preemption():
                parts.append(f"allocate({d.task_id}→{d.node_id}, preempt={d.preempt_task_id})")
            else:
                parts.append(f"allocate({d.task_id}→{d.node_id})")

        return f"Action([{', '.join(parts)}])"
