"""Common heuristic scheduling policies."""

from __future__ import annotations

import random
from typing import Callable

from rackopt.core.action import Action, AllocationDecision, REJECT
from rackopt.core.observation import Observation, NodeState, TaskInfo
from rackopt.policies.base import BasePolicy


class FirstFit(BasePolicy):
    """First-Fit scheduling: allocate to first node with sufficient resources."""

    def __init__(self):
        """Initialize FirstFit policy."""
        super().__init__(name="FirstFit")

    def select_action(self, observation: Observation) -> Action:
        """Select first available node for each pending task.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        action = Action()

        for task in observation.pending_tasks:
            allocated = False

            # Try each node in order
            for node in observation.nodes:
                if self._can_allocate(node, task):
                    action.add_decision(task.task_id, node.node_id)
                    allocated = True
                    break

            # Reject if no node can accommodate
            if not allocated:
                action.add_decision(task.task_id, REJECT)

        return action

    def _can_allocate(self, node: NodeState, task: TaskInfo) -> bool:
        """Check if task can be allocated to node.

        Args:
            node: Node to check
            task: Task to allocate

        Returns:
            True if allocation is possible
        """
        for resource, demand in task.demands.items():
            available = node.capacity.get(resource, 0.0) - node.usage.get(resource, 0.0)
            if demand > available:
                return False
        return True


class BestFit(BasePolicy):
    """Best-Fit scheduling: allocate to node with least remaining resources."""

    def __init__(self):
        """Initialize BestFit policy."""
        super().__init__(name="BestFit")

    def select_action(self, observation: Observation) -> Action:
        """Select node with minimum remaining capacity for each task.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        action = Action()

        for task in observation.pending_tasks:
            best_node = None
            min_remaining = float("inf")

            # Find node with minimum remaining capacity
            for node in observation.nodes:
                if self._can_allocate(node, task):
                    remaining = self._get_remaining_capacity(node, task)
                    if remaining < min_remaining:
                        min_remaining = remaining
                        best_node = node

            if best_node is not None:
                action.add_decision(task.task_id, best_node.node_id)
            else:
                action.add_decision(task.task_id, REJECT)

        return action

    def _can_allocate(self, node: NodeState, task: TaskInfo) -> bool:
        """Check if task can be allocated to node."""
        for resource, demand in task.demands.items():
            available = node.capacity.get(resource, 0.0) - node.usage.get(resource, 0.0)
            if demand > available:
                return False
        return True

    def _get_remaining_capacity(self, node: NodeState, task: TaskInfo) -> float:
        """Get sum of remaining capacity after allocation.

        Args:
            node: Node to check
            task: Task to allocate

        Returns:
            Total remaining capacity across all resources
        """
        total_remaining = 0.0
        for resource in node.capacity.keys():
            available = node.capacity[resource] - node.usage.get(resource, 0.0)
            demand = task.demands.get(resource, 0.0)
            remaining = available - demand
            total_remaining += remaining
        return total_remaining


class WorstFit(BasePolicy):
    """Worst-Fit scheduling: allocate to node with most remaining resources."""

    def __init__(self):
        """Initialize WorstFit policy."""
        super().__init__(name="WorstFit")

    def select_action(self, observation: Observation) -> Action:
        """Select node with maximum remaining capacity for each task.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        action = Action()

        for task in observation.pending_tasks:
            best_node = None
            max_remaining = -1.0

            # Find node with maximum remaining capacity
            for node in observation.nodes:
                if self._can_allocate(node, task):
                    remaining = self._get_remaining_capacity(node, task)
                    if remaining > max_remaining:
                        max_remaining = remaining
                        best_node = node

            if best_node is not None:
                action.add_decision(task.task_id, best_node.node_id)
            else:
                action.add_decision(task.task_id, REJECT)

        return action

    def _can_allocate(self, node: NodeState, task: TaskInfo) -> bool:
        """Check if task can be allocated to node."""
        for resource, demand in task.demands.items():
            available = node.capacity.get(resource, 0.0) - node.usage.get(resource, 0.0)
            if demand > available:
                return False
        return True

    def _get_remaining_capacity(self, node: NodeState, task: TaskInfo) -> float:
        """Get sum of remaining capacity after allocation."""
        total_remaining = 0.0
        for resource in node.capacity.keys():
            available = node.capacity[resource] - node.usage.get(resource, 0.0)
            demand = task.demands.get(resource, 0.0)
            remaining = available - demand
            total_remaining += remaining
        return total_remaining


class RandomPolicy(BasePolicy):
    """Random scheduling: allocate to random available node."""

    def __init__(self, seed: int | None = None):
        """Initialize RandomPolicy.

        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(name="Random")
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def select_action(self, observation: Observation) -> Action:
        """Select random available node for each task.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        action = Action()

        for task in observation.pending_tasks:
            # Get all nodes that can accommodate task
            available_nodes = [
                node for node in observation.nodes
                if self._can_allocate(node, task)
            ]

            if available_nodes:
                # Pick random node
                chosen_node = random.choice(available_nodes)
                action.add_decision(task.task_id, chosen_node.node_id)
            else:
                action.add_decision(task.task_id, REJECT)

        return action

    def _can_allocate(self, node: NodeState, task: TaskInfo) -> bool:
        """Check if task can be allocated to node."""
        for resource, demand in task.demands.items():
            available = node.capacity.get(resource, 0.0) - node.usage.get(resource, 0.0)
            if demand > available:
                return False
        return True

    def reset(self) -> None:
        """Reset random seed."""
        if self.seed is not None:
            random.seed(self.seed)


class RoundRobin(BasePolicy):
    """Round-Robin scheduling: rotate through nodes."""

    def __init__(self):
        """Initialize RoundRobin policy."""
        super().__init__(name="RoundRobin")
        self.next_node_index = 0

    def select_action(self, observation: Observation) -> Action:
        """Allocate tasks in round-robin fashion across nodes.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        action = Action()
        num_nodes = len(observation.nodes)

        if num_nodes == 0:
            # No nodes available - reject all
            for task in observation.pending_tasks:
                action.add_decision(task.task_id, REJECT)
            return action

        for task in observation.pending_tasks:
            allocated = False

            # Try nodes starting from next_node_index
            for offset in range(num_nodes):
                node_idx = (self.next_node_index + offset) % num_nodes
                node = observation.nodes[node_idx]

                if self._can_allocate(node, task):
                    action.add_decision(task.task_id, node.node_id)
                    self.next_node_index = (node_idx + 1) % num_nodes
                    allocated = True
                    break

            if not allocated:
                action.add_decision(task.task_id, REJECT)

        return action

    def _can_allocate(self, node: NodeState, task: TaskInfo) -> bool:
        """Check if task can be allocated to node."""
        for resource, demand in task.demands.items():
            available = node.capacity.get(resource, 0.0) - node.usage.get(resource, 0.0)
            if demand > available:
                return False
        return True

    def reset(self) -> None:
        """Reset round-robin counter."""
        self.next_node_index = 0


class PriorityPolicy(BasePolicy):
    """Priority-based scheduling: allocate high-priority tasks first."""

    def __init__(self, base_policy: BasePolicy | None = None):
        """Initialize PriorityPolicy.

        Args:
            base_policy: Underlying policy for allocation (default: FirstFit)
        """
        super().__init__(name="Priority")
        self.base_policy = base_policy or FirstFit()

    def select_action(self, observation: Observation) -> Action:
        """Allocate tasks in priority order.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(
            observation.pending_tasks,
            key=lambda t: t.priority,
            reverse=True
        )

        # Create modified observation with sorted tasks
        sorted_observation = Observation(
            nodes=observation.nodes,
            pending_tasks=sorted_tasks,
            running_tasks=observation.running_tasks,
            current_time=observation.current_time,
            metrics=observation.metrics,
        )

        # Use base policy for actual allocation
        return self.base_policy.select_action(sorted_observation)

    def reset(self) -> None:
        """Reset base policy."""
        self.base_policy.reset()
