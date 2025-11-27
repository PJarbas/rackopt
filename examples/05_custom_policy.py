"""Example of implementing a custom scheduling policy."""

from rackopt import ClusterEnv
from rackopt.config.config import ConfigProfiles
from rackopt.core.action import Action, REJECT
from rackopt.core.observation import Observation
from rackopt.policies.base import BasePolicy


class LoadBalancingPolicy(BasePolicy):
    """Custom policy that balances load across nodes.

    Allocates tasks to the node with lowest current utilization.
    """

    def __init__(self):
        """Initialize load balancing policy."""
        super().__init__(name="LoadBalancing")

    def select_action(self, observation: Observation) -> Action:
        """Select node with lowest utilization for each task.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        action = Action()

        for task in observation.pending_tasks:
            best_node = None
            min_utilization = float("inf")

            # Find node with minimum utilization that can fit task
            for node in observation.nodes:
                if self._can_allocate(node, task):
                    # Calculate average utilization
                    avg_util = sum(node.utilization.values()) / len(node.utilization)

                    if avg_util < min_utilization:
                        min_utilization = avg_util
                        best_node = node

            if best_node is not None:
                action.add_decision(task.task_id, best_node.node_id)
            else:
                action.add_decision(task.task_id, REJECT)

        return action

    def _can_allocate(self, node, task) -> bool:
        """Check if task can be allocated to node."""
        for resource, demand in task.demands.items():
            available = node.capacity.get(resource, 0.0) - node.usage.get(resource, 0.0)
            if demand > available:
                return False
        return True


class DeadlineAwarePolicy(BasePolicy):
    """Custom policy that prioritizes tasks with tight deadlines.

    Allocates tasks with approaching deadlines first.
    """

    def __init__(self):
        """Initialize deadline-aware policy."""
        super().__init__(name="DeadlineAware")

    def select_action(self, observation: Observation) -> Action:
        """Prioritize tasks by deadline urgency.

        Args:
            observation: Current cluster state

        Returns:
            Action with allocations
        """
        action = Action()
        current_time = observation.current_time

        # Sort tasks by deadline (soonest first), tasks without deadlines last
        sorted_tasks = sorted(
            observation.pending_tasks,
            key=lambda t: t.deadline if t.deadline is not None else float("inf")
        )

        for task in sorted_tasks:
            # Try to allocate using first-fit
            allocated = False
            for node in observation.nodes:
                if self._can_allocate(node, task):
                    action.add_decision(task.task_id, node.node_id)
                    allocated = True
                    break

            if not allocated:
                action.add_decision(task.task_id, REJECT)

        return action

    def _can_allocate(self, node, task) -> bool:
        """Check if task can be allocated to node."""
        for resource, demand in task.demands.items():
            available = node.capacity.get(resource, 0.0) - node.usage.get(resource, 0.0)
            if demand > available:
                return False
        return True


def compare_policies():
    """Compare custom policies against baseline heuristics."""
    from rackopt.policies.heuristics import FirstFit, BestFit

    print("=" * 80)
    print("RackOpt - Custom Policy Comparison")
    print("=" * 80)
    print()

    # Create configuration with deadlines
    config = ConfigProfiles.interactive_workload()  # Has many deadlines
    config.workload.num_tasks = 150

    # Policies to compare
    policies = [
        FirstFit(),
        BestFit(),
        LoadBalancingPolicy(),
        DeadlineAwarePolicy(),
    ]

    print(f"Configuration: Interactive workload with {config.workload.num_tasks} tasks")
    print(f"Nodes: {config.num_nodes}")
    print(f"Deadline probability: {config.workload.deadline_probability:.0%}")
    print()

    print(f"{'Policy':<20} {'Completed':>10} {'Rejected':>10} {'SLA Viol.':>12} {'Avg Response':>15}")
    print("-" * 80)

    for policy in policies:
        config.policy_name = policy.name

        # Run simulation
        env = ClusterEnv(config, seed=42)
        obs = env.reset()
        done = False

        while not done:
            action = policy.select_action(obs)
            obs, reward, done, info = env.step(action)

        # Get metrics
        metrics = env.metrics.get_current_metrics()

        print(f"{policy.name:<20} "
              f"{metrics['tasks_completed']:>10} "
              f"{metrics['tasks_rejected']:>10} "
              f"{metrics['sla_violations']:>12} "
              f"{metrics['avg_response_time']:>15.2f}")

    print()
    print("=" * 80)


def main():
    """Main function."""
    compare_policies()


if __name__ == "__main__":
    main()
