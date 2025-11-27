"""Example using different workload profiles."""

from rackopt import ClusterEnv
from rackopt.config.config import ConfigProfiles
from rackopt.policies.heuristics import FirstFit, BestFit, WorstFit


def run_simulation(config_name: str, config, policy):
    """Run a single simulation and return metrics."""
    env = ClusterEnv(config, seed=42)
    policy_instance = policy()

    obs = env.reset()
    done = False

    while not done:
        action = policy_instance.select_action(obs)
        obs, reward, done, info = env.step(action)

    return env.metrics.get_current_metrics()


def main():
    """Compare policies across different workload profiles."""
    print("=" * 80)
    print("RackOpt - Workload Profile Comparison")
    print("=" * 80)
    print()

    # Define configurations to test
    configs = [
        ("Small Cluster", ConfigProfiles.small_cluster()),
        ("Batch Workload", ConfigProfiles.batch_workload()),
        ("Interactive Workload", ConfigProfiles.interactive_workload()),
    ]

    # Define policies to test
    policies = [
        ("FirstFit", FirstFit),
        ("BestFit", BestFit),
        ("WorstFit", WorstFit),
    ]

    # Run experiments
    for config_name, config in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config_name}")
        print(f"{'=' * 80}")
        print(f"Nodes: {config.num_nodes}")
        print(f"Arrival rate: {config.workload.arrival_rate}")
        print(f"Tasks: {config.workload.num_tasks}")
        print()

        print(f"{'Policy':<15} {'Completed':>10} {'Rejected':>10} {'Avg Response':>15} {'Rejection %':>15}")
        print("-" * 80)

        for policy_name, policy_class in policies:
            config.policy_name = policy_name
            metrics = run_simulation(config_name, config, policy_class)

            print(f"{policy_name:<15} "
                  f"{metrics['tasks_completed']:>10} "
                  f"{metrics['tasks_rejected']:>10} "
                  f"{metrics['avg_response_time']:>15.2f} "
                  f"{metrics['rejection_rate'] * 100:>14.1f}%")

    print()
    print("=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
