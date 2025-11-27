"""Simple hello world example using FirstFit policy."""

from rackopt import ClusterEnv
from rackopt.config.config import ConfigProfiles
from rackopt.policies.heuristics import FirstFit


def main():
    """Run a simple cluster simulation."""
    # Create environment with small cluster configuration
    config = ConfigProfiles.small_cluster()
    env = ClusterEnv(config, seed=42)

    # Create policy
    policy = FirstFit()

    # Reset environment
    obs = env.reset()
    done = False

    print("=" * 60)
    print("RackOpt - Hello World Example")
    print("=" * 60)
    print(f"Cluster: {config.num_nodes} nodes")
    print(f"Resources per node: {config.node_resources}")
    print(f"Policy: {policy.name}")
    print(f"Max simulation time: {config.max_simulation_time}")
    print("=" * 60)
    print()

    step = 0
    while not done and step < 100:  # Limit steps for demo
        # Policy selects action
        action = policy.select_action(obs)

        # Environment executes action
        obs, reward, done, info = env.step(action)

        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d} | "
                  f"Time: {info['time']:7.2f} | "
                  f"Pending: {info['num_pending']:3d} | "
                  f"Reward: {reward:7.2f}")

        step += 1

    print()
    print("=" * 60)
    print("Final Metrics:")
    print("=" * 60)
    env.metrics.print_summary()


if __name__ == "__main__":
    main()
