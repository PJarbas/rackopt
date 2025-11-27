"""Example with Pygame visualization."""

from rackopt import ClusterEnv
from rackopt.config.config import ConfigProfiles
from rackopt.policies.heuristics import FirstFit

try:
    from rackopt.viz.pygame_rack import PygameRackRenderer
except ImportError:
    print("ERROR: Pygame not installed!")
    print("Install with: pip install rackopt[viz]")
    exit(1)


def main():
    """Run cluster simulation with visual rendering."""
    # Create environment with small cluster
    config = ConfigProfiles.small_cluster()
    config.policy_name = "FirstFit"  # Set for display
    env = ClusterEnv(config, seed=42)

    # Create renderer
    renderer = PygameRackRenderer(
        env,
        width=1200,
        height=800,
        title="RackOpt - Cluster Visualization Demo",
        max_slots_per_node=16,
        fps=30,
    )

    # Create policy
    policy = FirstFit()

    print("=" * 60)
    print("RackOpt - Pygame Visualization Example")
    print("=" * 60)
    print(f"Cluster: {config.num_nodes} nodes")
    print(f"Resources per node: {config.node_resources}")
    print(f"Policy: {policy.name}")
    print("=" * 60)
    print("\nVisualization controls:")
    print("  - ESC or close window to exit")
    print("  - Window will update automatically as simulation runs")
    print("=" * 60)
    print()

    # Reset environment
    obs = env.reset()
    done = False

    step = 0
    try:
        while not done:
            # Check if renderer was closed
            if renderer.closed:
                print("\nRenderer closed by user.")
                break

            # Policy selects action
            action = policy.select_action(obs)

            # Environment executes action
            obs, reward, done, info = env.step(action)

            # Update visualization
            renderer.render_step()

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step:3d} | "
                      f"Time: {info['time']:7.2f} | "
                      f"Pending: {info['num_pending']:3d} | "
                      f"Reward: {reward:7.2f}")

            step += 1

            # Limit steps for demo
            if step >= 200:
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")

    finally:
        # Close renderer
        renderer.close()

    print()
    print("=" * 60)
    print("Final Metrics:")
    print("=" * 60)
    env.metrics.print_summary()


if __name__ == "__main__":
    main()
