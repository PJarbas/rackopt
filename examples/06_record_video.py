#!/usr/bin/env python3
"""Example: Recording simulation to video file.

This example demonstrates how to record a simulation as an MP4 video.
Requires: pip install rackopt[video]

The video will be saved to 'output/simulation.mp4'.
"""

from pathlib import Path

from rackopt import ClusterEnv, ClusterConfig
from rackopt.policies.heuristics import FirstFit
from rackopt.viz.pygame_rack import PygameRackRenderer


def main():
    """Run simulation and record to video."""
    print("=" * 60)
    print("RackOpt - Video Recording Example")
    print("=" * 60)

    # Create cluster configuration
    config = ClusterConfig(
        num_nodes=6,
        node_resources={"cpu": 16.0, "ram": 64.0},
        max_simulation_time=200.0,
    )

    # Create environment and policy
    env = ClusterEnv(config, seed=42)
    policy = FirstFit()

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    video_path = output_dir / "simulation.avi"  # AVI format for better codec compatibility

    print(f"\nCluster: {config.num_nodes} nodes")
    print(f"Resources per node: {config.node_resources}")
    print(f"Policy: {policy.__class__.__name__}")
    print(f"Video output: {video_path}")
    print("=" * 60)

    # Create renderer with video recording enabled
    try:
        renderer = PygameRackRenderer(
            env,
            width=1200,
            height=800,
            fps=30,
            record_video=True,
            video_path=video_path,
        )
    except Exception as e:
        print(f"\n❌ Failed to start recording: {e}")
        print("\nMake sure you have opencv-python installed:")
        print("  pip install opencv-python")
        return

    print("\n🎬 Recording simulation...")
    print("   (Close window or press ESC to stop early)\n")

    # Run simulation
    obs = env.reset()
    step = 0

    while not renderer.closed:
        # Render current state
        renderer.render_step()

        if renderer.closed:
            break

        # Select action and step
        action = policy.select_action(obs)
        obs, reward, done, info = env.step(action)

        if step % 20 == 0:
            print(f"  Step {step:4d} | Time: {obs.current_time:7.2f} | "
                  f"Pending: {len(obs.pending_tasks):3d}")

        step += 1

        if done:
            # Render a few more frames at the end
            for _ in range(30):
                renderer.render_step()
            break

    # Close renderer (this also stops recording)
    renderer.close()

    # Print final results
    print("\n" + "=" * 60)
    print("Recording Complete!")
    print("=" * 60)

    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Video saved: {video_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"\nPlay with: vlc {video_path}")
        print(f"       or: ffplay {video_path}")
    else:
        print("\n❌ Video file not created")


if __name__ == "__main__":
    main()
