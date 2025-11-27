#!/usr/bin/env python3
"""Generate a demo GIF for the README."""

import os
from pathlib import Path

# Set SDL to use dummy video driver (no window)
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rackopt import ClusterEnv, ClusterConfig
from rackopt.policies.heuristics import FirstFit
from rackopt.viz.pygame_rack import PygameRackRenderer

import pygame
import imageio.v2 as imageio
import numpy as np


def main():
    print("Generating demo GIF...")
    
    # Create output directory
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # Create environment
    config = ClusterConfig(
        num_nodes=6,
        node_resources={"cpu": 16.0, "ram": 64.0},
        max_simulation_time=150.0,
    )
    config.policy_name = "FirstFit"
    
    env = ClusterEnv(config, seed=42)
    policy = FirstFit()
    
    # Create renderer
    renderer = PygameRackRenderer(
        env,
        width=900,
        height=550,
        fps=30,
    )
    
    # Collect frames
    frames = []
    obs = env.reset()
    step = 0
    
    print("Recording frames...")
    while not env.done and step < 300:
        # Render
        renderer.render_step()
        
        if renderer.closed:
            break
        
        # Capture frame every 2 steps (to reduce GIF size)
        if step % 2 == 0:
            # Get pygame surface as array
            frame = pygame.surfarray.array3d(renderer.screen)
            frame = np.transpose(frame, (1, 0, 2))  # Fix orientation
            frames.append(frame)
        
        # Step environment
        action = policy.select_action(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        
        if step % 50 == 0:
            print(f"  Step {step}, Time: {obs.current_time:.1f}")
    
    renderer.close()
    
    # Save GIF
    gif_path = output_dir / "demo.gif"
    print(f"Saving GIF with {len(frames)} frames...")
    imageio.mimsave(str(gif_path), frames, fps=15, loop=0)
    
    # Get file size
    size_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {gif_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
