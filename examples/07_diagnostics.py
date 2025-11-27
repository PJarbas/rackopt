#!/usr/bin/env python3
"""Example: Environment diagnostics and learning validation.

This example shows how to:
1. Validate that the environment is working correctly
2. Compare different policies
3. Track learning progress (for RL agents)
"""

from rackopt import ClusterEnv, ClusterConfig
from rackopt.policies.heuristics import FirstFit, BestFit, WorstFit, RandomPolicy
from rackopt.utils.diagnostics import (
    EnvironmentDiagnostics,
    LearningTracker,
    run_baseline_comparison,
)


def main():
    # =========================================================================
    # Part 1: Environment Diagnostics
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Environment Diagnostics")
    print("=" * 70)
    
    # Define factories for creating env and policy
    def env_factory(seed=42):
        config = ClusterConfig(
            num_nodes=4,
            node_resources={"cpu": 16.0, "ram": 64.0},
            max_simulation_time=200.0,
        )
        return ClusterEnv(config, seed=seed)
    
    def policy_factory():
        return FirstFit()
    
    # Run diagnostics
    diagnostics = EnvironmentDiagnostics()
    all_passed = diagnostics.run_all_tests(env_factory, policy_factory)
    
    if all_passed:
        print("\n✅ Environment is working correctly!")
    else:
        print("\n❌ Some tests failed. Please review the issues above.")
        return

    # =========================================================================
    # Part 2: Baseline Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Policy Comparison")
    print("=" * 70)
    
    policies = {
        "FirstFit": FirstFit(),
        "BestFit": BestFit(),
        "WorstFit": WorstFit(),
        "Random": RandomPolicy(),
    }
    
    results = run_baseline_comparison(
        env_factory=env_factory,
        policies=policies,
        num_episodes=5,
        seed=42,
    )
    
    # =========================================================================
    # Part 3: Learning Tracker Demo
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Learning Tracker Demo")
    print("=" * 70)
    print("\nSimulating a learning agent improving over time...\n")
    
    tracker = LearningTracker()
    
    # Record baseline performance
    for _ in range(5):
        env = env_factory(seed=42)
        obs = env.reset()
        policy = RandomPolicy()
        
        while not env.done:
            action = policy.select_action(obs)
            obs, _, _, _ = env.step(action)
        
        tracker.record_baseline("Random", env.total_reward)
    
    # Simulate learning: start with random, gradually improve
    # In real RL, this would be your training loop
    import numpy as np
    
    for episode in range(200):
        # Simulate improving performance
        # Early episodes: more like random
        # Later episodes: more like FirstFit
        
        env = env_factory(seed=42 + episode)
        obs = env.reset()
        
        # Mix of random and good policy (simulating learning)
        learning_progress = min(1.0, episode / 150)
        random_policy = RandomPolicy()
        good_policy = FirstFit()
        
        step = 0
        while not env.done:
            # As learning progresses, use good policy more often
            if np.random.random() < learning_progress:
                action = good_policy.select_action(obs)
            else:
                action = random_policy.select_action(obs)
            
            obs, _, _, _ = env.step(action)
            step += 1
        
        metrics = env.metrics.get_current_metrics()
        tracker.record_episode(
            total_reward=env.total_reward,
            length=step,
            metrics=metrics,
        )
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Reward = {env.total_reward:.2f}")
    
    # Print learning summary
    print()
    tracker.print_summary()
    
    # Check if "agent" is learning
    if tracker.is_improving():
        print("\n✅ Agent is improving over time!")
    else:
        print("\n⚠️ Agent is not showing clear improvement")
    
    if tracker.beats_baseline("Random"):
        print("✅ Agent beats Random baseline!")
    else:
        print("❌ Agent does not beat Random baseline")
    
    # Try to plot learning curve (if matplotlib available)
    try:
        tracker.plot_learning_curve(save_path="output/learning_curve.png")
    except Exception as e:
        print(f"\nCould not plot learning curve: {e}")


if __name__ == "__main__":
    main()
