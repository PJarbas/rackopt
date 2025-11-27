"""Diagnostic tools to validate environment and track learning progress."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class EnvironmentDiagnostics:
    """Diagnostics for validating environment correctness.
    
    Use this to verify:
    1. Environment behaves deterministically with same seed
    2. Actions have expected effects
    3. Rewards are sensible
    4. State transitions are valid
    """
    
    passed_tests: list[str] = field(default_factory=list)
    failed_tests: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def run_all_tests(self, env_factory: Callable, policy_factory: Callable) -> bool:
        """Run all diagnostic tests.
        
        Args:
            env_factory: Callable that creates a new environment
            policy_factory: Callable that creates a policy
            
        Returns:
            True if all tests passed
        """
        print("=" * 60)
        print("🔬 RackOpt Environment Diagnostics")
        print("=" * 60)
        
        self.test_determinism(env_factory, policy_factory)
        self.test_reset_independence(env_factory)
        self.test_action_effects(env_factory)
        self.test_reward_sanity(env_factory, policy_factory)
        self.test_observation_validity(env_factory)
        self.test_done_condition(env_factory, policy_factory)
        
        print("\n" + "=" * 60)
        print("📊 Results Summary")
        print("=" * 60)
        print(f"✅ Passed: {len(self.passed_tests)}")
        print(f"❌ Failed: {len(self.failed_tests)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.failed_tests:
            print("\n❌ Failed tests:")
            for name, reason in self.failed_tests:
                print(f"   - {name}: {reason}")
                
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        print("=" * 60)
        return len(self.failed_tests) == 0
    
    def test_determinism(self, env_factory: Callable, policy_factory: Callable) -> None:
        """Test that same seed produces same trajectory."""
        print("\n🧪 Test: Determinism (same seed → same results)")
        
        try:
            # Run twice with same seed
            rewards1 = self._run_episode(env_factory, policy_factory, seed=42)
            rewards2 = self._run_episode(env_factory, policy_factory, seed=42)
            
            if np.allclose(rewards1, rewards2):
                self.passed_tests.append("determinism")
                print("   ✅ Environment is deterministic")
            else:
                self.failed_tests.append(("determinism", "Different rewards with same seed"))
                print("   ❌ Environment is NOT deterministic")
                print(f"      Run 1 total: {sum(rewards1):.2f}")
                print(f"      Run 2 total: {sum(rewards2):.2f}")
        except Exception as e:
            self.failed_tests.append(("determinism", str(e)))
            print(f"   ❌ Error: {e}")
    
    def test_reset_independence(self, env_factory: Callable) -> None:
        """Test that reset properly clears state."""
        print("\n🧪 Test: Reset Independence")
        
        try:
            env = env_factory(seed=42)
            
            # Run some steps
            obs1 = env.reset()
            for _ in range(10):
                from rackopt.policies.heuristics import FirstFit
                policy = FirstFit()
                action = policy.select_action(obs1)
                obs1, _, done, _ = env.step(action)
                if done:
                    break
            
            time_after_steps = env.current_time
            reward_after_steps = env.total_reward
            
            # Reset and check initial state
            obs2 = env.reset()
            
            if env.current_time == 0.0 and env.total_reward == 0.0:
                self.passed_tests.append("reset_independence")
                print("   ✅ Reset properly clears state")
            else:
                self.failed_tests.append(("reset_independence", 
                    f"State not cleared: time={env.current_time}, reward={env.total_reward}"))
                print("   ❌ Reset does NOT clear state properly")
        except Exception as e:
            self.failed_tests.append(("reset_independence", str(e)))
            print(f"   ❌ Error: {e}")
    
    def test_action_effects(self, env_factory: Callable) -> None:
        """Test that actions have expected effects."""
        print("\n🧪 Test: Action Effects")
        
        try:
            from rackopt.core.action import Action, REJECT
            
            env = env_factory(seed=42)
            obs = env.reset()
            
            # Wait for a pending task
            while len(obs.pending_tasks) == 0 and not env.done:
                obs, _, _, _ = env.step(Action())
            
            if len(obs.pending_tasks) == 0:
                self.warnings.append("No pending tasks generated for action test")
                print("   ⚠️  No pending tasks to test")
                return
            
            task = obs.pending_tasks[0]
            
            # Test allocation
            action = Action()
            action.add_decision(task.task_id, 0)  # Allocate to node 0
            
            obs_after, reward, _, _ = env.step(action)
            
            # Check task was allocated
            node0_tasks = [t for t in obs_after.nodes[0].task_ids]
            if task.task_id in node0_tasks:
                self.passed_tests.append("action_allocation")
                print("   ✅ Task allocation works correctly")
            else:
                self.failed_tests.append(("action_allocation", "Task not found on target node"))
                print("   ❌ Task allocation failed")
                
        except Exception as e:
            self.failed_tests.append(("action_effects", str(e)))
            print(f"   ❌ Error: {e}")
    
    def test_reward_sanity(self, env_factory: Callable, policy_factory: Callable) -> None:
        """Test that rewards are sensible."""
        print("\n🧪 Test: Reward Sanity")
        
        try:
            # Run with good policy
            good_rewards = self._run_episode(env_factory, policy_factory, seed=42)
            
            # Run with random rejection policy
            def bad_policy_factory():
                from rackopt.policies.base import BasePolicy
                from rackopt.core.action import Action, REJECT
                
                class RejectAllPolicy(BasePolicy):
                    def select_action(self, obs):
                        action = Action()
                        for task in obs.pending_tasks:
                            action.add_decision(task.task_id, REJECT)
                        return action
                return RejectAllPolicy()
            
            bad_rewards = self._run_episode(env_factory, bad_policy_factory, seed=42)
            
            good_total = sum(good_rewards)
            bad_total = sum(bad_rewards)
            
            if good_total > bad_total:
                self.passed_tests.append("reward_sanity")
                print(f"   ✅ Good policy ({good_total:.1f}) > Bad policy ({bad_total:.1f})")
            else:
                self.failed_tests.append(("reward_sanity", 
                    f"Good policy ({good_total:.1f}) <= Bad policy ({bad_total:.1f})"))
                print(f"   ❌ Reward ordering incorrect")
                
        except Exception as e:
            self.failed_tests.append(("reward_sanity", str(e)))
            print(f"   ❌ Error: {e}")
    
    def test_observation_validity(self, env_factory: Callable) -> None:
        """Test that observations are valid and consistent."""
        print("\n🧪 Test: Observation Validity")
        
        try:
            env = env_factory(seed=42)
            obs = env.reset()
            
            errors = []
            
            # Check observation structure
            if not hasattr(obs, 'nodes'):
                errors.append("Missing 'nodes' attribute")
            if not hasattr(obs, 'pending_tasks'):
                errors.append("Missing 'pending_tasks' attribute")
            if not hasattr(obs, 'current_time'):
                errors.append("Missing 'current_time' attribute")
            
            # Check node data
            for i, node in enumerate(obs.nodes):
                # NodeState has capacity and usage, calculate available
                available_cpu = node.capacity.get('cpu', 0) - node.usage.get('cpu', 0)
                available_ram = node.capacity.get('ram', 0) - node.usage.get('ram', 0)
                if available_cpu < -0.001:  # Small tolerance for floating point
                    errors.append(f"Node {i} has negative available CPU: {available_cpu}")
                if available_ram < -0.001:
                    errors.append(f"Node {i} has negative available RAM: {available_ram}")
            
            # Check conversion methods
            try:
                d = obs.to_dict()
                if not isinstance(d, dict):
                    errors.append("to_dict() doesn't return dict")
            except Exception as e:
                errors.append(f"to_dict() failed: {e}")
            
            try:
                arr = obs.to_numpy()
                if not isinstance(arr, dict):
                    errors.append("to_numpy() doesn't return dict")
                elif 'nodes' not in arr:
                    errors.append("to_numpy() missing 'nodes' key")
            except Exception as e:
                errors.append(f"to_numpy() failed: {e}")
            
            if errors:
                self.failed_tests.append(("observation_validity", "; ".join(errors)))
                print(f"   ❌ Observation issues: {errors}")
            else:
                self.passed_tests.append("observation_validity")
                print("   ✅ Observations are valid")
                
        except Exception as e:
            self.failed_tests.append(("observation_validity", str(e)))
            print(f"   ❌ Error: {e}")
    
    def test_done_condition(self, env_factory: Callable, policy_factory: Callable) -> None:
        """Test that episode terminates correctly."""
        print("\n🧪 Test: Episode Termination")
        
        try:
            env = env_factory(seed=42)
            policy = policy_factory()
            obs = env.reset()
            
            max_steps = 10000
            steps = 0
            
            while not env.done and steps < max_steps:
                action = policy.select_action(obs)
                obs, _, _, _ = env.step(action)
                steps += 1
            
            if env.done:
                self.passed_tests.append("done_condition")
                print(f"   ✅ Episode terminated after {steps} steps")
            else:
                self.failed_tests.append(("done_condition", f"No termination after {max_steps} steps"))
                print(f"   ❌ Episode did not terminate")
                
        except Exception as e:
            self.failed_tests.append(("done_condition", str(e)))
            print(f"   ❌ Error: {e}")
    
    def _run_episode(self, env_factory: Callable, policy_factory: Callable, 
                     seed: int = 42) -> list[float]:
        """Run a complete episode and return rewards."""
        env = env_factory(seed=seed)
        policy = policy_factory()
        obs = env.reset()
        
        rewards = []
        while not env.done:
            action = policy.select_action(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            
            if len(rewards) > 10000:  # Safety limit
                break
        
        return rewards


@dataclass 
class LearningTracker:
    """Track learning progress over training episodes.
    
    Use this to monitor:
    1. Reward trends (is the agent improving?)
    2. Policy quality (is it better than baselines?)
    3. Convergence (has learning stabilized?)
    """
    
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_metrics: list[dict] = field(default_factory=list)
    baseline_rewards: dict[str, list[float]] = field(default_factory=dict)
    
    def record_episode(self, total_reward: float, length: int, 
                       metrics: dict | None = None) -> None:
        """Record results from one episode.
        
        Args:
            total_reward: Total reward for episode
            length: Number of steps in episode
            metrics: Optional additional metrics
        """
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_metrics.append(metrics or {})
    
    def record_baseline(self, name: str, reward: float) -> None:
        """Record a baseline policy result.
        
        Args:
            name: Baseline policy name
            reward: Total reward achieved
        """
        if name not in self.baseline_rewards:
            self.baseline_rewards[name] = []
        self.baseline_rewards[name].append(reward)
    
    def get_moving_average(self, window: int = 100) -> list[float]:
        """Get moving average of rewards.
        
        Args:
            window: Window size for averaging
            
        Returns:
            List of moving average values
        """
        if len(self.episode_rewards) < window:
            return self.episode_rewards.copy()
        
        result = []
        for i in range(len(self.episode_rewards) - window + 1):
            avg = np.mean(self.episode_rewards[i:i+window])
            result.append(avg)
        return result
    
    def is_improving(self, window: int = 100, threshold: float = 0.05) -> bool:
        """Check if agent is improving.
        
        Args:
            window: Window size for comparison
            threshold: Minimum improvement ratio
            
        Returns:
            True if recent performance > earlier performance
        """
        if len(self.episode_rewards) < window * 2:
            return True  # Not enough data
        
        early = np.mean(self.episode_rewards[:window])
        recent = np.mean(self.episode_rewards[-window:])
        
        if early == 0:
            return recent > 0
        
        improvement = (recent - early) / abs(early)
        return improvement > threshold
    
    def beats_baseline(self, baseline_name: str, confidence: float = 0.95) -> bool:
        """Check if agent beats a baseline.
        
        Args:
            baseline_name: Name of baseline to compare
            confidence: Confidence level for comparison
            
        Returns:
            True if agent is better than baseline
        """
        if baseline_name not in self.baseline_rewards:
            return False
        
        if len(self.episode_rewards) < 10:
            return False
        
        agent_mean = np.mean(self.episode_rewards[-100:])
        baseline_mean = np.mean(self.baseline_rewards[baseline_name])
        
        return agent_mean > baseline_mean
    
    def print_summary(self, last_n: int = 100) -> None:
        """Print learning progress summary.
        
        Args:
            last_n: Number of recent episodes to summarize
        """
        print("=" * 60)
        print("📈 Learning Progress Summary")
        print("=" * 60)
        
        if not self.episode_rewards:
            print("No episodes recorded yet.")
            return
        
        n = min(last_n, len(self.episode_rewards))
        recent = self.episode_rewards[-n:]
        
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"\nLast {n} episodes:")
        print(f"  Mean reward:   {np.mean(recent):.2f}")
        print(f"  Std reward:    {np.std(recent):.2f}")
        print(f"  Min reward:    {np.min(recent):.2f}")
        print(f"  Max reward:    {np.max(recent):.2f}")
        
        if len(self.episode_rewards) >= 200:
            early = np.mean(self.episode_rewards[:100])
            late = np.mean(self.episode_rewards[-100:])
            improvement = ((late - early) / abs(early) * 100) if early != 0 else 0
            print(f"\nImprovement: {improvement:+.1f}% (early 100 → last 100)")
            
            if improvement > 10:
                print("  ✅ Agent is learning!")
            elif improvement > 0:
                print("  📊 Slight improvement")
            elif improvement > -10:
                print("  ⚠️  No significant change")
            else:
                print("  ❌ Performance degrading")
        
        if self.baseline_rewards:
            print("\nBaseline Comparison:")
            agent_mean = np.mean(recent)
            for name, rewards in self.baseline_rewards.items():
                baseline_mean = np.mean(rewards)
                diff = agent_mean - baseline_mean
                status = "✅" if diff > 0 else "❌"
                print(f"  {status} vs {name}: {diff:+.2f} ({agent_mean:.2f} vs {baseline_mean:.2f})")
        
        print("=" * 60)
    
    def plot_learning_curve(self, save_path: str | None = None) -> None:
        """Plot learning curve (requires matplotlib).
        
        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Reward curve
        ax1 = axes[0]
        ax1.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        
        if len(self.episode_rewards) >= 100:
            ma = self.get_moving_average(100)
            ax1.plot(range(99, len(self.episode_rewards)), ma, 
                    label='Moving Avg (100)', linewidth=2)
        
        # Add baselines
        for name, rewards in self.baseline_rewards.items():
            ax1.axhline(y=np.mean(rewards), linestyle='--', 
                       label=f'{name} baseline', alpha=0.7)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Learning Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode length curve
        ax2 = axes[1]
        ax2.plot(self.episode_lengths, alpha=0.5)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved learning curve to {save_path}")
        else:
            plt.show()


def run_baseline_comparison(env_factory: Callable, policies: dict[str, Any],
                           num_episodes: int = 10, seed: int = 42) -> dict[str, dict]:
    """Compare multiple policies on the same environment.
    
    Args:
        env_factory: Callable that creates environment
        policies: Dict of {name: policy} to compare
        num_episodes: Number of episodes per policy
        seed: Base random seed
        
    Returns:
        Dict of {policy_name: {metrics}}
    """
    print("=" * 60)
    print("🏆 Policy Comparison")
    print("=" * 60)
    
    results = {}
    
    for name, policy in policies.items():
        print(f"\nEvaluating: {name}")
        
        episode_rewards = []
        episode_completions = []
        episode_rejections = []
        
        for ep in range(num_episodes):
            env = env_factory(seed=seed + ep)
            obs = env.reset()
            
            while not env.done:
                action = policy.select_action(obs)
                obs, reward, done, info = env.step(action)
            
            metrics = env.metrics.get_current_metrics()
            episode_rewards.append(env.total_reward)
            episode_completions.append(metrics.get('tasks_completed', 0))
            episode_rejections.append(metrics.get('rejection_rate', 0))
        
        results[name] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_completions': np.mean(episode_completions),
            'mean_rejection_rate': np.mean(episode_rejections),
        }
        
        print(f"  Reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
    
    # Rank policies
    print("\n" + "=" * 60)
    print("📊 Rankings (by mean reward)")
    print("=" * 60)
    
    ranked = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    for i, (name, metrics) in enumerate(ranked, 1):
        print(f"  {i}. {name}: {metrics['mean_reward']:.2f}")
    
    return results
