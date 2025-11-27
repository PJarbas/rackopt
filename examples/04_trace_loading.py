"""Example of loading and simulating trace files."""

import json
from pathlib import Path

from rackopt import ClusterEnv
from rackopt.config.config import ClusterConfig, WorkloadConfig
from rackopt.policies.heuristics import FirstFit
from rackopt.workload.trace_schema import TraceTask, TraceLoader


def create_sample_trace(output_path: str, num_tasks: int = 50):
    """Create a sample trace file for demonstration.

    Args:
        output_path: Path to save trace file
        num_tasks: Number of tasks to generate
    """
    import random
    random.seed(42)

    tasks = []
    current_time = 0.0

    for i in range(num_tasks):
        # Simulate Poisson arrivals
        inter_arrival = random.expovariate(1.0)
        current_time += inter_arrival

        # Generate task
        task = TraceTask(
            task_id=i,
            arrival_time=current_time,
            duration=random.uniform(5.0, 30.0),
            cpu_demand=random.uniform(1.0, 8.0),
            ram_demand=random.uniform(2.0, 16.0),
            deadline=current_time + random.uniform(10.0, 60.0) if random.random() < 0.3 else None,
            priority=random.randint(0, 5),
            task_type=random.choice(["batch", "interactive", "analytics"]),
        )
        tasks.append(task)

    # Save trace
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    TraceLoader.save_json(tasks, output_path)
    print(f"Created sample trace with {num_tasks} tasks: {output_path}")


def main():
    """Demonstrate trace loading and simulation."""
    print("=" * 60)
    print("RackOpt - Trace Loading Example")
    print("=" * 60)
    print()

    # Create sample trace
    trace_path = "traces/sample_trace.json"
    create_sample_trace(trace_path, num_tasks=100)

    # Load and inspect trace
    print(f"\nLoading trace from: {trace_path}")
    tasks = TraceLoader.load_json(trace_path)
    print(f"Loaded {len(tasks)} tasks")
    print()

    # Show first few tasks
    print("First 5 tasks:")
    for task in tasks[:5]:
        print(f"  Task {task.task_id}: "
              f"arrival={task.arrival_time:.2f}, "
              f"duration={task.duration:.2f}, "
              f"cpu={task.cpu_demand:.1f}, "
              f"ram={task.ram_demand:.1f}, "
              f"type={task.task_type}")
    print()

    # Create configuration for trace-based workload
    config = ClusterConfig(
        num_nodes=8,
        node_resources={"cpu": 16.0, "ram": 64.0},
        workload=WorkloadConfig(
            type="trace",
            trace_path=trace_path,
        ),
        max_simulation_time=500.0,
        policy_name="FirstFit",
        seed=42,
    )

    # Run simulation
    print("Running simulation with trace workload...")
    env = ClusterEnv(config)
    policy = FirstFit()

    obs = env.reset()
    done = False
    step = 0

    while not done:
        action = policy.select_action(obs)
        obs, reward, done, info = env.step(action)

        if step % 20 == 0:
            print(f"  Step {step:3d} | "
                  f"Time: {info['time']:7.2f} | "
                  f"Pending: {info['num_pending']:3d}")

        step += 1

    print()
    print("=" * 60)
    print("Simulation Results:")
    print("=" * 60)
    env.metrics.print_summary()


if __name__ == "__main__":
    main()
