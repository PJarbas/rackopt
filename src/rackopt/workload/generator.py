"""Workload generation for simulation."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from rackopt.core.models import Task, TaskState
from rackopt.env.event_system import TaskArrival
from rackopt.workload.trace_schema import TraceLoader

if TYPE_CHECKING:
    from rackopt.config.config import WorkloadConfig


class WorkloadGenerator:
    """Generate synthetic workloads or load from traces.

    Supports various arrival and duration distributions for realistic
    workload simulation.
    """

    def __init__(self, config: WorkloadConfig, seed: int | None = None):
        """Initialize workload generator.

        Args:
            config: Workload configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self._task_counter = 0

        # Initialize RNG
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_events(self, until_time: float | None = None) -> list[TaskArrival]:
        """Generate task arrival events.

        Args:
            until_time: Generate events up to this time (None = use config)

        Returns:
            List of TaskArrival events
        """
        if self.config.type == "trace":
            return self._load_trace_events()
        else:
            return self._generate_synthetic_events(until_time)

    def _generate_synthetic_events(self, until_time: float | None = None) -> list[TaskArrival]:
        """Generate synthetic task arrivals.

        Args:
            until_time: Maximum time for arrivals

        Returns:
            List of TaskArrival events
        """
        events: list[TaskArrival] = []
        current_time = 0.0
        max_time = until_time if until_time is not None else float("inf")

        for _ in range(self.config.num_tasks):
            # Generate inter-arrival time
            inter_arrival = self._sample_arrival_time()
            current_time += inter_arrival

            if current_time > max_time:
                break

            # Generate task
            task = self._generate_task(arrival_time=current_time)

            # Create event
            event = TaskArrival(timestamp=current_time, task=task)
            events.append(event)

        return events

    def _load_trace_events(self) -> list[TaskArrival]:
        """Load task arrivals from trace file.

        Returns:
            List of TaskArrival events

        Raises:
            ValueError: If trace_path is not set
        """
        if self.config.trace_path is None:
            raise ValueError("trace_path must be set for trace-based workload")

        # Load trace based on file extension
        path = self.config.trace_path
        if path.endswith(".json"):
            trace_tasks = TraceLoader.load_json(path)
        else:
            trace_tasks = TraceLoader.load_csv(path)

        # Convert to events
        events: list[TaskArrival] = []

        for trace_task in trace_tasks:
            task = Task(
                task_id=trace_task.task_id if trace_task.task_id is not None else self._get_next_task_id(),
                demands=trace_task.get_demands(),
                duration=trace_task.duration,
                arrival_time=trace_task.arrival_time,
                deadline=trace_task.deadline,
                priority=trace_task.priority,
                task_type=trace_task.task_type,
                preemptible=trace_task.preemptible,
                state=TaskState.WAITING,
            )

            event = TaskArrival(timestamp=trace_task.arrival_time, task=task)
            events.append(event)

        # Sort by arrival time
        events.sort(key=lambda e: e.timestamp)

        return events

    def _generate_task(self, arrival_time: float) -> Task:
        """Generate a single synthetic task.

        Args:
            arrival_time: Time when task arrives

        Returns:
            Task object
        """
        task_id = self._get_next_task_id()

        # Generate duration
        duration = self._sample_duration()

        # Generate resource demands
        demands = self._sample_demands()

        # Generate deadline (optional)
        deadline = None
        if random.random() < self.config.deadline_probability:
            deadline = arrival_time + duration * self.config.deadline_slack_factor

        # Generate preemptible flag
        preemptible = random.random() < self.config.preemptible_probability

        return Task(
            task_id=task_id,
            demands=demands,
            duration=duration,
            arrival_time=arrival_time,
            deadline=deadline,
            preemptible=preemptible,
            state=TaskState.WAITING,
        )

    def _sample_arrival_time(self) -> float:
        """Sample inter-arrival time based on configured distribution.

        Returns:
            Time until next arrival
        """
        if self.config.arrival_distribution == "poisson":
            # Poisson process: exponential inter-arrival times
            return np.random.exponential(1.0 / self.config.arrival_rate)
        elif self.config.arrival_distribution == "uniform":
            # Uniform inter-arrival
            mean_interval = 1.0 / self.config.arrival_rate
            return np.random.uniform(0.5 * mean_interval, 1.5 * mean_interval)
        elif self.config.arrival_distribution == "constant":
            # Fixed inter-arrival time
            return 1.0 / self.config.arrival_rate
        else:
            # Default to exponential
            return np.random.exponential(1.0 / self.config.arrival_rate)

    def _sample_duration(self) -> float:
        """Sample task duration based on configured distribution.

        Returns:
            Task duration
        """
        dist = self.config.duration_distribution
        params = self.config.duration_params

        if dist == "exponential":
            mean = params.get("mean", 10.0)
            return np.random.exponential(mean)

        elif dist == "uniform":
            min_dur = params.get("min", 5.0)
            max_dur = params.get("max", 20.0)
            return np.random.uniform(min_dur, max_dur)

        elif dist == "normal":
            mean = params.get("mean", 10.0)
            std = params.get("std", 2.0)
            # Ensure positive duration
            return max(0.1, np.random.normal(mean, std))

        elif dist == "constant":
            return params.get("value", 10.0)

        else:
            # Default to exponential
            return np.random.exponential(params.get("mean", 10.0))

    def _sample_demands(self) -> dict[str, float]:
        """Sample resource demands based on configured specifications.

        Returns:
            Dictionary of resource demands
        """
        demands: dict[str, float] = {}

        for resource, spec in self.config.resource_demands.items():
            min_demand = spec.get("min", 1.0)
            max_demand = spec.get("max", min_demand * 2)

            # Sample uniformly between min and max
            demand = np.random.uniform(min_demand, max_demand)
            demands[resource] = demand

        return demands

    def _get_next_task_id(self) -> int:
        """Get next task ID.

        Returns:
            Unique task ID
        """
        task_id = self._task_counter
        self._task_counter += 1
        return task_id

    def reset(self) -> None:
        """Reset generator state."""
        self._task_counter = 0
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)


# Predefined workload profiles
class WorkloadProfiles:
    """Collection of predefined workload generation profiles."""

    @staticmethod
    def batch_processing() -> WorkloadGenerator:
        """Batch processing workload: large tasks, low arrival rate."""
        from rackopt.config.config import WorkloadConfig

        config = WorkloadConfig(
            type="synthetic",
            arrival_distribution="poisson",
            arrival_rate=0.5,
            duration_distribution="uniform",
            duration_params={"min": 50.0, "max": 150.0},
            resource_demands={
                "cpu": {"min": 8.0, "max": 16.0},
                "ram": {"min": 16.0, "max": 64.0},
            },
            num_tasks=100,
            deadline_probability=0.1,
            preemptible_probability=0.1,
        )
        return WorkloadGenerator(config)

    @staticmethod
    def interactive() -> WorkloadGenerator:
        """Interactive workload: small tasks, high arrival rate."""
        from rackopt.config.config import WorkloadConfig

        config = WorkloadConfig(
            type="synthetic",
            arrival_distribution="poisson",
            arrival_rate=3.0,
            duration_distribution="exponential",
            duration_params={"mean": 10.0},
            resource_demands={
                "cpu": {"min": 1.0, "max": 4.0},
                "ram": {"min": 2.0, "max": 8.0},
            },
            num_tasks=300,
            deadline_probability=0.7,
            deadline_slack_factor=1.5,
            preemptible_probability=0.3,
        )
        return WorkloadGenerator(config)

    @staticmethod
    def mixed() -> WorkloadGenerator:
        """Mixed workload: variety of task sizes and durations."""
        from rackopt.config.config import WorkloadConfig

        config = WorkloadConfig(
            type="synthetic",
            arrival_distribution="poisson",
            arrival_rate=1.5,
            duration_distribution="uniform",
            duration_params={"min": 10.0, "max": 100.0},
            resource_demands={
                "cpu": {"min": 2.0, "max": 16.0},
                "ram": {"min": 4.0, "max": 32.0},
            },
            num_tasks=200,
            deadline_probability=0.5,
            deadline_slack_factor=2.0,
            preemptible_probability=0.2,
        )
        return WorkloadGenerator(config)

    @staticmethod
    def bursty() -> WorkloadGenerator:
        """Bursty workload: periods of high and low load."""
        from rackopt.config.config import WorkloadConfig

        # Note: True bursty behavior would need custom arrival logic
        # This is an approximation with higher variance
        config = WorkloadConfig(
            type="synthetic",
            arrival_distribution="uniform",  # More variable than Poisson
            arrival_rate=2.0,
            duration_distribution="exponential",
            duration_params={"mean": 15.0},
            resource_demands={
                "cpu": {"min": 1.0, "max": 8.0},
                "ram": {"min": 2.0, "max": 16.0},
            },
            num_tasks=250,
            deadline_probability=0.4,
            preemptible_probability=0.25,
        )
        return WorkloadGenerator(config)
