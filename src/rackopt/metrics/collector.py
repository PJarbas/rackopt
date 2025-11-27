"""Metrics collection and tracking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventLog:
    """Log entry for a simulation event.

    Attributes:
        timestamp: When event occurred
        event_type: Type of event
        task_id: Related task ID
        node_id: Related node ID (if applicable)
        metadata: Additional event data
    """

    timestamp: float
    event_type: str
    task_id: int
    node_id: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collect and aggregate simulation metrics.

    Tracks various metrics over the course of simulation and provides
    aggregations and exports.
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        # Counters
        self.tasks_arrived = 0
        self.tasks_completed = 0
        self.tasks_rejected = 0
        self.tasks_preempted = 0
        self.sla_violations = 0

        # Timing tracking
        self.arrival_times: dict[int, float] = {}
        self.completion_times: dict[int, float] = {}
        self.response_times: list[float] = []

        # Node tracking
        self.allocations_per_node: dict[int, int] = defaultdict(int)

        # Event log
        self.events: list[EventLog] = []

        # Per-timestep metrics (for time series)
        self.timestep_metrics: list[dict[str, Any]] = []

    def record_arrival(self, task_id: int, time: float) -> None:
        """Record task arrival.

        Args:
            task_id: Task identifier
            time: Arrival time
        """
        self.tasks_arrived += 1
        self.arrival_times[task_id] = time

        self.events.append(EventLog(
            timestamp=time,
            event_type="arrival",
            task_id=task_id,
        ))

    def record_allocation(self, task_id: int, node_id: int, time: float) -> None:
        """Record task allocation to node.

        Args:
            task_id: Task identifier
            node_id: Node identifier
            time: Allocation time
        """
        self.allocations_per_node[node_id] += 1

        self.events.append(EventLog(
            timestamp=time,
            event_type="allocation",
            task_id=task_id,
            node_id=node_id,
        ))

    def record_completion(self, task_id: int, time: float) -> None:
        """Record task completion.

        Args:
            task_id: Task identifier
            time: Completion time
        """
        self.tasks_completed += 1
        self.completion_times[task_id] = time

        # Calculate response time
        if task_id in self.arrival_times:
            response_time = time - self.arrival_times[task_id]
            self.response_times.append(response_time)

        self.events.append(EventLog(
            timestamp=time,
            event_type="completion",
            task_id=task_id,
        ))

    def record_rejection(self, task_id: int, time: float) -> None:
        """Record task rejection.

        Args:
            task_id: Task identifier
            time: Rejection time
        """
        self.tasks_rejected += 1

        self.events.append(EventLog(
            timestamp=time,
            event_type="rejection",
            task_id=task_id,
        ))

    def record_preemption(self, task_id: int, time: float) -> None:
        """Record task preemption.

        Args:
            task_id: Task identifier
            time: Preemption time
        """
        self.tasks_preempted += 1

        self.events.append(EventLog(
            timestamp=time,
            event_type="preemption",
            task_id=task_id,
        ))

    def record_sla_violation(self, task_id: int, time: float) -> None:
        """Record SLA/deadline violation.

        Args:
            task_id: Task identifier
            time: Violation time
        """
        self.sla_violations += 1

        self.events.append(EventLog(
            timestamp=time,
            event_type="sla_violation",
            task_id=task_id,
        ))

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current aggregated metrics.

        Returns:
            Dictionary of metrics
        """
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0.0
        )

        rejection_rate = (
            self.tasks_rejected / self.tasks_arrived
            if self.tasks_arrived > 0 else 0.0
        )

        # Calculate throughput (tasks completed per time unit would need current time)
        # For now, just return count
        throughput = float(self.tasks_completed)

        return {
            "tasks_arrived": self.tasks_arrived,
            "tasks_completed": self.tasks_completed,
            "tasks_rejected": self.tasks_rejected,
            "tasks_preempted": self.tasks_preempted,
            "sla_violations": self.sla_violations,
            "avg_response_time": avg_response_time,
            "rejection_rate": rejection_rate,
            "throughput": throughput,
        }

    def get_response_time_stats(self) -> dict[str, float]:
        """Get detailed response time statistics.

        Returns:
            Dictionary with min, max, mean, std, p50, p95, p99
        """
        if not self.response_times:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        import numpy as np

        times = np.array(self.response_times)

        return {
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
        }

    def get_node_stats(self) -> dict[int, dict[str, Any]]:
        """Get per-node statistics.

        Returns:
            Dictionary mapping node_id to stats
        """
        return {
            node_id: {
                "allocations": count,
            }
            for node_id, count in self.allocations_per_node.items()
        }

    def export_dataframe(self) -> Any:
        """Export metrics to pandas DataFrame.

        Returns:
            DataFrame with time series of events (requires pandas)

        Raises:
            ImportError: If pandas not installed
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Pandas required for export_dataframe. "
                "Install with: pip install pandas"
            ) from e

        # Convert events to DataFrame
        data = []
        for event in self.events:
            row = {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "task_id": event.task_id,
                "node_id": event.node_id if event.node_id >= 0 else None,
            }
            row.update(event.metadata)
            data.append(row)

        df = pd.DataFrame(data)
        return df

    def export_summary(self) -> dict[str, Any]:
        """Export complete summary of metrics.

        Returns:
            Dictionary with all metrics and statistics
        """
        return {
            "current_metrics": self.get_current_metrics(),
            "response_time_stats": self.get_response_time_stats(),
            "node_stats": self.get_node_stats(),
            "total_events": len(self.events),
        }

    def print_summary(self) -> None:
        """Print human-readable summary of metrics."""
        metrics = self.get_current_metrics()
        rt_stats = self.get_response_time_stats()

        print("=" * 60)
        print("SIMULATION METRICS SUMMARY")
        print("=" * 60)
        print(f"Tasks Arrived:     {metrics['tasks_arrived']}")
        print(f"Tasks Completed:   {metrics['tasks_completed']}")
        print(f"Tasks Rejected:    {metrics['tasks_rejected']}")
        print(f"Tasks Preempted:   {metrics['tasks_preempted']}")
        print(f"SLA Violations:    {metrics['sla_violations']}")
        print()
        print(f"Rejection Rate:    {metrics['rejection_rate']:.2%}")
        print()
        print("Response Time Statistics:")
        print(f"  Mean:  {rt_stats['mean']:.2f}")
        print(f"  Std:   {rt_stats['std']:.2f}")
        print(f"  Min:   {rt_stats['min']:.2f}")
        print(f"  P50:   {rt_stats['p50']:.2f}")
        print(f"  P95:   {rt_stats['p95']:.2f}")
        print(f"  P99:   {rt_stats['p99']:.2f}")
        print(f"  Max:   {rt_stats['max']:.2f}")
        print("=" * 60)

    def __repr__(self) -> str:
        """String representation of metrics collector."""
        return (
            f"MetricsCollector(arrived={self.tasks_arrived}, "
            f"completed={self.tasks_completed}, "
            f"rejected={self.tasks_rejected}, "
            f"events={len(self.events)})"
        )
