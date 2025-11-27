"""Trace file schema validation with Pydantic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TraceTask(BaseModel):
    """Schema for a task in a trace file.

    Required fields:
        arrival_time: Time when task arrives
        duration: Task execution duration
        cpu_demand: CPU resource demand
        ram_demand: RAM resource demand

    Optional fields:
        task_id: Task identifier (auto-generated if not provided)
        deadline: Task deadline
        priority: Task priority
        task_type: Task classification
        Additional resource demands (gpu_demand, disk_demand, etc.)
    """

    # Required fields
    arrival_time: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)
    cpu_demand: float = Field(gt=0.0)
    ram_demand: float = Field(gt=0.0)

    # Optional fields
    task_id: int | None = None
    deadline: float | None = Field(default=None, ge=0.0)
    priority: int = 0
    task_type: str = "default"
    preemptible: bool = False

    # Additional resources (optional)
    gpu_demand: float = Field(default=0.0, ge=0.0)
    disk_demand: float = Field(default=0.0, ge=0.0)
    network_demand: float = Field(default=0.0, ge=0.0)

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, v: float | None, info: Any) -> float | None:
        """Validate deadline is after arrival_time."""
        if v is not None and "arrival_time" in info.data:
            if v < info.data["arrival_time"]:
                raise ValueError("deadline must be >= arrival_time")
        return v

    def get_demands(self) -> dict[str, float]:
        """Get all resource demands as dictionary.

        Returns:
            Dictionary of resource demands (excluding zero demands)
        """
        demands = {
            "cpu": self.cpu_demand,
            "ram": self.ram_demand,
        }

        if self.gpu_demand > 0:
            demands["gpu"] = self.gpu_demand
        if self.disk_demand > 0:
            demands["disk"] = self.disk_demand
        if self.network_demand > 0:
            demands["network"] = self.network_demand

        return demands


class TraceLoader:
    """Utility for loading and validating trace files."""

    @staticmethod
    def load_csv(path: str | Path, has_header: bool = True) -> list[TraceTask]:
        """Load tasks from CSV file.

        Expected CSV format (with header):
            arrival_time,duration,cpu_demand,ram_demand[,optional_fields...]

        Args:
            path: Path to CSV file
            has_header: Whether CSV has header row

        Returns:
            List of validated TraceTask objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is malformed or validation fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        tasks: list[TraceTask] = []

        with open(path, "r") as f:
            reader = csv.DictReader(f) if has_header else csv.reader(f)

            for i, row in enumerate(reader):
                try:
                    if isinstance(row, dict):
                        # Convert string values to appropriate types
                        data = TraceLoader._convert_csv_row(row)
                    else:
                        # No header - assume standard order
                        data = TraceLoader._convert_csv_list(row)

                    # Assign task_id if not provided
                    if "task_id" not in data or data["task_id"] is None:
                        data["task_id"] = i

                    task = TraceTask(**data)
                    tasks.append(task)

                except Exception as e:
                    raise ValueError(f"Error parsing row {i + 1}: {e}") from e

        return tasks

    @staticmethod
    def load_json(path: str | Path) -> list[TraceTask]:
        """Load tasks from JSON file.

        Expected JSON format:
            [
                {
                    "arrival_time": 0.0,
                    "duration": 10.0,
                    "cpu_demand": 4.0,
                    "ram_demand": 8.0,
                    ...
                },
                ...
            ]

        Args:
            path: Path to JSON file

        Returns:
            List of validated TraceTask objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is malformed or validation fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of tasks")

        tasks: list[TraceTask] = []

        for i, task_data in enumerate(data):
            try:
                # Assign task_id if not provided
                if "task_id" not in task_data or task_data["task_id"] is None:
                    task_data["task_id"] = i

                task = TraceTask(**task_data)
                tasks.append(task)

            except Exception as e:
                raise ValueError(f"Error parsing task {i}: {e}") from e

        return tasks

    @staticmethod
    def _convert_csv_row(row: dict[str, str]) -> dict[str, Any]:
        """Convert CSV row with headers to typed dictionary.

        Args:
            row: CSV row as dictionary

        Returns:
            Dictionary with converted types
        """
        data: dict[str, Any] = {}

        # Required fields
        for field in ["arrival_time", "duration", "cpu_demand", "ram_demand"]:
            if field in row:
                data[field] = float(row[field])

        # Optional numeric fields
        for field in ["task_id", "priority"]:
            if field in row and row[field]:
                data[field] = int(row[field])

        for field in ["deadline", "gpu_demand", "disk_demand", "network_demand"]:
            if field in row and row[field]:
                data[field] = float(row[field])

        # Optional string/boolean fields
        if "task_type" in row and row["task_type"]:
            data["task_type"] = row["task_type"]

        if "preemptible" in row and row["preemptible"]:
            data["preemptible"] = row["preemptible"].lower() in ("true", "1", "yes")

        return data

    @staticmethod
    def _convert_csv_list(row: list[str]) -> dict[str, Any]:
        """Convert CSV row without headers (positional) to typed dictionary.

        Assumes order: arrival_time,duration,cpu_demand,ram_demand[,deadline]

        Args:
            row: CSV row as list

        Returns:
            Dictionary with converted types
        """
        if len(row) < 4:
            raise ValueError("CSV row must have at least 4 columns")

        data = {
            "arrival_time": float(row[0]),
            "duration": float(row[1]),
            "cpu_demand": float(row[2]),
            "ram_demand": float(row[3]),
        }

        if len(row) > 4 and row[4]:
            data["deadline"] = float(row[4])

        return data

    @staticmethod
    def save_csv(tasks: list[TraceTask], path: str | Path) -> None:
        """Save tasks to CSV file.

        Args:
            tasks: List of tasks to save
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not tasks:
            # Create empty file with header
            with open(path, "w") as f:
                f.write("arrival_time,duration,cpu_demand,ram_demand,task_id,deadline,priority,task_type,preemptible\n")
            return

        with open(path, "w", newline="") as f:
            # Get all fields from first task
            fieldnames = [
                "arrival_time", "duration", "cpu_demand", "ram_demand",
                "task_id", "deadline", "priority", "task_type", "preemptible",
                "gpu_demand", "disk_demand", "network_demand"
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for task in tasks:
                row = task.model_dump()
                writer.writerow(row)

    @staticmethod
    def save_json(tasks: list[TraceTask], path: str | Path) -> None:
        """Save tasks to JSON file.

        Args:
            tasks: List of tasks to save
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [task.model_dump() for task in tasks]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
