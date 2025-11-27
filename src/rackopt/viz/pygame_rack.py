"""Professional Pygame-based rack visualization for cluster simulation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import pygame
except ImportError as e:
    raise ImportError(
        "Pygame visualization requires pygame. Install with: pip install rackopt[viz]"
    ) from e

# Optional video recording support
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class RendererClosedError(Exception):
    """Exception raised when renderer is closed but render_step is called."""

    pass


class VideoRecordingError(Exception):
    """Exception raised when video recording fails."""

    pass


# Color scheme for professional visualization
COLORS = {
    "background": (20, 20, 20),
    "rack_bg": (30, 30, 40),
    "node_border": (200, 200, 200),
    "node_bg": (40, 40, 50),
    "slot_empty": (60, 60, 70),
    "slot_running_ok": (0, 180, 0),
    "slot_running_warning": (200, 180, 0),
    "slot_running_late": (200, 0, 0),
    "slot_finished": (80, 80, 80),
    "slot_rejected": (150, 0, 150),
    "text": (230, 230, 230),
    "text_dark": (150, 150, 150),
    "kpi_good": (0, 200, 0),
    "kpi_bad": (200, 0, 0),
    "kpi_neutral": (230, 230, 230),
    "header_bg": (25, 25, 30),
    "panel_bg": (25, 25, 30),
}


class PygameRackRenderer:
    """Professional visualization of cluster rack with task slots.

    Renders nodes as vertical blades with colored slots representing tasks.
    Includes header, KPI panel, and legend.
    Supports video recording to MP4 files.
    """

    def __init__(
        self,
        env: Any,
        width: int = 1200,
        height: int = 800,
        title: str = "RackOpt - Cluster Simulation",
        max_slots_per_node: int | None = None,
        fps: int = 30,
        record_video: bool = False,
        video_path: str | Path | None = None,
    ):
        """Initialize Pygame rack renderer.

        Args:
            env: ClusterEnv instance (must have get_state_snapshot() method)
            width: Window width in pixels
            height: Window height in pixels
            title: Window title
            max_slots_per_node: Maximum slots to display per node (None = auto)
            fps: Frames per second limit
            record_video: Whether to record video (requires opencv-python)
            video_path: Path to save video file (default: simulation.mp4)
        """
        self.env = env
        self.width = width
        self.height = height
        self.title = title
        self.max_slots_per_node = max_slots_per_node or 32
        self.fps = fps
        self.closed = False

        # Video recording state
        self._recording = False
        self._video_writer: Any = None
        self._video_path: Path | None = None
        self._frame_count = 0

        # Layout constants
        self.HEADER_HEIGHT = 60
        self.PANEL_HEIGHT = 150
        self.MARGIN_LEFT = 40
        self.MARGIN_RIGHT = 40
        self.MARGIN_TOP = self.HEADER_HEIGHT + 20
        self.MARGIN_BOTTOM = self.PANEL_HEIGHT + 20
        self.NODE_PADDING = 10
        self.SLOT_PADDING = 2

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Policy name (try to get from env config)
        self.policy_name = "Unknown"
        if hasattr(env, "config") and hasattr(env.config, "policy_name"):
            self.policy_name = env.config.policy_name

        # Seed info
        self.seed = None
        if hasattr(env, "seed"):
            self.seed = env.seed

        # Start recording if requested
        if record_video:
            self.start_recording(video_path)

    def start_recording(self, video_path: str | Path | None = None) -> None:
        """Start recording video.

        Args:
            video_path: Path to save video file (default: simulation.avi)

        Raises:
            VideoRecordingError: If OpenCV is not available or recording fails
        """
        if not HAS_OPENCV:
            raise VideoRecordingError(
                "Video recording requires opencv-python. Install with: pip install opencv-python"
            )

        if self._recording:
            return  # Already recording

        # Set video path (use AVI for better codec compatibility)
        if video_path is None:
            self._video_path = Path("simulation.avi")
        else:
            self._video_path = Path(video_path)
            # Recommend AVI format for compatibility
            if self._video_path.suffix.lower() == ".mp4":
                self._video_path = self._video_path.with_suffix(".avi")
                print(f"Note: Using AVI format for better compatibility: {self._video_path}")
        
        # Create output directory if needed
        self._video_path.parent.mkdir(parents=True, exist_ok=True)

        # Use MJPG codec (more compatible than mp4v)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._video_writer = cv2.VideoWriter(
            str(self._video_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self._video_writer.isOpened():
            raise VideoRecordingError(f"Failed to open video writer for {self._video_path}")

        self._recording = True
        self._frame_count = 0
        print(f"🎬 Recording started: {self._video_path}")

    def stop_recording(self) -> Path | None:
        """Stop recording and save video.

        Returns:
            Path to saved video file, or None if not recording
        """
        if not self._recording or self._video_writer is None:
            return None

        self._video_writer.release()
        self._video_writer = None
        self._recording = False

        saved_path = self._video_path
        print(f"🎬 Recording saved: {saved_path} ({self._frame_count} frames)")
        self._frame_count = 0

        return saved_path

    def is_recording(self) -> bool:
        """Check if video recording is active.

        Returns:
            True if recording, False otherwise
        """
        return self._recording

    def save_screenshot(self, path: str | Path | None = None) -> Path:
        """Save current frame as a PNG image.

        Args:
            path: Path to save image (default: screenshot_<timestamp>.png)

        Returns:
            Path to saved image file
        """
        if path is None:
            import time
            timestamp = int(time.time() * 1000)
            path = Path(f"screenshot_{timestamp}.png")
        else:
            path = Path(path)

        # Create output directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save pygame surface as PNG
        pygame.image.save(self.screen, str(path))
        return path

    def _capture_frame(self) -> None:
        """Capture current frame for video recording."""
        if not self._recording or self._video_writer is None:
            return

        # Get pygame surface as string buffer
        frame_data = pygame.surfarray.array3d(self.screen)

        # Pygame uses (width, height, 3) in RGB
        # OpenCV expects (height, width, 3) in BGR
        frame = np.transpose(frame_data, (1, 0, 2))  # Swap width/height
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

        self._video_writer.write(frame)
        self._frame_count += 1

    def render_step(self) -> None:
        """Render one frame of the visualization.

        Raises:
            RendererClosedError: If renderer was closed
        """
        if self.closed:
            return  # Silently return if closed

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return

        # Get state snapshot from environment
        snapshot = self.env.get_state_snapshot()

        # Clear screen
        self.screen.fill(COLORS["background"])

        # Draw components
        self._draw_header(snapshot)
        self._draw_rack(snapshot)
        self._draw_panel(snapshot)
        self._draw_legend()

        # Capture frame for video if recording
        if self._recording:
            self._capture_frame()

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        """Close the renderer and cleanup resources."""
        if not self.closed:
            # Stop recording if active
            if self._recording:
                self.stop_recording()
            pygame.quit()
            self.closed = True

    def _draw_header(self, snapshot: dict[str, Any]) -> None:
        """Draw header with time, policy, and seed info.

        Args:
            snapshot: State snapshot from environment
        """
        # Background
        header_rect = pygame.Rect(0, 0, self.width, self.HEADER_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["header_bg"], header_rect)

        # Time
        time = snapshot.get("time", 0.0)
        time_text = self.font_large.render(f"Time: {time:.1f}", True, COLORS["text"])
        self.screen.blit(time_text, (20, 15))

        # Policy name
        policy_text = self.font_medium.render(
            f"Policy: {self.policy_name}", True, COLORS["text_dark"]
        )
        self.screen.blit(policy_text, (self.width // 2 - 100, 20))

        # Seed info
        if self.seed is not None:
            seed_text = self.font_small.render(
                f"Seed: {self.seed}", True, COLORS["text_dark"]
            )
            self.screen.blit(seed_text, (self.width - 150, 25))

    def _draw_rack(self, snapshot: dict[str, Any]) -> None:
        """Draw the cluster rack with nodes and task slots.

        Args:
            snapshot: State snapshot from environment
        """
        nodes = snapshot.get("nodes", [])
        if not nodes:
            return

        num_nodes = len(nodes)

        # Calculate layout
        rack_width = self.width - self.MARGIN_LEFT - self.MARGIN_RIGHT
        rack_height = (
            self.height - self.MARGIN_TOP - self.MARGIN_BOTTOM - self.PANEL_HEIGHT
        )

        # Node dimensions
        node_width = (rack_width - (num_nodes + 1) * self.NODE_PADDING) / num_nodes
        node_width = max(50, min(node_width, 150))  # Clamp width

        # Calculate max slots needed
        max_tasks = max(
            (len(node.get("tasks", [])) for node in nodes), default=0
        )
        num_slots = min(max_tasks + 2, self.max_slots_per_node)  # +2 for buffer
        num_slots = max(num_slots, 4)  # Minimum 4 slots

        # Draw each node
        for i, node in enumerate(nodes):
            node_x = self.MARGIN_LEFT + i * (node_width + self.NODE_PADDING)
            node_y = self.MARGIN_TOP

            self._draw_node(node, node_x, node_y, node_width, rack_height, num_slots)

    def _draw_node(
        self,
        node: dict[str, Any],
        x: float,
        y: float,
        width: float,
        height: float,
        num_slots: int,
    ) -> None:
        """Draw a single node blade with task slots.

        Args:
            node: Node data from snapshot
            x: X position
            y: Y position
            width: Node width
            height: Node height
            num_slots: Number of slots to display
        """
        # Node background
        node_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLORS["node_bg"], node_rect)
        pygame.draw.rect(self.screen, COLORS["node_border"], node_rect, 2)

        # Node name at top
        node_name = node.get("name", f"Node {node.get('id', '?')}")
        name_text = self.font_small.render(node_name, True, COLORS["text"])
        name_rect = name_text.get_rect(centerx=x + width / 2, top=y + 5)
        self.screen.blit(name_text, name_rect)

        # Resource utilization
        capacity = node.get("capacity", {})
        usage = node.get("usage", {})

        # CPU utilization
        cpu_util = 0.0
        if "cpu" in capacity and capacity["cpu"] > 0:
            cpu_util = usage.get("cpu", 0.0) / capacity["cpu"]

        util_text = self.font_small.render(
            f"CPU {cpu_util * 100:.0f}%", True, COLORS["text_dark"]
        )
        util_rect = util_text.get_rect(centerx=x + width / 2, top=y + 25)
        self.screen.blit(util_text, util_rect)

        # Slots area
        slots_top = y + 50
        slots_height = height - 55
        slot_height = slots_height / num_slots

        tasks = node.get("tasks", [])

        # Draw slots (bottom to top)
        for slot_idx in range(num_slots):
            slot_y = slots_top + slots_height - (slot_idx + 1) * slot_height
            slot_rect = pygame.Rect(
                x + self.SLOT_PADDING,
                slot_y + self.SLOT_PADDING,
                width - 2 * self.SLOT_PADDING,
                slot_height - 2 * self.SLOT_PADDING,
            )

            # Determine slot color
            if slot_idx < len(tasks):
                task = tasks[slot_idx]
                color = self._get_task_color(task)
            else:
                color = COLORS["slot_empty"]

            # Draw slot
            pygame.draw.rect(self.screen, color, slot_rect)
            pygame.draw.rect(self.screen, COLORS["node_border"], slot_rect, 1)

    def _get_task_color(self, task: dict[str, Any]) -> tuple[int, int, int]:
        """Get color for a task based on its state.

        Args:
            task: Task data from snapshot

        Returns:
            RGB color tuple
        """
        state = task.get("state", "")
        deadline_status = task.get("deadline_status", "")

        if state == "running":
            if deadline_status == "late":
                return COLORS["slot_running_late"]
            elif deadline_status == "warning":
                return COLORS["slot_running_warning"]
            else:
                return COLORS["slot_running_ok"]
        elif state == "finished":
            return COLORS["slot_finished"]
        elif state in ("rejected", "preempted"):
            return COLORS["slot_rejected"]
        else:
            return COLORS["slot_empty"]

    def _draw_panel(self, snapshot: dict[str, Any]) -> None:
        """Draw KPI panel at bottom of screen.

        Args:
            snapshot: State snapshot from environment
        """
        panel_y = self.height - self.PANEL_HEIGHT
        panel_rect = pygame.Rect(0, panel_y, self.width, self.PANEL_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["panel_bg"], panel_rect)

        metrics = snapshot.get("metrics", {})

        # Reserve space for legend on the right (180px)
        legend_width = 180
        kpi_area_width = self.width - legend_width - 40  # 40 for margins
        
        # KPI positions (5 columns in the left area)
        kpi_width = kpi_area_width // 5
        kpi_y = panel_y + 20

        # Total Reward
        total_reward = metrics.get("total_reward", 0.0)
        reward_color = COLORS["kpi_good"] if total_reward >= 0 else COLORS["kpi_bad"]
        self._draw_kpi(
            "Total Reward",
            f"{total_reward:.1f}",
            20,
            kpi_y,
            reward_color,
        )

        # Tasks Completed
        tasks_completed = metrics.get("tasks_completed", 0)
        self._draw_kpi(
            "Completed",
            str(tasks_completed),
            20 + kpi_width,
            kpi_y,
            COLORS["kpi_neutral"],
        )

        # Average Response Time
        avg_response = metrics.get("avg_response_time", 0.0)
        self._draw_kpi(
            "Avg Response",
            f"{avg_response:.2f}",
            20 + kpi_width * 2,
            kpi_y,
            COLORS["kpi_neutral"],
        )

        # Rejection Rate
        rejection_rate = metrics.get("rejection_rate", 0.0)
        rejection_pct = rejection_rate * 100
        rejection_color = (
            COLORS["kpi_bad"] if rejection_rate > 0.05 else COLORS["kpi_good"]
        )
        self._draw_kpi(
            "Rejection",
            f"{rejection_pct:.1f}%",
            20 + kpi_width * 3,
            kpi_y,
            rejection_color,
        )

        # Throughput
        throughput = metrics.get("throughput", 0.0)
        self._draw_kpi(
            "Throughput",
            f"{throughput:.2f}",
            20 + kpi_width * 4,
            kpi_y,
            COLORS["kpi_neutral"],
        )

    def _draw_kpi(
        self, label: str, value: str, x: float, y: float, color: tuple[int, int, int]
    ) -> None:
        """Draw a single KPI card.

        Args:
            label: KPI label
            value: KPI value
            x: X position
            y: Y position
            color: Text color
        """
        # Label
        label_text = self.font_small.render(label, True, COLORS["text_dark"])
        self.screen.blit(label_text, (x, y))

        # Value
        value_text = self.font_large.render(value, True, color)
        self.screen.blit(value_text, (x, y + 25))

    def _draw_legend(self) -> None:
        """Draw color legend in bottom-right corner."""
        # Position legend in reserved area on the right
        legend_x = self.width - 160
        legend_y = self.height - self.PANEL_HEIGHT + 15

        # Title
        title_text = self.font_small.render("Legend", True, COLORS["text"])
        self.screen.blit(title_text, (legend_x, legend_y))

        legend_items = [
            ("Running (OK)", COLORS["slot_running_ok"]),
            ("Warning", COLORS["slot_running_warning"]),
            ("Late", COLORS["slot_running_late"]),
            ("Empty", COLORS["slot_empty"]),
            ("Rejected", COLORS["slot_rejected"]),
        ]

        y_offset = 22  # Start below title
        for label, color in legend_items:
            # Color square
            square_rect = pygame.Rect(legend_x, legend_y + y_offset, 12, 12)
            pygame.draw.rect(self.screen, color, square_rect)
            pygame.draw.rect(self.screen, COLORS["text_dark"], square_rect, 1)

            # Label
            label_text = self.font_small.render(label, True, COLORS["text_dark"])
            self.screen.blit(label_text, (legend_x + 18, legend_y + y_offset - 2))

            y_offset += 18

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self.closed else "active"
        recording = ", recording" if self._recording else ""
        return f"PygameRackRenderer(size={self.width}x{self.height}, fps={self.fps}, {status}{recording})"
