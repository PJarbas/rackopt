"""Visualization modules (optional pygame dependency)."""

# Pygame imports are lazy-loaded to avoid import errors if pygame not installed
__all__ = [
    "PygameRackRenderer",
]

def __getattr__(name: str):
    """Lazy import pygame renderer."""
    if name == "PygameRackRenderer":
        try:
            from rackopt.viz.pygame_rack import PygameRackRenderer
            return PygameRackRenderer
        except ImportError as e:
            raise ImportError(
                "Pygame visualization requires pygame. "
                "Install with: pip install rackopt[viz]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
