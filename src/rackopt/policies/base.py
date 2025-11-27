"""Base class for scheduling policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rackopt.core.action import Action
from rackopt.core.observation import Observation


class BasePolicy(ABC):
    """Abstract base class for scheduling policies.

    All scheduling algorithms should inherit from this class and implement
    the select_action method.
    """

    def __init__(self, name: str | None = None):
        """Initialize policy.

        Args:
            name: Human-readable name for the policy
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select_action(self, observation: Observation) -> Action:
        """Select scheduling action based on current observation.

        Args:
            observation: Current cluster state

        Returns:
            Action containing scheduling decisions
        """
        pass

    def reset(self) -> None:
        """Reset policy state (if stateful).

        Override this method if your policy maintains internal state
        that should be reset between episodes.
        """
        pass

    def __repr__(self) -> str:
        """String representation of policy."""
        return f"{self.__class__.__name__}(name='{self.name}')"
