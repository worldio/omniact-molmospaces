"""Declarative robot embodiment configuration."""

from molmo_spaces.robots.configured.config import ConfiguredRobotConfig
from molmo_spaces.robots.configured.loader import load_embodiment_spec
from molmo_spaces.robots.configured.robot import ConfiguredRobot
from molmo_spaces.robots.configured.robot_view import ConfiguredRobotView
from molmo_spaces.robots.configured.spec import EmbodimentSpec

__all__ = [
    "ConfiguredRobot",
    "ConfiguredRobotConfig",
    "ConfiguredRobotView",
    "EmbodimentSpec",
    "load_embodiment_spec",
]
