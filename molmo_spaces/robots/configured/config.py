"""BaseRobotConfig compatibility adapter for configured embodiments."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

from mujoco import MjData
from pydantic import Field

from molmo_spaces.configs.robot_configs import ActionNoiseConfig, BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.configured.loader import load_embodiment_spec
from molmo_spaces.robots.configured.robot_view import ConfiguredRobotView
from molmo_spaces.robots.configured.spec import EmbodimentSpec
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory


class ConfiguredRobotConfig(BaseRobotConfig):
    """Expose an :class:`EmbodimentSpec` through existing runtime hooks."""

    embodiment: EmbodimentSpec
    control_profile: str | None = None

    robot_cls: type[Robot] | None = None
    robot_factory: Callable[[MjData, Any], Robot] | None = None
    robot_view_factory: RobotViewFactory | None = None
    robot_namespace: str = ""
    command_mode: dict[str, str] = Field(default_factory=dict)
    init_qpos: dict[str, list[float]] = Field(default_factory=dict)
    init_qpos_noise_range: dict[str, list[float]] | None = None
    name: str | None = None
    robot_xml_path: Path = Path(".")

    def model_post_init(self, context) -> None:
        from molmo_spaces.robots.configured.robot import ConfiguredRobot

        embodiment_config = self.embodiment.robot_config
        model = embodiment_config.model
        selected_profile = self.control_profile or embodiment_config.default_control_profile
        if selected_profile not in embodiment_config.control_profiles:
            raise ValueError(f"Unknown configured control profile {selected_profile!r}")
        object.__setattr__(self, "control_profile", selected_profile)
        profile = embodiment_config.control_profiles[selected_profile]

        object.__setattr__(self, "robot_cls", ConfiguredRobot)
        object.__setattr__(self, "robot_factory", ConfiguredRobot)
        object.__setattr__(
            self,
            "robot_view_factory",
            partial(
                ConfiguredRobotView,
                view_spec=self.embodiment.robot_view,
                robot_name=(
                    self.embodiment.robot_view.runtime_name
                    or model.resource_key
                    or self.embodiment.embodiment_id
                ),
                grasp_profiles=self.embodiment.grasp_profiles,
            ),
        )
        object.__setattr__(self, "robot_namespace", embodiment_config.namespace)
        object.__setattr__(self, "init_qpos", deepcopy(embodiment_config.initial_qpos))
        object.__setattr__(
            self,
            "init_qpos_noise_range",
            deepcopy(embodiment_config.initial_qpos_noise),
        )
        object.__setattr__(
            self,
            "command_mode",
            {
                group_id: self.embodiment.robot.controllers[controller_id].kind
                for group_id, controller_id in profile.controller_bindings.items()
            },
        )
        object.__setattr__(
            self,
            "base_placement_forward_offset",
            embodiment_config.placement.forward_offset,
        )
        object.__setattr__(
            self,
            "base_placement_forward_offset_step",
            embodiment_config.placement.forward_offset_step,
        )
        object.__setattr__(self, "gravcomp", embodiment_config.gravity_compensation.enabled)

        if model.source == "builtin":
            object.__setattr__(self, "name", model.resource_key)
            object.__setattr__(self, "robot_dir", None)
            object.__setattr__(self, "robot_xml_path", model.mjcf_path)
        else:
            xml_path = model.mjcf_path
            if not xml_path.is_absolute():
                assert model.root_path is not None
                xml_path = model.root_path / xml_path
            object.__setattr__(self, "name", self.embodiment.embodiment_id)
            object.__setattr__(self, "robot_dir", xml_path.parent)
            object.__setattr__(self, "robot_xml_path", Path(xml_path.name))

        noise = embodiment_config.action_noise
        if noise is None:
            action_noise = ActionNoiseConfig(enabled=False)
        else:
            action_noise = ActionNoiseConfig(
                enabled=noise.enabled,
                action_scale_factor=noise.action_scale_factor,
                rotation_noise_scale=noise.rotation_noise_scale,
                max_tcp_position_noise=noise.max_tcp_position_noise,
                max_tcp_rotation_noise=noise.max_tcp_rotation_noise,
                base_action_scale_factor=noise.base_action_scale_factor,
                max_base_position_noise=noise.max_base_position_noise,
                max_base_rotation_noise=noise.max_base_rotation_noise,
            )
        object.__setattr__(self, "action_noise_config", action_noise)
        super().model_post_init(context)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        control_profile: str | None = None,
    ) -> ConfiguredRobotConfig:
        """Load a YAML embodiment and expose it as a runtime robot config."""
        return cls(
            embodiment=load_embodiment_spec(path),
            control_profile=control_profile,
        )
