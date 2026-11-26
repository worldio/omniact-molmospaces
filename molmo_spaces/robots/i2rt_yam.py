"""i2rt YAM robot implementation for the framework."""

from typing import TYPE_CHECKING, cast

from mujoco import MjData, MjSpec, mjtGeom

from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.i2rt_yam_view import (
    PLANNER_GRASP_SITE_NAME,
    PLANNER_GRASP_SITE_POS,
    PLANNER_GRASP_SITE_XYAXES,
)

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import I2rtYamRobotConfig


class I2rtYamRobot(Robot):
    """i2rt YAM 6-DOF arm robot implementation."""

    def __init__(
        self,
        mj_data: MjData,
        config: "MlSpacesExpConfig",
    ) -> None:
        super().__init__(mj_data, config)
        self._robot_view = config.robot_config.robot_view_factory(
            mj_data, config.robot_config.robot_namespace
        )
        self._kinematics = MlSpacesKinematics(config.robot_config)

        self._parallel_kinematics = SimpleWarpKinematics(config.robot_config)

        arm_controller_cls = (
            JointPosController
            if config.robot_config.command_mode == {}
            or config.robot_config.command_mode.get("arm") == "joint_position"
            else JointRelPosController
        )
        self._controllers = {
            "arm": arm_controller_cls(self._robot_view.get_move_group("arm")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

    @property
    def namespace(self):
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        return self._parallel_kinematics

    @property
    def controllers(self):
        return self._controllers

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [
            TCPPoseSensor(uuid="tcp_pose"),
        ]

    def get_arm_move_group_ids(self) -> list[str]:
        """YAM has a single arm move group."""
        return ["arm"]

    def reset(self) -> None:
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = default_pos

    @staticmethod
    def robot_model_root_name() -> str:
        return "arm"

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "I2rtYamRobotConfig") -> None:
        super().apply_control_overrides(spec, robot_config)
        if not hasattr(robot_config, "gripper_effort_limit"):
            return

        gripper_actuator = spec.actuator(f"{robot_config.robot_namespace}gripper")
        if gripper_actuator is None:
            raise ValueError("YAM gripper actuator not found")

        # The hardware SDK regulates force after the gripper stalls. MuJoCo uses a
        # damped position servo, so tune it to reach an analogous effort cap stably.
        gain = robot_config.gripper_position_gain
        velocity_gain = robot_config.gripper_velocity_gain
        effort_limit = robot_config.gripper_effort_limit
        gripper_actuator.gainprm[0] = gain
        gripper_actuator.biasprm[1] = -gain
        gripper_actuator.biasprm[2] = -velocity_gain
        gripper_actuator.forcelimited = 1
        gripper_actuator.forcerange[:] = [-effort_limit, effort_limit]

    @classmethod
    def _load_robot_spec(
        cls, robot_config: "I2rtYamRobotConfig", strip_meshes: bool = False
    ) -> MjSpec:
        robot_spec = super()._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        link_6 = robot_spec.body("link_6")
        if link_6 is None:
            raise ValueError("Robot body 'link_6' not found")
        link_6.add_site(
            name=PLANNER_GRASP_SITE_NAME,
            pos=PLANNER_GRASP_SITE_POS,
            xyaxes=PLANNER_GRASP_SITE_XYAXES,
            group=4,
            rgba=[0, 1, 1, 1],
        )
        return robot_spec

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "I2rtYamRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("I2rtYamRobotConfig", robot_config)
        add_base = robot_config.base_size is not None
        pos = pos + [0.0] if len(pos) == 2 else pos

        # Create a mocap body to control the robot base pose
        robot_body = spec.worldbody.add_body(
            name=f"{prefix}base",
            pos=pos,
            quat=quat,
            mocap=True,
        )

        if add_base:
            base_height = robot_config.base_size[2]

            # Create a base material (plain dark wood color)
            material_name = f"{prefix}robot_base_material"
            spec.add_material(name=material_name, rgba=[0.3, 0.2, 0.1, 1.0])

            # Add base geometry (platform)
            robot_body.add_geom(
                type=mjtGeom.mjGEOM_BOX,
                size=[x / 2 for x in robot_config.base_size],
                pos=[0, 0, base_height / 2],
                material=material_name,
                group=0,  # Visual group
            )
            attach_frame = robot_body.add_frame(pos=[0, 0, base_height])
        else:
            attach_frame = robot_body.add_frame()

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root_name = cls.robot_model_root_name()
        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot {robot_root_name=} not found in {robot_spec}")
        attach_frame.attach_body(robot_root, prefix, "")
