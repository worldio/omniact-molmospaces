"""Kinematics builders for configured robots."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import (
    DummyParallelKinematics,
)
from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.configured.names import resolve_frame_id
from molmo_spaces.robots.configured.spec import (
    DampedLeastSquaresKinematicsSpec,
    FloatingFrameKinematicsSpec,
    KinematicsSpec,
    NoKinematicsSpec,
)
from molmo_spaces.utils.linalg_utils import global_to_relative_transform
from molmo_spaces.utils.mj_model_and_data_utils import body_pose, site_pose

if TYPE_CHECKING:
    from molmo_spaces.robots.configured.config import ConfiguredRobotConfig


def _active_ik_groups(spec, robot_view, requested: list[str] | None) -> list[str]:
    if requested is None:
        return list(spec.optimizable_groups)
    unknown = set(requested) - set(robot_view.move_group_ids())
    if unknown:
        raise ValueError(f"IK references unknown move groups: {sorted(unknown)}")
    active = [group_id for group_id in requested if group_id in spec.optimizable_groups]
    if not active:
        raise ValueError("IK request does not unlock any configured optimizable group")
    return active


class ConfiguredDampedLeastSquaresKinematics(MlSpacesKinematics):
    """Use configured defaults with the existing online MuJoCo IK solver."""

    def __init__(
        self,
        robot_config: ConfiguredRobotConfig,
        spec: DampedLeastSquaresKinematicsSpec,
    ) -> None:
        self._configured_spec = spec
        super().__init__(robot_config)

    def ik(
        self,
        move_group_id: str,
        pose: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0: dict[str, np.ndarray],
        base_pose: np.ndarray,
        rel_to_base: bool = False,
        eps: float | None = None,
        max_iter: int | None = None,
        damping: float | None = None,
        dt: float | None = None,
    ):
        unlocked_move_group_ids = _active_ik_groups(
            self._configured_spec,
            self._robot_view,
            unlocked_move_group_ids,
        )
        return super().ik(
            move_group_id,
            pose,
            unlocked_move_group_ids,
            q0,
            base_pose,
            rel_to_base=rel_to_base,
            eps=(
                self._configured_spec.online_success_tolerance
                or self._configured_spec.success_tolerance
                if eps is None
                else eps
            ),
            max_iter=(
                self._configured_spec.online_max_iterations or self._configured_spec.max_iterations
                if max_iter is None
                else max_iter
            ),
            damping=self._configured_spec.damping if damping is None else damping,
            dt=self._configured_spec.integration_step if dt is None else dt,
        )


class ConfiguredWarpKinematics(SimpleWarpKinematics):
    """Use configured defaults with the existing Warp batch IK solver."""

    def __init__(
        self,
        robot_config: ConfiguredRobotConfig,
        spec: DampedLeastSquaresKinematicsSpec,
    ) -> None:
        self._configured_spec = spec
        super().__init__(robot_config)

    def ik(
        self,
        move_group_id: str,
        poses: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
        converge_eps: float | None = None,
        success_eps: float | None = None,
        max_iter: int | None = None,
        damping: float | None = None,
        dt: float | None = None,
    ):
        unlocked_move_group_ids = _active_ik_groups(
            self._configured_spec,
            self._robot_view,
            unlocked_move_group_ids,
        )
        return super().ik(
            move_group_id,
            poses,
            unlocked_move_group_ids,
            q0_dicts,
            base_poses,
            rel_to_base=rel_to_base,
            converge_eps=(
                self._configured_spec.convergence_tolerance
                if converge_eps is None
                else converge_eps
            ),
            success_eps=(
                self._configured_spec.success_tolerance if success_eps is None else success_eps
            ),
            max_iter=self._configured_spec.max_iterations if max_iter is None else max_iter,
            damping=self._configured_spec.damping if damping is None else damping,
            dt=self._configured_spec.integration_step if dt is None else dt,
        )


class ConfiguredFloatingFrameKinematics(MlSpacesKinematics):
    """Solve a floating frame by directly placing a seven-DoF base group."""

    def __init__(
        self,
        robot_config: ConfiguredRobotConfig,
        spec: FloatingFrameKinematicsSpec,
    ) -> None:
        self._configured_spec = spec
        super().__init__(robot_config)
        self._tcp_frame_id = resolve_frame_id(
            self._mj_model, spec.tcp_frame, robot_config.robot_namespace
        )

    def ik(
        self,
        move_group_id: str,
        pose: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0: dict[str, np.ndarray],
        base_pose: np.ndarray,
        rel_to_base: bool = False,
        **kwargs,
    ):
        base_group_id = self._configured_spec.base_group
        if unlocked_move_group_ids is not None and base_group_id not in unlocked_move_group_ids:
            return None
        base_group = self._robot_view.get_move_group(base_group_id)
        if base_group.pos_dim != 7:
            raise ValueError("Floating-frame kinematics requires a seven-position base group")

        self._robot_view.base.pose = base_pose
        self._robot_view.set_qpos_dict(q0)
        if rel_to_base:
            pose = base_pose @ pose
        if move_group_id == base_group_id:
            desired_base = pose
        else:
            mujoco.mj_fwdPosition(self._mj_model, self._mj_data)
            if self._configured_spec.tcp_frame.kind == "site":
                tcp_world = site_pose(self._mj_data, self._tcp_frame_id)
            else:
                tcp_world = body_pose(self._mj_data, self._tcp_frame_id)
            tcp_to_base = global_to_relative_transform(tcp_world, self._robot_view.base.pose)
            desired_base = pose @ np.linalg.inv(tcp_to_base)
        result = deepcopy(q0)
        result[base_group_id] = np.concatenate(
            [
                desired_base[:3, 3],
                R.from_matrix(desired_base[:3, :3]).as_quat(scalar_first=True),
            ]
        )
        return result


class UnavailableKinematics:
    """Explicit no-kinematics implementation for direct-control embodiments."""

    def __getattr__(self, name: str):
        raise NotImplementedError(f"This embodiment declares no kinematics; {name} is unavailable")


class UnavailableParallelKinematics(ParallelKinematics):
    def warmup_ik(self, batch_size: int):
        raise NotImplementedError("This embodiment declares no kinematics")

    def fk(self, qpos_dicts, base_poses, rel_to_base: bool = False):
        raise NotImplementedError("This embodiment declares no kinematics")

    def ik(
        self,
        move_group_id,
        poses,
        unlocked_move_group_ids,
        q0_dicts,
        base_poses,
        rel_to_base: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("This embodiment declares no kinematics")


def build_kinematics(
    robot_config: ConfiguredRobotConfig,
    spec: KinematicsSpec,
) -> tuple[MlSpacesKinematics | UnavailableKinematics, ParallelKinematics]:
    """Build matching online and batch kinematics implementations."""
    if isinstance(spec, DampedLeastSquaresKinematicsSpec):
        online = ConfiguredDampedLeastSquaresKinematics(robot_config, spec)
        parallel: ParallelKinematics
        if spec.batch_backend == "warp":
            parallel = ConfiguredWarpKinematics(robot_config, spec)
        elif spec.batch_backend == "sequential":
            parallel = DummyParallelKinematics(robot_config, online)
        else:
            parallel = UnavailableParallelKinematics(robot_config)
        return online, parallel
    if isinstance(spec, FloatingFrameKinematicsSpec):
        online = ConfiguredFloatingFrameKinematics(robot_config, spec)
        return online, DummyParallelKinematics(robot_config, online)
    if isinstance(spec, NoKinematicsSpec):
        unavailable = UnavailableKinematics()
        return unavailable, UnavailableParallelKinematics(robot_config)
    raise TypeError(f"Unsupported kinematics kind: {spec.kind!r}")
