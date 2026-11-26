"""
This module contains functionality for filtering and sampling grasps based on heuristics.
"""

import logging
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.utils.profiler_utils import Timer

log = logging.getLogger(__name__)


def get_grasp_collision_body_name(grasp_idx: int) -> str:
    return f"grasp_collision_{grasp_idx}"


def add_grasp_collision_bodies(
    spec: mujoco.MjSpec,
    num_grasps: int,
    grasp_width: float,
    grasp_length: float,
    grasp_height: float,
    grasp_base_pos: np.ndarray,
):
    """Add grasp collision bodies to the scene."""
    for i in range(num_grasps):
        # init grasp bodies in the sky (below the ground causes collision with the floor)
        grasp_body = spec.worldbody.add_body(
            name=get_grasp_collision_body_name(i),
            pos=[0, 0, 10],
            gravcomp=1.0,
        )
        grasp_body.add_freejoint()

        geom_kwargs = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=[0, 0, 1, 1],
            group=3,
            contype=0,
            conaffinity=0b1111,
        )

        base_geom = grasp_body.add_geom(**geom_kwargs)
        base_geom.size[0] = grasp_height / 2
        base_geom.fromto[:3] = np.array([0, -grasp_width / 2, 0]) + grasp_base_pos
        base_geom.fromto[3:] = np.array([0, grasp_width / 2, 0]) + grasp_base_pos

        finger1_geom = grasp_body.add_geom(**geom_kwargs)
        finger1_geom.size[0] = grasp_height / 2
        finger1_geom.fromto[:3] = np.array([0, -grasp_width / 2, 0]) + grasp_base_pos
        finger1_geom.fromto[3:] = np.array([0, -grasp_width / 2, grasp_length]) + grasp_base_pos

        finger2_geom = grasp_body.add_geom(**geom_kwargs)
        finger2_geom.size[0] = grasp_height / 2
        finger2_geom.fromto[:3] = np.array([0, grasp_width / 2, 0]) + grasp_base_pos
        finger2_geom.fromto[3:] = np.array([0, grasp_width / 2, grasp_length]) + grasp_base_pos


def get_noncolliding_grasp_mask(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    grasp_poses_world: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n_grasps = len(grasp_poses_world)
    grasp_bodies = [
        create_mlspaces_body(mj_data, get_grasp_collision_body_name(i)) for i in range(batch_size)
    ]
    start_poses = [body.pose.copy() for body in grasp_bodies]
    grasp_body_ids = set(body.body_id for body in grasp_bodies)

    try:
        colliding_grasp_mask = np.zeros(n_grasps, dtype=bool)
        for i in range(0, n_grasps, batch_size):
            grasp_bid_to_idx = {}
            n_grasps_in_batch = min(batch_size, n_grasps - i)
            for j in range(i, i + n_grasps_in_batch):
                grasp_body = grasp_bodies[j - i]
                grasp_body.pose = grasp_poses_world[j]
                grasp_bid_to_idx[grasp_body.body_id] = j
            for j in range(n_grasps_in_batch, len(grasp_bodies)):
                grasp_bodies[j].pose = start_poses[j]

            mujoco.mj_kinematics(mj_model, mj_data)
            mujoco.mj_collision(mj_model, mj_data)
            for contact in mj_data.contact:
                bid1 = mj_model.geom_bodyid[contact.geom1]
                bid2 = mj_model.geom_bodyid[contact.geom2]
                if bid1 in grasp_body_ids or bid2 in grasp_body_ids:
                    grasp_bid = bid1 if bid1 in grasp_body_ids else bid2
                    other_bid = bid2 if grasp_bid == bid1 else bid1
                    assert other_bid not in grasp_body_ids

                    grasp_idx = grasp_bid_to_idx[grasp_bid]
                    colliding_grasp_mask[grasp_idx] = True

        return ~colliding_grasp_mask
    finally:
        # move the grasp bodies back out of the way
        for body, pose in zip(grasp_bodies, start_poses):
            body.pose = pose
        mujoco.mj_fwdPosition(mj_model, mj_data)


def get_feasible_grasp_idx(
    mg_id: str,
    robot: Robot,
    grasp_poses_world: np.ndarray,
    n_ik_checks: int,
    ik_batch_size: int,
    additional_ik_poses_world: list[np.ndarray] | None = None,
    pose_names: list[str] | None = None,
    diagnostic: dict | None = None,
):
    pose_sets = [grasp_poses_world, *(additional_ik_poses_world or [])]
    if any(len(poses) != len(grasp_poses_world) for poses in pose_sets):
        raise ValueError("All IK pose sets must contain the same number of candidates")
    if pose_names is None:
        pose_names = ["grasp", *[f"pose_{index}" for index in range(1, len(pose_sets))]]
    if len(pose_names) != len(pose_sets):
        raise ValueError("IK pose names must match the number of pose sets")

    n_checks_done = 0
    ret: int | None = None
    n_ik_checks = min(n_ik_checks, len(grasp_poses_world))
    phase_counts = {name: {"feasible": 0, "cumulative": 0} for name in pose_names}

    with Timer() as ik_check_time:
        for i in range(0, n_ik_checks, ik_batch_size):
            real_batch_size = min(ik_batch_size, n_ik_checks - i)
            n_checks_done += real_batch_size
            feasible_mask = np.ones(real_batch_size, dtype=bool)

            for name, poses in zip(pose_names, pose_sets):
                pose_batch = poses[i : i + real_batch_size]
                if real_batch_size < ik_batch_size:
                    pose_batch = np.concatenate(
                        [
                            pose_batch,
                            np.broadcast_to(
                                pose_batch[-1:], (ik_batch_size - real_batch_size, 4, 4)
                            ),
                        ]
                    )

                ik_result = robot.parallel_kinematics.ik(
                    mg_id,
                    pose_batch,
                    None,
                    robot.robot_view.get_qpos_dict(),
                    robot.robot_view.base.pose,
                    rel_to_base=False,
                )
                phase_mask = np.array(
                    [result is not None for result in ik_result[:real_batch_size]]
                )
                feasible_mask &= phase_mask
                phase_counts[name]["feasible"] += int(np.sum(phase_mask))
                phase_counts[name]["cumulative"] += int(np.sum(feasible_mask))

            feasible_indices = np.flatnonzero(feasible_mask)
            if feasible_indices.size > 0:
                ret = i + int(feasible_indices[0])
            if ret is not None:
                break
    if diagnostic is not None:
        diagnostic["ik_checked_count"] = n_checks_done
        diagnostic["phase_feasible_counts"] = phase_counts
        diagnostic["jointly_feasible_count"] = (
            phase_counts[pose_names[-1]]["cumulative"] if pose_names else 0
        )
    log.info(
        f"Feasibility-checked {n_checks_done} grasps across {len(pose_sets)} poses each "
        f"in {ik_check_time.value:.3f}s, found feasible grasp: {ret is not None}"
    )

    return ret


def _best_effort_grasp_pose(
    robot,
    mg_id,
    candidate_ids,
    grasp_poses_world,
    additional_ik_poses_world,
    rot_tol,
    pos_tol,
    max_candidates,
    ik_batch_size,
    diagnostic=None,
):
    """Best-effort grasp when no candidate is strictly IK-feasible.

    The DLS solver will not simply reach the target position with a relaxed orientation (given an
    unreachable target it stalls at a poor position+orientation compromise). So instead, walk the
    cost-sorted collision-free candidates (most preferred first) and, for each, tilt its approach
    away from the requested direction in small steps (up to ``rot_tol``) until every phase
    (grasp + pregrasp + lift) is *fully* IK-feasible. Returns the least-tilted such converged pose,
    or None. The returned pose is exactly reachable, so no execution-threshold relaxation is
    required (mirrors how best-effort-lift reduces height until reachable).

    Solved through the BATCH solver, not one ``kinematics.ik`` call per pose. This sweep is the
    planner's worst case: it fires exactly when strict feasibility found nothing, so it tends to run
    to exhaustion -- and a sequential solve was measured at ~376 ms per FAILING call on a generated
    7-dof arm (failures run to the iteration cap, so the hopeless case is the slow case). The old
    1500-call budget was therefore ~9 minutes of unlogged grinding, which is what actually consumed
    datagen's whole per-attempt budget. The same sweep through ``robot.parallel_kinematics`` in
    ``ik_batch_size`` chunks measures ~3 ms per pose on the CPU warp backend, so even the exhaustive
    no-feasible case resolves in seconds and no call budget is needed at all.

    Per-pose feasibility is independent, so batching preserves the selection exactly: the first
    combination in (candidate cost order, then tilt magnitude, then azimuth) whose grasp AND every
    phase solve. The one semantic change is the solver itself -- the same batch solver
    ``check_feasible_ik`` and ``get_feasible_grasp_idx`` already use, seeded once from the current
    qpos rather than per-combo.
    """
    rv = robot.robot_view
    q0 = rv.get_qpos_dict()
    base_pose = rv.base.pose

    tilt_step = np.radians(10.0)
    # steps of tilt_step up to AND INCLUDING the tolerance cap (so a grasp reachable only right
    # at the cap, e.g. 45 deg, is still tried rather than skipped by arange's exclusive end).
    tilts = [*np.arange(tilt_step, rot_tol - 1e-6, tilt_step), rot_tol]
    azimuths = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)

    combos = []                        # preference order: candidate, then tilt, then azimuth
    for gid in candidate_ids[:max_candidates]:
        gid = int(gid)
        G = grasp_poses_world[gid]
        offsets = [np.asarray(p[gid])[:3, 3] - G[:3, 3] for p in additional_ik_poses_world]
        for tilt in tilts:
            for az in azimuths:
                axis = np.array([np.cos(az), np.sin(az), 0.0])
                Gp = G.copy()
                Gp[:3, :3] = R.from_rotvec(tilt * axis).as_matrix() @ G[:3, :3]
                combos.append((gid, tilt, Gp, offsets))
    if not combos:
        return None

    # The spec's max_iterations is a SCREEN budget (50) -- deliberately below the online solver's
    # 1000, because a uniform 1000 was measured to cost 23x per strict-screen call (2.4 s -> 55 s,
    # ~4 calls per episode, 615 s -> 2935 s end to end) while finding nothing more. This fallback is
    # the exception: it REPLACED a 1000-iteration sequential code path, so it pins that budget
    # explicitly for recall parity by construction. Measured while the budgets differed: at 50 the
    # batch solved a strict subset (10 of sequential's 14 on a 24-pose bank), at 100 it matched that
    # bank exactly, and seeded with sequential's solutions it accepted all of them -- same success
    # criterion, smaller search. A fully-failing 256-batch costs 15.9 s at 1000 iterations, so the
    # exhaustive no-feasible sweep (~2400 poses) is ~190 s measured -- against ~9 minutes for the
    # sequential loop this replaced; 100 (~16 s) is the cheap point if that ever matters.
    batch_max_iter = 1000
    def batch_feasible(poses: np.ndarray) -> np.ndarray:
        """Per-pose solvability, padded to the warmed kernel's batch size exactly as
        get_feasible_grasp_idx pads -- a different batch size would compile a second kernel."""
        mask = np.zeros(len(poses), dtype=bool)
        for start in range(0, len(poses), ik_batch_size):
            chunk = poses[start : start + ik_batch_size]
            real = len(chunk)
            if real < ik_batch_size:
                chunk = np.concatenate(
                    [chunk, np.broadcast_to(chunk[-1:], (ik_batch_size - real, 4, 4))]
                )
            results = robot.parallel_kinematics.ik(
                mg_id, chunk, None, q0, base_pose, rel_to_base=False, max_iter=batch_max_iter
            )
            mask[start : start + real] = [r is not None for r in results[:real]]
        return mask

    with Timer() as sweep_time:
        grasp_ok = batch_feasible(np.stack([pose for _, _, pose, _ in combos]))
        reachable = [combo for combo, ok in zip(combos, grasp_ok) if ok]
        chosen = None
        if reachable and additional_ik_poses_world:
            per_combo = len(additional_ik_poses_world)
            group = max(1, ik_batch_size // per_combo)
            for start in range(0, len(reachable), group):
                block = reachable[start : start + group]
                phases = []
                for _, _, Gp, offsets in block:
                    for off in offsets:
                        P = Gp.copy()
                        P[:3, 3] = Gp[:3, 3] + off
                        phases.append(P)
                phase_ok = batch_feasible(np.stack(phases)).reshape(len(block), per_combo)
                hits = np.flatnonzero(phase_ok.all(axis=1))
                if hits.size:
                    chosen = block[int(hits[0])]
                    break
        elif reachable:
            chosen = reachable[0]

    if chosen is None:
        log.info(
            "Best-effort grasp: swept %d tilted candidates in %.1fs; none fully IK-feasible",
            len(combos),
            sweep_time.value,
        )
        return None
    gid, tilt, Gp, _ = chosen
    if diagnostic is not None:
        diagnostic["orientation_fallback"] = {
            "selected_index": gid,
            "tilt_degrees": float(np.degrees(tilt)),
        }
    log.info(
        "Best-effort grasp: no strictly-feasible grasp; tilted candidate %d "
        "approach by %.0f deg to reach a converged grasp (swept %d poses in %.1fs)",
        gid,
        np.degrees(tilt),
        len(combos),
        sweep_time.value,
    )
    return Gp


def select_grasp_pose(
    env: CPUMujocoEnv,
    grasp_poses_world: np.ndarray,
    object_pose: np.ndarray,
    check_collision: bool,
    n_collision_checks: int,
    collision_batch_size: int,
    check_ik: bool,
    n_ik_checks: int,
    ik_batch_size: int,
    pos_cost_weight: float = 1.0,
    rot_cost_weight: float = 0.01,
    vertical_cost_weight: float = 2.0,
    horizontal_cost_weight: float = 0,
    com_dist_cost_weight: float = 8.0,
    additional_ik_poses_world: list[np.ndarray] | None = None,
    ik_pose_names: list[str] | None = None,
    exclude_object_grasps: list[np.ndarray] | None = None,
    exclude_radius: float = 0.0,
    best_effort_rot_tol: float | None = None,
    best_effort_pos_tol: float = 0.01,
    best_effort_max_candidates: int = 120,
    diagnostic: dict | None = None,
) -> np.ndarray:
    robot = env.current_robot
    if any(poses.shape != grasp_poses_world.shape for poses in (additional_ik_poses_world or [])):
        raise ValueError("Additional IK poses must match the grasp pose array shape")
    num_pose_sets = 1 + len(additional_ik_poses_world or [])
    if ik_pose_names is None:
        ik_pose_names = [
            "grasp",
            *[f"pose_{index}" for index in range(1, num_pose_sets)],
        ]
    if len(ik_pose_names) != num_pose_sets:
        raise ValueError("IK pose names must match the grasp and additional IK pose sets")
    if diagnostic is not None:
        diagnostic.clear()
        diagnostic.update(
            {
                "status": "evaluating",
                "candidate_count": len(grasp_poses_world),
                "ranked_count": 0,
                "collision_free_count": 0,
                "ik_checked_count": 0,
                "phase_feasible_counts": {},
                "jointly_feasible_count": 0,
                "selected_index": None,
                "failure_cause": None,
            }
        )

    gripper_mg_id = robot.robot_view.get_gripper_movegroup_ids()[0]
    tcp_pose = robot.robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
    tcp_pose_inv = np.linalg.inv(tcp_pose)

    dist_tcp = tcp_pose_inv @ grasp_poses_world  # shape (N,4,4)
    dists_tcp_p = np.linalg.norm(dist_tcp[:, :3, 3], axis=1)
    dist_tcp_o = R.from_matrix(dist_tcp[:, :3, :3]).magnitude() * 180 / np.pi

    dists_up = grasp_poses_world[:, 2, 2]  # range = [-1, 1]

    dists_com = np.linalg.norm((np.linalg.inv(object_pose) @ grasp_poses_world)[:, :3, 3], axis=1)

    # --- experimental grasp-selection ablation (env GRASP_EXP; "baseline" leaves this unchanged) ---
    # Object-center proxy: mean of candidate grasp positions (grasps blanket the object surface).
    _exp_mode = os.environ.get("GRASP_EXP", "baseline")
    _obj_centroid = grasp_poses_world[:, :3, 3].mean(axis=0)
    _centrality = np.linalg.norm(grasp_poses_world[:, :3, 3] - _obj_centroid, axis=1)
    if _exp_mode == "centroid":
        # fix #3: anchor the "com" cost at the object centroid instead of the body-frame origin
        dists_com = _centrality

    # Cost for horizontal orientation: 0 = perfectly horizontal (z-axis parallel to XY plane), 1 = vertical
    # Lower cost = more horizontal, so we want to minimize this
    # Use squared term to more strongly penalize vertical orientations
    dists_xy_parallel = np.abs(dists_up) ** 2

    dist_total = (
        pos_cost_weight * dists_tcp_p
        + rot_cost_weight * dist_tcp_o
        + vertical_cost_weight * dists_up
        + horizontal_cost_weight * dists_xy_parallel
        + com_dist_cost_weight * dists_com
    )
    if _exp_mode == "stability":
        # fix #2: add an explicit centrality (moment-arm) penalty on top of the base cost
        dist_total = dist_total + 20.0 * _centrality
    close_grasp_ids = np.argsort(dist_total, kind="stable")  # weight positions and orientations
    close_grasp_ids = close_grasp_ids[:n_collision_checks]
    if diagnostic is not None:
        diagnostic["ranked_count"] = len(close_grasp_ids)

    # filter for noncolliding grasps
    if check_collision:
        with Timer() as collision_check_time:
            noncolliding_grasp_mask = get_noncolliding_grasp_mask(
                env.current_model,
                env.current_data,
                grasp_poses_world[close_grasp_ids],
                collision_batch_size,
            )

        log.info(
            f"Collision-checked {len(close_grasp_ids)} grasps in {collision_check_time.value:.3f}s, found {np.sum(noncolliding_grasp_mask)} non-colliding grasps"
        )
    else:
        noncolliding_grasp_mask = np.ones(len(close_grasp_ids), dtype=bool)

    noncolliding_close_grasp_ids = close_grasp_ids[noncolliding_grasp_mask]
    if diagnostic is not None:
        diagnostic["collision_free_count"] = int(len(noncolliding_close_grasp_ids))
    # colliding_close_grasp_ids = close_grasp_ids[~noncolliding_grasp_mask]

    # filter for feasibility/reachability
    if check_ik:
        grasp_idx: int | None = None

        # On retry, prefer grasp sites not attempted yet this episode (candidates whose contact
        # point is farther than exclude_radius from every previously-tried grasp), and only fall
        # back to already-tried sites if none of the fresh ones are reachable ("otherwise any grasp").
        id_groups: list[np.ndarray] = [noncolliding_close_grasp_ids]
        if exclude_object_grasps and exclude_radius > 0 and noncolliding_close_grasp_ids.size > 0:
            cand_obj_pos = (
                np.linalg.inv(object_pose) @ grasp_poses_world[noncolliding_close_grasp_ids]
            )[:, :3, 3]
            excl_pos = np.stack([np.asarray(g)[:3, 3] for g in exclude_object_grasps])
            nearest = np.min(
                np.linalg.norm(cand_obj_pos[:, None, :] - excl_pos[None, :, :], axis=2), axis=1
            )
            fresh_mask = nearest > exclude_radius
            if fresh_mask.any() and not fresh_mask.all():
                id_groups = [
                    noncolliding_close_grasp_ids[fresh_mask],
                    noncolliding_close_grasp_ids[~fresh_mask],
                ]

        if _exp_mode == "topk" and noncolliding_close_grasp_ids.size > 0:
            # fix #4: keep the reachability-ranked shortlist, then choose the most central of it
            k = min(20, noncolliding_close_grasp_ids.size)
            shortlist = noncolliding_close_grasp_ids[:k]
            id_groups = [shortlist[np.argsort(_centrality[shortlist])]]

        for group_ids in id_groups:
            if group_ids.size == 0:
                continue
            group_grasp_idx = get_feasible_grasp_idx(
                gripper_mg_id,
                robot,
                grasp_poses_world[group_ids],
                n_ik_checks,
                ik_batch_size,
                additional_ik_poses_world=[
                    poses[group_ids] for poses in (additional_ik_poses_world or [])
                ],
                pose_names=ik_pose_names,
                diagnostic=diagnostic,
            )
            if group_grasp_idx is not None:
                grasp_idx = int(group_ids[group_grasp_idx])
                break
    elif noncolliding_close_grasp_ids.size > 0:
        grasp_idx = int(noncolliding_close_grasp_ids[0])
    else:
        grasp_idx = None

    if (
        grasp_idx is None
        and check_ik
        and best_effort_rot_tol is not None
        and noncolliding_close_grasp_ids.size > 0
    ):
        # No candidate is strictly IK-feasible. Fall back to the most-preferred grasp whose
        # approach can be tilted (within tolerance) to a fully reachable orientation.
        best_effort_pose = _best_effort_grasp_pose(
            robot,
            gripper_mg_id,
            noncolliding_close_grasp_ids,
            grasp_poses_world,
            additional_ik_poses_world or [],
            best_effort_rot_tol,
            best_effort_pos_tol,
            best_effort_max_candidates,
            ik_batch_size,
            diagnostic,
        )
        if best_effort_pose is not None:
            if diagnostic is not None:
                diagnostic["status"] = "selected_best_effort"
                diagnostic["selected_index"] = None
            return best_effort_pose

    if grasp_idx is None:
        if diagnostic is not None:
            diagnostic["status"] = "failed"
            if len(noncolliding_close_grasp_ids) == 0:
                diagnostic["failure_phase"] = "collision"
                diagnostic["failure_cause"] = (
                    "No ranked candidate passed grasp-volume collision filtering"
                )
            elif check_ik:
                phase_counts = diagnostic.get("phase_feasible_counts", {})
                failure_phase = next(
                    (
                        phase
                        for phase in ik_pose_names
                        if phase_counts.get(phase, {}).get("cumulative", 0) == 0
                    ),
                    ik_pose_names[-1],
                )
                diagnostic["failure_phase"] = failure_phase
                if phase_counts.get(failure_phase, {}).get("feasible", 0) == 0:
                    diagnostic["failure_cause"] = (
                        f"Every checked candidate failed {failure_phase} IK"
                    )
                else:
                    diagnostic["failure_cause"] = (
                        f"Candidates passed {failure_phase} IK individually, but none also "
                        "passed every preceding phase"
                    )
            else:
                diagnostic["failure_cause"] = "No candidate remained after filtering"
        raise ValueError("No feasible grasp found")

    if diagnostic is not None:
        diagnostic["status"] = "selected" if check_ik else "selected_without_ik_check"
        diagnostic["selected_index"] = int(grasp_idx)
    return grasp_poses_world[grasp_idx]
