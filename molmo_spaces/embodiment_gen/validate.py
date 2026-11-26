"""Step 4 -- decide whether an emitted entry is usable, cheapest tier first.

Three tiers so the expensive one only ever sees survivors. A failure is a verdict, not an exception:
generation is a filter, and a seed that cannot be driven is dropped rather than crashing a fleet.

    T1  structural   the spec loads, the model compiles, every referenced name resolves       ms
    T2  reachability MolmoSpaces' OWN IK solves a pose bank to the gripper's leaf frame        ~s
    T3  gauntlet     one scripted pick through the real datagen path                    (external)

T2 is the tier that matters, and it has one hard requirement: it must use the solver the PLANNER
uses. ``check_feasible_ik`` routes through ``kinematics.ik`` (single) and ``parallel_kinematics.ik``
(batch) at plan time, so a screen built on any other solver measures a different robot. This was
measured, not assumed: an earlier screen used a damped-least-squares solver local to ``builder`` and
its score correlated with real planner success at rho = +0.05 over 36 accepted embodiments -- it
gated usefully but ranked nothing, exactly as a mismatched screen would.

T2 also solves on the ASSEMBLED robot rather than the raw MJCF, because the frame it aims at may not
exist before assembly: the gripper move group's ``leaf_frame`` for ``yam_jaw`` is
``gripper/planner_grasp_site``, which the entry's own ``add_site`` operation creates at load time.
Aiming at the compiled model's ``grasp_site`` instead lands 49.7 mm away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# A workspace pose bank, expressed as offsets from where the gripper already rests. Anchoring on the
# rest pose rather than on an absolute height is what makes the bank robot-agnostic: base.pose is at
# the FLOOR, not at the arm root, so "table height" is not z = 0 in that frame -- an earlier version
# assumed it was and put all 30 targets at ankle height, which every arm failed.
POSE_BANK_FORWARD = (-0.15, 0.0, 0.15, 0.30)      # metres along the base's +x, from the rest pose
POSE_BANK_LATERAL = (-0.20, 0.0, 0.20)            # metres along the base's +y
POSE_BANK_VERTICAL = (-0.15, 0.0)                 # metres along the base's +z: at rest, and lower


@dataclass
class Verdict:
    """Why an entry passed or failed, tier by tier, so a rejection is diagnosable."""

    entry: Path
    tier: str = "T1"
    passed: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def line(self) -> str:
        state = "pass" if self.passed else f"FAIL {self.tier}"
        extra = "  ".join(f"{k}={v}" for k, v in self.details.items())
        return f"{self.entry.name}\t{state}\t{self.reason}\t{extra}".rstrip()


def pose_bank(rest: np.ndarray) -> np.ndarray:
    """``(n, 4, 4)`` world targets: a box of positions around ``rest``, holding its orientation.

    Orientation is held rather than forced downward on purpose. The screen asks whether the arm can
    be MOVED around its workspace by the planner's own solver; forcing a tool-down orientation asks
    a second question -- whether the wrist can reach that orientation at all -- and conflates the
    two. The rest pose is itself chosen to present the jaws usefully (see ``builder.rest_pose``), so
    holding it keeps the bank answerable for every gripper.

    The box is in METRES, not scaled by reach: the workspace a task occupies is tabletop-sized and
    does not grow with the arm."""
    poses = []
    for forward in POSE_BANK_FORWARD:
        for lateral in POSE_BANK_LATERAL:
            for vertical in POSE_BANK_VERTICAL:
                pose = rest.copy()
                pose[:3, 3] = rest[:3, 3] + rest[:3, :3] @ np.array([0.0, 0.0, 0.0]) + \
                    np.array([forward, lateral, vertical])
                poses.append(pose)
    return np.stack(poses)


def tier1(entry: Path) -> Verdict:
    """The spec loads, the model compiles, and the move groups agree with the model.

    Structural only: no solver, no physics. It exists to make the later tiers cheap, and to catch
    the failure that used to reach datagen -- a spec naming a frame, camera or actuator the model
    does not declare, which is a hard error at scene setup rather than a degraded robot."""
    from molmo_spaces.robots.configured import load_embodiment_spec

    verdict = Verdict(entry=entry, tier="T1")
    spec_path = entry / "spec.yaml"
    if not spec_path.is_file():
        verdict.reason = "no spec.yaml"
        return verdict
    try:
        spec = load_embodiment_spec(spec_path)
    except Exception as error:                                   # pydantic, yaml, missing refs
        verdict.reason = f"spec did not load: {type(error).__name__}: {error}"[:200]
        return verdict

    mjcf = entry / spec.robot_config.model.mjcf_path
    if not mjcf.is_file():
        verdict.reason = f"spec names {spec.robot_config.model.mjcf_path}, absent from the entry"
        return verdict
    try:
        from molmo_spaces.utils.asset_tokens import from_tokens

        model = mujoco.MjModel.from_xml_string(from_tokens(mjcf.read_text()))
    except Exception as error:
        verdict.reason = f"model did not compile: {error}"[:200]
        return verdict

    verdict.details = {"embodiment": spec.embodiment_id, "version": spec.embodiment_version,
                       "nu": model.nu, "nq": model.nq, "ncam": model.ncam}
    verdict.passed = True
    verdict.reason = "structural"
    return verdict


def tier2(entry: Path, control_profile: str | None = None,
          minimum: float = 0.30) -> Verdict:
    """Fraction of the pose bank MolmoSpaces' own IK can reach, on the assembled robot.

    Mirrors ``BaseObjectManipulationPlannerPolicy.check_feasible_ik`` exactly: same solver, same
    move group (the first gripper group), same unlocked groups, same base pose. What differs from
    the planner is only the target set -- a fixed bank rather than sampled grasps -- because the
    screen has to be independent of which object a task happens to pick."""
    from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
    from molmo_spaces.robots.configured.config import ConfiguredRobotConfig

    verdict = Verdict(entry=entry, tier="T2")
    try:
        robot_config = ConfiguredRobotConfig.from_file(entry / "spec.yaml",
                                                       control_profile=control_profile)
    except Exception as error:
        verdict.reason = f"config did not build: {type(error).__name__}: {error}"[:200]
        return verdict

    try:
        kinematics = MlSpacesKinematics(robot_config)
        scene = mujoco.MjSpec()
        # the prefix must be the config's OWN namespace: a ConfiguredRobot rejects any other, and
        # the view has to be built under the same one or every joint lookup misses
        prefix = robot_config.robot_namespace or ""
        robot_config.robot_cls.add_robot_to_scene(
            robot_config, scene, prefix=prefix, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
        model = scene.compile()
        data = mujoco.MjData(model)
        robot_view = robot_config.robot_view_factory(data, prefix)
        robot_view.set_qpos_dict(robot_config.init_qpos)
        mujoco.mj_forward(model, data)
        group_id = robot_view.get_gripper_movegroup_ids()[0]
        group = robot_view.get_move_group(group_id)
    except Exception as error:
        verdict.reason = f"assembly failed: {type(error).__name__}: {error}"[:200]
        return verdict

    # The rest pose must be STRUCTURAL-contact-free on the assembled robot, checked before the IK
    # bank because it is the cheaper test and its failure is terminal. Structural means any contact
    # not internal to the gripper: gripper-internal contacts are the asset's own business -- the
    # shipped i2rt_yam rests with 6 penetrating jaw contacts, and every yam_jaw graft inherits
    # exactly those -- whereas an arm or pedestal contact means the robot starts embedded in itself.
    #
    # Assembled, not the bare build() model, because the pedestal only exists after assembly. That
    # gap is not hypothetical: gen_2198 shipped with link2 30 mm inside its pedestal, interpenetrated
    # it in 398/400 of sampled joint space (no rest pose could fix it), scored the highest T2 of its
    # cohort because IK knows nothing of collision -- and then failed every planner episode.
    structural = _structural_contacts(model, data, prefix)
    if structural:
        depth, first, second = min(structural)
        verdict.reason = (f"rest pose interpenetrates: {len(structural)} structural contacts, "
                          f"worst {first}~{second} {depth * 1000:.1f} mm")
        verdict.details = {"structural_contacts": len(structural),
                           "worst_mm": round(depth * 1000, 1)}
        return verdict

    # reach is reported as a diagnostic only -- the bank is anchored on the rest pose itself, so
    # nothing is scaled by it; it is here because "which arm is this" is the first thing asked of a
    # rejection
    leaf = np.asarray(group.leaf_frame_to_world, float)
    base = np.asarray(robot_view.base.pose, float)
    reach = float(np.linalg.norm(leaf[:3, 3] - base[:3, 3])) or 0.5

    # q0 is the view's CURRENT qpos, as check_feasible_ik passes it -- not robot_config.init_qpos,
    # which omits the base group and makes the solver reject the call outright ("q0 keys must match
    # move group ids"). A raised exception is a harness fault and is surfaced, never counted as an
    # unreachable pose: swallowing it is what made an earlier version report a truthful-looking 0/30.
    q0 = robot_view.get_qpos_dict()
    unlocked = robot_view.move_group_ids()
    solved = 0
    bank = pose_bank(leaf)
    for pose in bank:
        result = kinematics.ik(group_id, pose, unlocked, q0, robot_view.base.pose)
        solved += result is not None

    share = solved / len(bank)
    verdict.details = {"reach": f"{reach:.3f}m", "solved": f"{solved}/{len(bank)}",
                       "share": f"{share:.3f}"}
    verdict.passed = share >= minimum
    verdict.reason = f"ik reach {share:.3f}" + ("" if verdict.passed else f" below {minimum:.2f}")
    return verdict


def _structural_contacts(model, data, prefix: str) -> list:
    """Penetrating contacts not internal to the gripper: ``(depth, body1, body2)`` per contact.

    A contact is gripper-internal when BOTH bodies are gripper-side (the ``gripper/`` graft prefix,
    plus the camera mount that rides on a gripper body). Everything else -- arm links, the flange,
    the pedestal the assembly adds -- is structure, and a rest pose pressing structure into structure
    is not a pose the planner can start from."""
    def side(geom: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[geom])) or ""
        bare = name[len(prefix):] if name.startswith(prefix) else name
        return "G" if bare.startswith("gripper/") or bare == "wrist_camera_mount" else "S"

    out = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if float(contact.dist) > -1.0e-4:
            continue                                   # grazing, not penetrating
        if side(int(contact.geom1)) == "G" and side(int(contact.geom2)) == "G":
            continue
        names = tuple(
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
             or "")[len(prefix):] for g in (int(contact.geom1), int(contact.geom2)))
        out.append((float(contact.dist), *names))
    return out


def validate(entry: Path, control_profile: str | None = None,
             min_ik_reach: float = 0.30, tiers: tuple = ("T1", "T2")) -> Verdict:
    """Run the tiers in order and return the first failure, or the last pass."""
    verdict = Verdict(entry=entry, tier="T1", passed=True, reason="not run")
    if "T1" in tiers:
        verdict = tier1(entry)
        if not verdict:
            return verdict
    if "T2" in tiers:
        second = tier2(entry, control_profile=control_profile, minimum=min_ik_reach)
        second.details = {**verdict.details, **second.details}
        return second
    return verdict
