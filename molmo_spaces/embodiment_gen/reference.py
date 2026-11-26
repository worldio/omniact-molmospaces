"""The parameters that ARE the shipped robots -- the witness that they are in the distribution.

``sampling.py`` decides which values a seed draws; ``builder.py`` decides how values become a
model. Because neither knows the other, a hand-derived ``EmbodimentParams`` builds exactly like a
sampled one. That is what makes "is franka_droid in the distribution?" answerable instead of
rhetorical: exhibit its parameters, check they are admissible, and build them.

    measure(<shipped MJCF>)  ->  EmbodimentParams  ->  builder.build()  ->  model.xml

Every field here is READ OFF the shipped model. Nothing is a literal, and this module is used only
by tests and inspection -- the generator itself never imports it, so no real robot's numbers can
leak into the sampled population.

    python -m molmo_spaces.embodiment_gen.reference        # print both witnesses
"""

from __future__ import annotations

import mujoco
import numpy as np

from molmo_spaces.molmo_spaces_constants import get_robot_path

from .constants import GENERATOR_VERSION
from .grippers import load_catalog, source_spec
from .skeleton import mat_to_quat, quat_to_mat
from .constants import GRIPPER_PREFIX
from .geom_pool import (
    BASE_INDEX,
    SOURCES,
    harvest,
    harvest_base,
    harvest_cameras,
    harvest_sites,
)
from .params import (
    ArmParams,
    CameraParams,
    SiteParams,
    DressingParams,
    EmbodimentParams,
    GeomParams,
    InertialParams,
    JointParams,
    LinkParams,
    SegmentParams,
)

# Which catalog gripper each shipped robot wears. The one axis that is EQUALITY rather than
# approximation: the graft attaches the identical asset the real robot uses.
GRIPPER_OF = {"franka_droid": "robotiq_2f85", "i2rt_yam": "yam_jaw"}

# Values not encoded in the arm's MJCF (the pedestal is a scene-scoped operation, and colours are
# unused because every reference link wears real geometry). Chosen, not measured.
_PEDESTAL_HEIGHT = 0.70
_PEDESTAL_RADIUS = 0.15


def _arm_joints(model, exclude) -> list:
    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
    return [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            and not any(s in name(j) for s in exclude)]


def _pose(position, quaternion):
    return np.asarray(position, float), quat_to_mat(quaternion)


def _compose(a, b):
    """(pos, R) composition: a then b."""
    return a[0] + a[1] @ b[0], a[1] @ b[1]


def _invert(a):
    return -a[1].T @ a[0], a[1].T


def measure_mount(source: str, gripper: str) -> tuple:
    """The flange -> gripper transform that puts the graft where the shipped robot has it.

    Not zero, and not directly readable off either model. The built chain is

        flange  o  mount_frame  o  source_root_pose  =  gripper root

    where ``source_root_pose`` is the graft source's own root body pose, which ``attach_body``
    carries along. So ``mount = shipped_relative o source_root_pose^-1``.

    The trap this closes: franka_droid was itself assembled by attaching its gripper at a FRAME, so
    the AUTHORED ``gripper/base`` pos reads (0, 0, 0) while the compiled placement carries the
    frame's pose -- taking the authored value put the gripper 0.107 m and 90 degrees out.
    """
    config = SOURCES[source]
    model = mujoco.MjModel.from_xml_path(str(get_robot_path(source) / config["mjcf"]))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    asset = load_catalog()[gripper]
    last_arm_body = harvest(source)[-1].body
    root_body = asset.graft_root if asset.claims_flange_link else f"{GRIPPER_PREFIX}{asset.graft_root}"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body) < 0:
        root_body = asset.graft_root
    ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
           for name in (last_arm_body, root_body)}
    world = {name: (np.array(data.xpos[i], float),
                    np.array(data.xmat[i], float).reshape(3, 3)) for name, i in ids.items()}
    relative = _compose(_invert(world[last_arm_body]), world[root_body])

    # `graft` cancels the source root's own pose, so the mount IS the relative transform.
    return tuple(float(v) for v in relative[0]), mat_to_quat(relative[1])


def params_of(source: str) -> EmbodimentParams:
    """The witness ``EmbodimentParams`` for one shipped robot, read off its MJCF."""
    if source not in SOURCES:
        raise KeyError(f"{source!r} is not a reference source; known: {sorted(SOURCES)}")
    spec = SOURCES[source]
    model = mujoco.MjModel.from_xml_path(str(get_robot_path(source) / spec["mjcf"]))
    joints = _arm_joints(model, spec["exclude"])
    actuator_of = {int(model.actuator_trnid[a][0]): a for a in range(model.nu)
                   if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
    pool = harvest(source)

    segments, joint_params, links = [], [], []
    for index, joint in enumerate(joints):
        body = int(model.jnt_bodyid[joint])
        segments.append(SegmentParams(
            quat=tuple(float(v) for v in model.body_quat[body]),
            offset=tuple(float(v) for v in model.body_pos[body]),
        ))
        actuator = actuator_of.get(joint)
        dof = int(model.jnt_dofadr[joint])
        joint_params.append(JointParams(
            kind="hinge",
            range=tuple(float(v) for v in model.jnt_range[joint]),
            kp=float(model.actuator_gainprm[actuator][0]) if actuator is not None else 0.0,
            kv=float(-model.actuator_biasprm[actuator][2]) if actuator is not None else 0.0,
            # 0 means unlimited, which is how FR3 leaves its arm actuators
            force=(float(abs(model.actuator_forcerange[actuator][1]))
                   if actuator is not None else 0.0),
            armature=float(model.dof_armature[dof]),
            damping=float(model.dof_damping[dof]),
            frictionloss=float(model.dof_frictionloss[dof]),
        ))
        # A COHERENT pool draw: this source, this link index. Exactly the draw sampling.py makes
        # with probability 0.5 -- the witness uses the ordinary mechanism, not a private path.
        entry = pool[index]
        links.append(LinkParams(
            # "inherit": the link wears the source model's own materials, which is the
            # only route to FR3's white/black/button set. An ordinary sampler mode.
            geom=GeomParams(mode="pool", source=source, link_index=index,
                            material_mode="inherit"),
            inertial=InertialParams(mode="explicit", density=0.0, mass=entry.mass,
                                    inertia=entry.inertia, com=entry.com,
                                    gravcomp=float(model.body_gravcomp[body])),
        ))

    # The base carries the mounting structure -- FR3's fr3_link0 (8 geoms, 2.82 kg), YAM's motor
    # housing and plate. It has no joint, so it needs its own slot or the robot is built with an
    # invisible, massless base.
    mount_pos, mount_quat = measure_mount(source, GRIPPER_OF[source])
    # Cameras the shipped robot declares on its own arm. Gripper-side ones are excluded by the
    # harvest because they ride along with the graft; re-emitting them would duplicate.
    last_arm = harvest(source)[-1].body
    cameras = tuple(CameraParams(
        name=camera.name,
        mount="flange" if camera.parent == last_arm else "base",
        body_pos=camera.body_pos, body_quat=camera.body_quat,
        pos=camera.pos, quat=camera.quat, fovy=camera.fovy, resolution=camera.resolution,
    ) for camera in harvest_cameras(source))
    # Sites the shipped robot declares on its ARM. Those on the claimed flange link (yam's
    # tcp_site and grasp_site live on link_6) arrive with the graft, so re-emitting them would
    # duplicate; FR3's attachment_site is on its own link7 and must be emitted here.
    claimed = load_catalog()[GRIPPER_OF[source]].claims_flange_link
    sites = tuple(SiteParams(name=site.name, mount="flange", pos=site.pos, quat=site.quat,
                             group=site.group)
                  for site in harvest_sites(source)
                  if not (claimed and site.parent == last_arm))
    base_entry = harvest_base(source)
    base = LinkParams(
        geom=GeomParams(mode="pool", source=source, link_index=BASE_INDEX,
                        material_mode="inherit"),
        inertial=InertialParams(mode="explicit", density=0.0, mass=base_entry.mass,
                                inertia=base_entry.inertia, com=base_entry.com),
    )
    arm = ArmParams(
        grammar="reference",
        segments=tuple(segments),
        joints=tuple(joint_params),
        links=tuple(links),
        base=base,
        cameras=cameras,
        sites=sites,
        mount_pos=mount_pos,
        mount_quat=mount_quat,
        pedestal_height=_PEDESTAL_HEIGHT,
        pedestal_radius=_PEDESTAL_RADIUS,
    )
    return EmbodimentParams(
        seed=-1, version=GENERATOR_VERSION, arm=arm, gripper=GRIPPER_OF[source],
        dressing=DressingParams(palette={"base": [0.5, 0.5, 0.5, 1.0],
                                         "accent": [0.5, 0.5, 0.5, 1.0],
                                         "joint": [0.3, 0.3, 0.3, 1.0]},
                                radius=0.04, taper=0.7, accent_period=2),
    )


def describe(source: str) -> str:
    """The witness, as a table -- what to read when asking 'which parameters give me an FR3?'."""
    params = params_of(source)
    arm = params.arm
    rows = [f"{source}  ->  EmbodimentParams(dof={arm.dof}, gripper={params.gripper!r})",
            f"  mount_pos={arm.mount_pos}  mount_quat={arm.mount_quat}",
            "",
            f"  {'i':>2s} {'segment.quat (wxyz)':>34s} {'segment.offset (m)':>28s}",
            ]
    for index, segment in enumerate(arm.segments, start=1):
        rows.append(f"  {index:2d} {str(np.round(segment.quat, 5)):>34s} "
                    f"{str(np.round(segment.offset, 5)):>28s}")
    rows += ["",
             f"  {'i':>2s} {'kind':>6s} {'range':>18s} {'kp':>8s} {'kv':>7s} {'force':>6s} "
             f"{'armat':>7s} {'damp':>5s} {'fric':>5s}"]
    for index, joint in enumerate(arm.joints, start=1):
        rows.append(f"  {index:2d} {joint.kind:>6s} "
                    f"[{joint.range[0]:7.3f},{joint.range[1]:7.3f}] {joint.kp:8.1f} "
                    f"{joint.kv:7.1f} {joint.force:6.1f} {joint.armature:7.4f} "
                    f"{joint.damping:5.2f} {joint.frictionloss:5.2f}")
    rows += ["",
             f"  {'i':>2s} {'geom.mode':>10s} {'source':>14s} {'idx':>4s} {'inertial':>9s} "
             f"{'mass kg':>8s} {'inertia diag':>26s}"]
    for index, link in enumerate(arm.links, start=1):
        geom, inertial = link.geom, link.inertial
        rows.append(f"  {index:2d} {geom.mode:>10s} {str(geom.source):>14s} "
                    f"{geom.link_index:4d} {inertial.mode:>9s} {inertial.mass:8.3f} "
                    f"{str(np.round(inertial.inertia, 6)):>26s}")
    return "\n".join(rows)


def main() -> None:
    for source in sorted(SOURCES):
        print(describe(source))
        print()


if __name__ == "__main__":
    main()
