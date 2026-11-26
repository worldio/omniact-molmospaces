"""Node / action representations and the A <-> B <-> C conversions -- see docs/REPRESENTATIONS.md.

Four forms (Env / Semantic / Abstract, each with a Proprio and an Action) with parallel field names; the
dataclass TYPE carries the frame/units. Conversions are ``.to_<adjacent>()``, strictly A <-> B <-> C -- the
action inverse goes C -> B -> A, there is no C -> A shortcut. ``Kinematics`` is the one stateful piece (a
cached compiled robot model).

The module-level functions are the low-level math, byte-identical to the former ``node_kinematics.py``; the
datagen converter uses them batched over time, while the dataclass methods below wrap them for the
per-frame rollout server."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from molmo_spaces.embodiment_gen.constants import GRIPPER_PREFIX, TCP_SITE
from molmo_spaces.utils.pose import GL_FROM_CV      # env-native OpenCV cam2world -> OpenGL

AXIS_LENGTH = 0.07          # metres along the joint axis to the axis_tip anchor (frozen convention)
# OpenGL camera-local (x right, y UP, z BACKWARD) -> WODS camera-local (x right, y up, z FORWARD).
# Right-multiplied onto a cam-to-world rotation; flips only the camera's own z. NOT virtual_vla's
# CV_TO_WODS = diag(1,-1,1) -- MolmoSpaces records OpenGL cameras (invariant asserted in c2w_from_cam2world).
GL_TO_WODS = np.diag([1.0, 1.0, -1.0])


# ============================ low-level math (B<->C geometry, A<->B gripper) ============================
def camera_frame(world: np.ndarray, cam2world_gl: np.ndarray) -> np.ndarray:
    """``(T, N, P, 3)`` anchors in the WODS CAMERA frame, from world anchors and per-frame GL cam2world."""
    out = np.zeros(world.shape[:3] + (3,), np.float64)
    for frame in range(world.shape[0]):
        rotation = np.asarray(cam2world_gl[frame][:3, :3], float) @ GL_TO_WODS
        position = np.asarray(cam2world_gl[frame][:3, 3], float)
        out[frame] = (world[frame] - position) @ rotation      # world -> camera-local (wods axes)
    return out


def c2w_from_cam2world(cam2world_gl: np.ndarray) -> np.ndarray:
    """``(T, 3, 4)`` wods camera-to-world, from the recorder's OpenGL cam-to-world matrices."""
    matrices = np.asarray(cam2world_gl, float)
    out = np.empty((matrices.shape[0], 3, 4), dtype=np.float32)
    for i, matrix in enumerate(matrices):
        cam_to_world = matrix[:3, :4].copy()
        cam_to_world[:, :3] = cam_to_world[:, :3] @ GL_TO_WODS
        out[i] = cam_to_world
    look = -matrices[:, :3, 2]                       # OpenGL cameras look along their own -z
    if not np.allclose(out[:, :3, 2], look, atol=1e-5):
        raise ValueError("wods c2w +z must be the camera look direction; it is not")
    if not np.allclose(out[:, :3, 1], matrices[:, :3, 1], atol=1e-5):
        raise ValueError("wods c2w +y must stay the camera up axis; it changed")
    if not np.allclose(out[:, :3, 3], matrices[:, :3, 3], atol=1e-5):
        raise ValueError("wods c2w must not move the camera position; it did")
    return out


def ctrl_to_qpos(command, ctrl_range, qpos_at_ctrl, driver: int):
    """Actuator ``command`` (ctrl units) -> driver ``driver``'s joint qpos, via the measured two-point
    calibration. Element-wise, so ``command`` may be an array."""
    low, high = ctrl_range
    at_low, at_high = qpos_at_ctrl[0][driver], qpos_at_ctrl[1][driver]
    fraction = (command - low) / ((high - low) or 1e-9)
    return at_low + fraction * (at_high - at_low)


def qpos_to_ctrl(qpos, ctrl_range, qpos_at_ctrl, driver: int):
    """Inverse of ``ctrl_to_qpos``: driver ``driver``'s target qpos -> actuator ctrl."""
    low, high = ctrl_range
    at_low, at_high = qpos_at_ctrl[0][driver], qpos_at_ctrl[1][driver]
    fraction = (qpos - at_low) / ((at_high - at_low) or 1e-9)
    return low + fraction * (high - low)


def driver_semantic_qpos_map_to_semantic_proprio_map(driver_semantic_qpos_map, n_arm):
    """Drivers' (semantic node order -> full-model qpos address) map -> (semantic node order -> move-group
    proprio column) map. Column = the driver's rank by address (move-group orders joints ascending --
    verified across the shipped gripper types)."""
    order = sorted(driver_semantic_qpos_map)
    return [n_arm + order.index(a) for a in driver_semantic_qpos_map]


def driver_semantic_qpos_map_to_proprio_qpos_map(driver_semantic_qpos_map):
    """Drivers' (semantic node order -> qpos address) map -> (move-group proprio column -> qpos address) map:
    the addresses in ascending == proprio-column order. ``anchors`` scatters the move-group gripper to these."""
    return sorted(driver_semantic_qpos_map)


def proprio_to_semantic_node_proprio(proprio, driver_proprio_index, n_arm, n_nodes, dtype=np.float32):
    """``(T, N)`` UNCENTERED per-node scatter of a move-group qpos vector: arm 1:1, each driver node from its
    proprio column, terminals zero. Batched over time. ``dtype`` is float32 for proprio; the action path
    passes float64 for an absolute-target / current-qpos row."""
    proprio = np.asarray(proprio, dtype)
    out = np.zeros((proprio.shape[0], n_nodes), dtype)
    out[:, :n_arm] = proprio[:, :n_arm]
    for node, col in enumerate(driver_proprio_index, start=n_arm):
        out[:, node] = proprio[:, col]
    return out


def semantic_node_proprio_to_abstract_node_proprio(semantic, semantic_node_proprio_center):
    """``(T, N)`` centered (Abstract) node_proprio -- subtract the per-node center."""
    return np.asarray(semantic, np.float32) - np.asarray(semantic_node_proprio_center, np.float32)


def abstract_node_proprio_to_semantic_node_proprio(abstract, semantic_node_proprio_center):
    """``(T, N)`` UNCENTERED (Semantic) node_proprio -- add the per-node center back."""
    return np.asarray(abstract, np.float32) + np.asarray(semantic_node_proprio_center, np.float32)


def action_to_semantic_node_action(action, n_arm, n_nodes, n_drivers, ctrl_range=None, qpos_at_ctrl=None,
                                   dtype=np.float32):
    """``(T, N)`` move-group action (arm targets + one gripper ctrl command) -> per-node ABSOLUTE target qpos:
    arm 1:1, driver node k = ``ctrl_to_qpos(command, k)`` (calibrated by node rank -- one command fans out to
    all drivers, so no proprio-column gather), terminals zero."""
    action = np.asarray(action, dtype)
    out = np.zeros((action.shape[0], n_nodes), dtype)
    out[:, :n_arm] = action[:, :n_arm]
    if n_arm < action.shape[1]:
        command = action[:, n_arm]
        for k in range(n_drivers):
            out[:, n_arm + k] = (ctrl_to_qpos(command, ctrl_range, qpos_at_ctrl, k)
                                 if (ctrl_range and qpos_at_ctrl) else command)
    return out


def semantic_node_action_to_abstract_node_action(semantic_node_action, semantic_node_proprio, dtype=np.float32):
    """``(T, N)`` relative node_action = absolute target - the current (uncentered) semantic node_proprio."""
    return np.asarray(semantic_node_action, dtype) - np.asarray(semantic_node_proprio, dtype)


def abstract_node_action_to_semantic_node_action(abstract_node_action, semantic_node_proprio, dtype=np.float32):
    """``(T, N)`` absolute node_action = relative action + the current semantic node_proprio (inverse of above)."""
    return np.asarray(abstract_node_action, dtype) + np.asarray(semantic_node_proprio, dtype)


def relative_node_action(action: np.ndarray, proprio: np.ndarray, spec: dict, driver_proprio_index: list[int],
                         ctrl_range: list | None = None,
                         qpos_at_ctrl: list | None = None) -> np.ndarray:
    """``(T, N)`` datagen action -> relative node_action in one pass: the absolute per-node target minus the
    current semantic node_proprio."""
    n_arm, n_nodes = int(spec["n_arm"]), len(spec["nodes"])
    semantic = action_to_semantic_node_action(action, n_arm, n_nodes, len(driver_proprio_index),
                                              ctrl_range, qpos_at_ctrl)
    current = proprio_to_semantic_node_proprio(proprio, driver_proprio_index, n_arm, n_nodes)
    return semantic_node_action_to_abstract_node_action(semantic, current)


# ============================ schema ============================
@dataclass(frozen=True)
class NodeSpec:
    """The embodiment's fixed node schema; wraps the ``node_spec.json`` dict (``.raw``) so the low-level
    functions above keep taking a plain dict while callers get typed access + the static graph."""
    raw: dict

    @classmethod
    def from_json(cls, path) -> "NodeSpec":
        return cls(json.loads(Path(path).read_text()))

    @property
    def n_arm(self) -> int: return int(self.raw["n_arm"])
    @property
    def n_nodes(self) -> int: return len(self.raw["nodes"])
    @property
    def driver_qpos(self) -> list[int]: return self.raw["driver_qpos"]
    @property
    def qpos_center(self) -> np.ndarray: return np.asarray(self.raw["qpos_center"], np.float64)
    @property
    def gripper_ctrl_range(self): return self.raw.get("gripper_ctrl_range")
    @property
    def gripper_qpos_at_ctrl(self): return self.raw.get("gripper_qpos_at_ctrl")
    @property
    def node_names(self) -> list[str]: return [n["name"] for n in self.raw["nodes"]]

    def graph(self) -> dict:
        """Static per-embodiment graph: edges, axis_mask, joint_type, node_names (was node_graph_tensors)."""
        types = torch.tensor(self.raw["joint_type"], dtype=torch.long)
        return {"edges": torch.tensor(self.raw["edges"], dtype=torch.long).reshape(-1, 2),
                "axis_mask": (types == 1) | (types == 2),   # revolute | prismatic; terminal (0) and TCP (3) False
                "joint_type": types, "node_names": self.node_names}

    @property
    def driver_proprio_index(self) -> list[int]:
        """Each driver node's proprio column (move-group index), from the stored full-model driver_qpos."""
        return driver_semantic_qpos_map_to_semantic_proprio_map(self.driver_qpos, self.n_arm)

    @property
    def semantic_node_proprio_center(self) -> np.ndarray:
        """Per-node center for B<->C: qpos_center scattered onto nodes (reusing the node scatter). Model-free
        and exact -- driver_qpos (full-model addresses) index the full-model qpos_center directly."""
        c = np.asarray(self.qpos_center, np.float32)
        return proprio_to_semantic_node_proprio(c[None], self.driver_qpos, self.n_arm, self.n_nodes)[0]


def node_graph_tensors(spec: dict) -> dict:
    """Static per-embodiment graph from a spec dict (== ``NodeSpec(spec).graph()``); kept as a free function
    for the batched datagen converter, which passes a plain dict."""
    return NodeSpec(spec).graph()


# ============================ FK (A -> B geometry) ============================
def _ensure_tcp_site(embodiment, entry: Path) -> None:
    """Give the FK model the gripper's ``tcp_frame`` site when the entry's spec predates the catalog's second
    add_site. Injected IN MEMORY into this robot-only model, never written back: the datagen scene model and
    the replay bundles must stay byte-identical to what was recorded (the exact-replay fingerprint hashes
    model names, so a site added to a bundle spec would break its reconstruction). Fresh emissions carry
    the site in spec.yaml and are left alone. Fails loudly for a gripper family without a catalog frame."""
    import json
    from pydantic import TypeAdapter
    from molmo_spaces.embodiment_gen.grippers import load_catalog
    from molmo_spaces.robots.configured.spec import ModelOperationSpec

    ops = embodiment.robot.model_operations
    if any(getattr(op, "kind", None) == "add_site" and op.name.name == GRIPPER_PREFIX + TCP_SITE for op in ops):
        return
    family = json.loads((Path(entry) / "params.json").read_text())["gripper"]
    match = [op for op in load_catalog()[family].model_operations
             if op.get("name", {}).get("name") == GRIPPER_PREFIX + TCP_SITE]
    assert match, f"gripper family {family!r}: the catalog declares no {GRIPPER_PREFIX + TCP_SITE} add_site"
    ops.append(TypeAdapter(ModelOperationSpec).validate_python(match[0]))


class Kinematics:
    """FK on the ASSEMBLED robot (cached compiled model). ``anchors`` mutates ``self.data`` in place and is
    reused across frames -- the one stateful object; everything else here is pure."""

    def __init__(self, entry: Path, spec: dict) -> None:
        import mujoco
        from molmo_spaces.robots.configured.config import ConfiguredRobotConfig
        from molmo_spaces.robots.configured.loader import load_embodiment_spec

        embodiment = load_embodiment_spec(Path(entry) / "spec.yaml")
        _ensure_tcp_site(embodiment, Path(entry))
        config = ConfiguredRobotConfig(embodiment=embodiment)
        scene = mujoco.MjSpec()
        self.prefix = config.robot_namespace or ""
        config.robot_cls.add_robot_to_scene(config, scene, prefix=self.prefix, pos=[0.0, 0.0, 0.0],
                                            quat=[1.0, 0.0, 0.0, 0.0])
        self.mujoco = mujoco
        self.model = scene.compile()
        self.data = mujoco.MjData(self.model)
        self.spec = spec
        self.n_arm = int(spec["n_arm"])
        self.nodes = spec["nodes"]                   # includes the TCP node when the spec carries it
        # move-group proprio column -> full-model qpos address, for the gripper -> data.qpos write below
        self.driver_proprio_qpos_map = driver_semantic_qpos_map_to_proprio_qpos_map(spec.get("driver_qpos", []))

    def pose(self, proprio_row: dict) -> None:
        """Put the model at one move-group proprio row (``{"arm", "gripper"}``) and run FK."""
        arm = proprio_row.get("arm")
        gripper = proprio_row.get("gripper")
        arm = [] if arm is None else arm
        gripper = [] if gripper is None else gripper
        self.data.qpos[:] = 0.0
        self.data.qpos[:len(arm)] = arm
        # drivers can be non-contiguous (robotiq: qpos n_arm & n_arm+3, passives between); scatter each
        # move-group value to its real full-model address. Contiguous grippers land on the old slots.
        for i, address in enumerate(self.driver_proprio_qpos_map):
            if i < len(gripper):
                self.data.qpos[address] = gripper[i]
        self.mujoco.mj_forward(self.model, self.data)

    def _to_world(self, points: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
        """Base-local points ``(..., 3)`` -> house-world via ``base_pose`` (xyz + wxyz)."""
        rotation = np.zeros(9)
        self.mujoco.mju_quat2Mat(rotation, np.asarray(base_pose[3:7], np.float64))
        return points @ rotation.reshape(3, 3).T + np.asarray(base_pose[:3], np.float64)

    def anchors(self, proprio_row: dict, base_pose: np.ndarray) -> np.ndarray:
        """``(N, 2, 3)`` world anchors (centre, axis_tip) per node at one move-group proprio; terminals ->
        [centre, centre]. ``base_pose`` (xyz + wxyz) lifts base-local FK into house-world."""
        from molmo_spaces.embodiment_gen import node_spec as ns

        self.pose(proprio_row)
        out = np.zeros((len(self.nodes), 2, 3), np.float64)
        for index, node in enumerate(self.nodes):
            named = node | {"name": self.prefix + node["name"]}
            centre = ns.node_world(self.model, self.data, named)
            axis = ns.node_axis_world(self.model, self.data, named)
            out[index, 0] = centre
            out[index, 1] = centre if axis is None else centre + AXIS_LENGTH * axis
        return self._to_world(out, base_pose)

    def poses(self, proprio_row: dict, base_pose: np.ndarray) -> np.ndarray:
        """``(N, 4, 3)`` world points ``[centre, centre + x, centre + y, centre + z]`` per node at one move-group
        proprio: the TRANSMITTED axes of the pose nodes -- the TCP site (joint type 3), the one frame that is
        canonical across families. The site itself is GL-style like a camera (x toward the right pad, y
        wrist-image-up, z back into the wrist); what travels is WODS-style like a camera's ``c2w``: x and y kept,
        z = -site z, forward into the fingertips (``GL_TO_WODS``, det -1 against the site). Every other node has
        no frame to transmit (a body/geom frame is the asset's own convention) and gets its centre four times,
        which ``pose_from_points`` turns into an all-zero matrix. POINTS rather than a matrix so the same point
        transform as the anchors (``camera_frame``) carries them into a view."""
        from molmo_spaces.embodiment_gen import node_spec as ns

        self.pose(proprio_row)
        out = np.zeros((len(self.nodes), 4, 3), np.float64)
        for index, node in enumerate(self.nodes):
            named = node | {"name": self.prefix + node["name"]}
            centre = ns.node_world(self.model, self.data, named)
            has_frame = self.spec["joint_type"][index] == 3                   # the TCP; the other nodes: none
            axes = ns.node_frame(self.model, self.data, named)[1] @ GL_TO_WODS if has_frame else np.zeros((3, 3))
            out[index] = centre + np.concatenate([np.zeros((1, 3)), axes.T])
        return self._to_world(out, base_pose)


def pose_from_points(points: np.ndarray) -> np.ndarray:
    """``(..., 4, 3)`` frame points ``[origin, origin + x, origin + y, origin + z]`` -> ``(..., 3, 4)`` SE(3)
    ``[x y z | origin]``: the frame's own coordinates into the coordinates the points are given in. The axes keep
    their length under any rigid map of the points; a reflecting one (``camera_frame``, into the WODS camera
    frame) flips the determinant. A node without a frame (coincident points, see ``Kinematics.poses``) comes
    out all zero."""
    points = np.asarray(points, np.float64)
    origin = points[..., :1, :]
    axes = points[..., 1:, :] - origin                                        # rows x, y, z
    framed = np.linalg.norm(axes[..., 0, :], axis=-1) > 0.5                   # unit length, or none
    matrix = np.concatenate([axes, origin], axis=-2).swapaxes(-1, -2)         # (..., 3, 4)
    return np.where(framed[..., None, None], matrix, 0.0)


def _project(world_anchors: np.ndarray, cam2world: dict) -> np.ndarray:
    """``(N, 2, V, 3)`` -- world anchors in each of the V camera frames (dict order). Each env-native CV
    pose is flipped to OpenGL (``GL_FROM_CV``) then projected to the WODS frame (``camera_frame``) --
    the same two steps, in order, that the rollout server did per view."""
    geoms = []
    for cv in cam2world.values():
        gl = np.asarray(cv, float) @ GL_FROM_CV
        geoms.append(camera_frame(world_anchors[None], gl[None])[0])
    return np.stack(geoms, axis=2).astype(np.float32)      # WODS node_geometry is float32 (the wire dtype)


def c2w_per_view(cam2world: dict) -> np.ndarray:
    """``(V, 3, 4)`` WODS camera-to-world per camera (dict order) from the env-native OpenCV poses — the same
    CV -> GL flip ``_project`` applies, then ``c2w_from_cam2world``. This is the c2w that composes with
    ``node_pose`` / ``node_geometry``: the world pose is ``c2w[v] · node_pose[·, v]`` (SE(3) compose)."""
    return np.stack([c2w_from_cam2world((np.asarray(cv, float) @ GL_FROM_CV)[None])[0]
                     for cv in cam2world.values()], axis=0).astype(np.float32)   # (V, 3, 4)


def _flat_qpos(qpos: dict) -> np.ndarray:
    """Move-group dict -> the flat ``[arm | gripper]`` qpos the node scatter indexes with ``driver_qpos``
    (float64). A conversion detail, not env state -- so it lives here, not on ``EnvProprio``."""
    arm, gripper = qpos.get("arm"), qpos.get("gripper")
    arm = [] if arm is None else arm
    gripper = [] if gripper is None else gripper
    return np.concatenate([np.asarray(arm, np.float64), np.asarray(gripper, np.float64)])


# PROPRIO to_*() take only self (a proprio carries its own spec + kin). ACTION dataclasses are pure data;
# their to_*() take the paired proprio (or its spec) -- an action is relative to a state, so the state
# supplies the live qpos / FK context / schema. Every adjacent conversion is implemented: proprio A<->B<->C
# and action A->B, B<->C, B->A / C->B. The one that does NOT exist is proprio B->A: the node form has
# dropped base_pose / cam2world / kin, so the env state is genuinely not recoverable.

# ============================ Form A: env controller ============================
@dataclass
class EnvProprio:
    qpos: dict                        # {"arm","gripper","base"} native joint qpos, per move-group
    base_pose: np.ndarray             # (7,) xyz + wxyz (native sensor form)
    cam2world: dict                   # {cam_name: (4,4)} OpenCV cam-to-world -- env-native (registry.get_pose())
    spec: NodeSpec                    # embodiment schema (context)
    kin: Kinematics                   # FK evaluator (context)

    def to_semantic(self) -> "SemanticProprio":   # A -> B
        flat = _flat_qpos(self.qpos)
        node_proprio = proprio_to_semantic_node_proprio(
            flat[None], self.spec.driver_proprio_index, self.spec.n_arm, self.spec.n_nodes)[0]
        row = {"arm": self.qpos.get("arm"), "gripper": self.qpos.get("gripper")}
        world = self.kin.anchors(row, self.base_pose)
        # the TCP node's frame (every other row zero) as four points, projected per view like the anchors
        frames = _project(self.kin.poses(row, self.base_pose), self.cam2world)                   # (N, 4, V, 3)
        node_pose = pose_from_points(np.transpose(frames, (0, 2, 1, 3))).astype(np.float32)    # (N, V, 3, 4)
        return SemanticProprio(node_proprio=torch.from_numpy(node_proprio),
                               node_geometry=torch.from_numpy(_project(world, self.cam2world)),
                               node_pose=torch.from_numpy(node_pose), spec=self.spec)


@dataclass
class EnvAction:
    move_groups: dict                 # {"arm": abs joint targets (radians), "gripper": ctrl units, ...}
                                      # per move-group -- exactly what task.step / robot.update_control take
    info: dict = field(default_factory=dict)   # how the arm target was resolved at the commit (IK outcome / errors, head gap) -- rollout only

    def to_semantic(self, proprio: EnvProprio) -> "SemanticAction":   # A -> B, paired with the issuing state
        spec = proprio.spec
        arm = np.asarray(self.move_groups.get("arm"), np.float64)
        gripper = np.asarray(self.move_groups.get("gripper")).reshape(-1)
        raw = np.concatenate([arm, gripper[:1].astype(np.float64)]) if gripper.size else arm
        node_action = action_to_semantic_node_action(   # absolute target
            raw[None], spec.n_arm, spec.n_nodes, len(spec.driver_qpos),
            spec.gripper_ctrl_range, spec.gripper_qpos_at_ctrl, dtype=np.float64)[0]
        mg_gripper = [0.0] * len(spec.driver_qpos)          # move-group order; anchors scatters to real addrs
        for k, col in enumerate(spec.driver_proprio_index):
            mg_gripper[col - spec.n_arm] = ctrl_to_qpos(
                float(gripper[0]), spec.gripper_ctrl_range, spec.gripper_qpos_at_ctrl, k)
        target_row = {"arm": self.move_groups.get("arm"), "gripper": mg_gripper}
        world = proprio.kin.anchors(target_row, proprio.base_pose)
        # the TCP node's COMMANDED frame (other rows zero) in house-world (camera-independent; == wods node_action_pose)
        action_pose = pose_from_points(proprio.kin.poses(target_row, proprio.base_pose)).astype(np.float32)   # (N, 3, 4)
        return SemanticAction(node_action=torch.from_numpy(node_action),
                              node_action_geometry=torch.from_numpy(_project(world, proprio.cam2world)),
                              node_action_pose=torch.from_numpy(action_pose))


# ============================ Form B: per-node semantic (WODS camera frame) ============================
@dataclass
class SemanticProprio:
    node_proprio: torch.Tensor        # (N,) native, uncentered (terminals 0)
    node_geometry: torch.Tensor       # (N, 2, V, 3) WODS camera-frame anchors
    node_pose: torch.Tensor           # (N, V, 3, 4) achieved TCP frame [R | t] per view: WODS camera <- WODS-style TCP axes (det +1); other rows 0
    spec: NodeSpec

    def to_abstract(self) -> "AbstractProprio":   # B -> C
        g = self.spec.graph()
        centered = semantic_node_proprio_to_abstract_node_proprio(
            np.asarray(self.node_proprio), self.spec.semantic_node_proprio_center)
        return AbstractProprio(
            node_proprio=torch.from_numpy(centered),
            node_geometry=self.node_geometry,                              # already WODS camera frame (torch)
            node_pose=self.node_pose,
            edges=g["edges"], axis_mask=g["axis_mask"], joint_type=g["joint_type"], node_names=g["node_names"],
            spec=self.spec)


@dataclass
class SemanticAction:
    node_action: torch.Tensor                     # (N,) native ABSOLUTE target qpos (terminals 0)
    node_action_pose: torch.Tensor                # (N, 3, 4) commanded TCP frame [R | t]: HOUSE-WORLD <- WODS-style TCP axes (det -1, like c2w); other rows 0. Zero the mask to ignore a head, never None
    node_action_geometry: torch.Tensor | None = None  # (N, 2, V, 3) WODS camera anchors of target pose (datagen label only)

    def to_abstract(self, proprio: "SemanticProprio") -> "AbstractAction":   # B -> C  (relative to the current state)
        rel = semantic_node_action_to_abstract_node_action(self.node_action, proprio.node_proprio)
        return AbstractAction(node_action=torch.from_numpy(rel), node_action_pose=self.node_action_pose,
                              node_action_geometry=self.node_action_geometry)

    def to_env(self, spec: NodeSpec, *, current: "EnvProprio", action_mask, node_action_pose_mask, ik=None) -> EnvAction:   # B -> A, the env commit
        """Commit the semantic action to the joint targets the sim executes, against the LIVE state ``current``.
        The two masks are REQUIRED execution inputs -- every dim is explicitly on or off. Every joint starts HELD
        at ``current``; ``action_mask`` selects the ``node_action`` entries that drive their joint, and
        ``node_action_pose_mask`` (only the TCP row) sends ``node_action_pose`` through ``ik`` for the arm -- with
        the joint head's arm target as the posture reference when it also covers the arm (hybrid). A pose mask on a
        non-TCP row, or masks that command nothing, are an error; the IK outcome rides out on ``EnvAction.info``."""
        if current is None:
            raise ValueError("to_env needs the live state (current=)")
        if action_mask is None or node_action_pose_mask is None:
            raise ValueError("to_env needs explicit action_mask and node_action_pose_mask (every dim on or off)")
        n_arm, n = spec.n_arm, spec.n_nodes
        absolute = np.asarray(self.node_action, np.float64).reshape(n)
        result = proprio_to_semantic_node_proprio(                     # every node HELD at its live value
            _flat_qpos(current.qpos)[None], spec.driver_proprio_index, n_arm, n, dtype=np.float64)[0].copy()
        joint = np.asarray(action_mask, bool).reshape(n)
        result[joint] = absolute[joint]
        tcp = np.asarray(spec.raw["joint_type"]) == 3
        posed = np.asarray(node_action_pose_mask, bool).reshape(n)
        if posed.any() and not np.array_equal(posed, tcp):
            raise ValueError("node_action_pose_mask may enable only the TCP row (joint type 3)")
        if not joint.any() and not posed.any():
            raise ValueError("the masks command nothing (both action_mask and node_action_pose_mask are all-False)")
        info = {}
        if posed.any():
            if ik is None:
                raise ValueError("node_action_pose needs an IK solver")
            pose = np.asarray(self.node_action_pose, np.float64).reshape(n, 3, 4)[int(np.flatnonzero(tcp)[0])]
            arm, gripper = current.qpos.get("arm"), current.qpos.get("gripper")
            posture = None
            if joint[:n_arm].any():                                    # hybrid: the joint head's absolute arm target
                from molmo_spaces.embodiment_gen.ik import pose_gap
                posture = result[:n_arm].copy()
                info.update(pose_gap(ik.fk_world(posture, gripper, current.base_pose), pose))
            q, ik_info = ik.command(pose, arm, gripper, current.base_pose, posture=posture)
            result[:n_arm] = q
            info.update(ik_info)
        ctrls = [qpos_to_ctrl(float(result[n_arm + k]), spec.gripper_ctrl_range,
                              spec.gripper_qpos_at_ctrl, k) for k in range(len(spec.driver_qpos))]
        return EnvAction(move_groups={"arm": result[:n_arm],
                                      "gripper": np.array([float(np.mean(ctrls))], np.float64)}, info=info)


# ============================ Form C: per-node abstract (camera frame) ============================
@dataclass
class AbstractProprio:
    node_proprio: torch.Tensor        # (N,) centered, native
    node_geometry: torch.Tensor       # (N, 2, V, 3) WODS camera frame
    node_pose: torch.Tensor           # (N, V, 3, 4) achieved TCP frame, WODS camera <- TCP (== wods node_pose)
    edges: torch.Tensor
    axis_mask: torch.Tensor
    joint_type: torch.Tensor
    node_names: list                  # edges..node_names are the serialized observation
    spec: NodeSpec                    # server-side context (not serialized); carries the C->B inverse

    def to_semantic(self) -> "SemanticProprio":   # C -> B  (un-center; geometry already WODS camera frame)
        native = abstract_node_proprio_to_semantic_node_proprio(
            np.asarray(self.node_proprio), self.spec.semantic_node_proprio_center)
        return SemanticProprio(node_proprio=torch.from_numpy(native), node_geometry=self.node_geometry,
                               node_pose=self.node_pose, spec=self.spec)


@dataclass
class AbstractAction:
    node_action: torch.Tensor                     # (N,) relative, native -- the joint head. Zero the mask to ignore a head, never None
    node_action_pose: torch.Tensor                # (N, 3, 4) house-world <- WODS-style TCP axes, the TCP row the pose head (== wods node_action_pose). Zero the mask to ignore a head, never None
    node_action_geometry: torch.Tensor | None = None  # (N, 2, V, 3) camera frame (datagen label only)

    def to_semantic(self, current: EnvProprio) -> SemanticAction:   # C -> B, against the live state
        """Structure-preserving inverse: turn the relative ``node_action`` absolute against the LIVE env qpos
        (float64, as the former ``invert_node_action`` did) and carry the pose head through unchanged. No
        kinematics -- the pose is resolved to joints only at the env commit (``SemanticAction.to_env``), where the
        masks (the execution inputs) are supplied -- so both heads stay first-class through B and C exactly as the
        datagen label does."""
        spec = current.spec
        node_proprio = proprio_to_semantic_node_proprio(
            _flat_qpos(current.qpos)[None], spec.driver_proprio_index, spec.n_arm, spec.n_nodes, dtype=np.float64)[0]
        target = abstract_node_action_to_semantic_node_action(         # relative + current -> absolute
            np.asarray(self.node_action, np.float64).reshape(-1), node_proprio, dtype=np.float64)
        return SemanticAction(node_action=torch.from_numpy(target),
                              node_action_pose=self.node_action_pose,
                              node_action_geometry=self.node_action_geometry)
