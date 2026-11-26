# Node / action representations

Robot state and action pass through **four forms** with **three conversions** between them. The
**abstract** form (C) is the env↔policy protocol — WODS on disk plus the rollout gRPC observation.
Everything below C is molmospaces' job (the env / rollout server); everything above C is the policy's
(virtual_vla). Neither side sees the other's internal form.

## The forms

### A — Env controller  · molmospaces env/robot
Per-move-group dicts `{"arm": …, "gripper": …, "base": …}`.
- **proprio**: `robot_view.get_qpos_dict()` — joint qpos, radians / metres.
- **action**: `task.step({"arm", "gripper"})` → `robot.update_control` — arm = absolute joint targets
  (radians); gripper = raw **actuator ctrl units** (e.g. `-255…0`), *not* qpos.

The only place ctrl units and move-group grouping exist.

### B — Semantic (per-node)  · WODS conventions, un-normalized
The typed, WODS-ordered node structure in **WODS conventions but native units** — the intermediate between
the env and the model. A node is `(joint_type, axis, place in the graph, geometry)`; `node_spec.derive`
fuses *identify nodes* + *order (with lr_swap)* + *type* in one pass, so it is never "robot-native." It
uses:
- **geometry**: node anchors in the **WODS camera frame** (`+x` right, `+y` up, `+z` forward, det = −1 —
  neither OpenGL nor OpenCV); the world-frame FK is an internal step of `Env→Semantic`.
- **proprio**: native units, **uncentered** (`SemanticProprio.node_proprio`).
- **the TCP node**: `gripper/tcp_frame` (the catalog's second `add_site`: closed-pad midpoint, GL-style like
  a camera — `x` toward the right pad, `y` wrist-image-up, `z` back into the wrist), `joint_type = JOINT_TCP
  (3)`, `axis_mask False`, parent = the last arm node (the wrist, like the drivers); its anchor axis is the
  approach, `−z`. It sits **after the drivers and before the
  terminals** (index `n_arm + n_drivers`): a sideless child of the wrist node, which is where the breadth-first
  `(depth, side, name)` order puts it, so the node list is `arm…, driver_L, driver_R, tcp, tip_L, tip_R` in every
  family. Find it by joint type. `node_spec.derive` emits it; the post-run fix chain (branch
  `embgen-postrun-fix`, Fix 6) inserts it into data emitted before it.
- **pose**: the TCP node's frame as an SE(3) `[R | t]` per view, **WODS camera ← WODS-style TCP axes** (`x`
  right, `y` up, `z` forward into the fingertips — the site's GL axes with `z` flipped, exactly as `c2w` treats
  a camera; det +1, the identity when the TCP looks along the camera) (`SemanticProprio.node_pose`,
  `(N, V, 3, 4)`), carried like the anchors (four points through the same `camera_frame`). Every other row is
  all zero: a body/geom frame is the asset's own convention, not a quantity to transmit; the TCP frame is the
  one that is canonical across families.
- **action**: native units, **absolute target** (`SemanticAction.node_action`).
`Semantic→Abstract` then only centers the proprio and relatives the action; the geometry passes straight
through (already WODS).

### C — Per-node abstract  ·  ← the env↔policy protocol (WODS + gRPC obs)
Typed nodes in a fixed graph. A node is `(joint_type ∈ {revolute, prismatic, terminal}, axis, place in the
graph, geometry)`. The structural role (arm / gripper) is **erased** (`node_spec.derive` deletes `_role`);
there is **no base node**. Names are carried as metadata and ignored by the model — this erasure is what
lets one model span embodiments.
- **units**: native (radians / metres), **no normalization**.
- **proprio**: **centered**, `q − qpos_center` (mid-range, so zero means the same on every robot).
- **action**: **per-node relative**, `a − q`; the gripper command goes through the measured ctrl↔qpos
  calibration (measured once per family on the bare catalog gripper on a rigid mount, zero gravity: the commanded
  position, a property of the gripper, not a loaded rest position on one particular arm).
- **geometry**: node anchors `(center, axis_tip)` in the **WODS camera frame** (`+x` right, `+y` up, `+z`
  forward, det = −1 — neither OpenGL nor OpenCV).
- **graph**: `{edges, axis_mask, joint_type, node_names}`. **Node ordering is load-bearing** — canonical
  `(depth, side, name)` with lr_swap; the model right-aligns expecting a consistent suffix.
- **node_pose**: `(T, N, 3, 4)` per view shard (wods) / `(N, V, 3, 4)` (gRPC obs) — the TCP node's frame at
  the recorded state as SE(3) `[R | t]`, **WODS camera ← WODS-style TCP axes** (`x` right, `y` up, `z` forward;
  det +1 — no improper rotation inside the WODS system), observation side, pairs with `node_geometry`. The
  TCP row is the one with joint type 3; every other row is all zero (those nodes have no canonical frame, so
  none is transmitted).
- **node_action_pose**: `(T, N, 3, 4)` — the same frame at the **commanded** configuration (FK at
  `proprio + action`), label side, pairs with `node_action_geometry`; row `i` aligns with `action[i]`. The one
  quantity **not** in a camera frame: `[R | t]` **house-world ← WODS-style TCP axes**, det −1 exactly like
  `c2w` (the improper rotation sits at the MuJoCo ↔ WODS boundary, never inside either system), and the same
  track is stored in every shard — a label must not depend on which camera looks at it, and `c2w⁻¹` takes it
  into any view (`node_pose = c2w⁻¹ · [R | t]` at the recorded state). `AbstractAction.node_action_pose`, like
  `node_action_geometry`. On the wire it keeps its name and shape: `StepRequest.node_action_pose` `(N, 3, 4)`,
  **absolute**, optional beside the optional relative joint-space `action` (at least one); the server reads the
  TCP row and ignores the rest. All four wire fields are **required**; each head carries a `(N,)` bool mask of
  what it commands (`action_mask` the entries, `node_action_pose_mask` the row, only the TCP row allowed), and a
  head you do not command is zeros with an all-False mask. Everything unmasked is held. The pose's arm joints go
  through IK (`embodiment_gen/ik.py`, inside `SemanticAction.to_env`, the env commit); an `action_mask` that also covers the
  arm makes it hybrid, its arm target the solver's posture reference (the `ik.py` module docstring). The TCP setup is the
  pose row plus the gripper driver entries of `action`.
- `N` includes the TCP node everywhere (`node_geometry`, `node_proprio`, `action`, the graph): its proprio and
  action columns are 0 and its `axis_mask` is False, so it is never an actuated slot.

### D — Model  · virtual_vla policy
One padded per-node vector for proprio and one for action per step, plus the graph tensors.
- **normalized**: prismatic ×10 → decimetres (`to_model_joint_units`, needs `joint_type`) and ÷`action_std`.
- right-aligned to a fixed `n_node`; node names fully erased (`node_id = arange`).

## The boundaries — who owns each conversion

| Boundary | Direction | Owner | Conversion sites |
|---|---|---|---|
| **A ↔ C** | env state → abstract; abstract action → env | **molmospaces** (env / server) | proprio `proprio_to_semantic_node_proprio` (+ `semantic_node_proprio_to_abstract_node_proprio` to center); geometry `Kinematics.anchors` → `camera_frame`; poses `Kinematics.poses` → `camera_frame` (`node_pose`) / world (`node_action_pose`), packed by `pose_from_points`; action forward `relative_node_action`, inverse `invert_node_action`, pose → arm targets `ik.TcpIK` (in `SemanticAction.to_env`); ctrl↔qpos `ctrl_to_qpos` / `qpos_to_ctrl`; spec/order/type `node_spec.derive` |
| **C** | the wire | shared contract | WODS shard fields + gRPC `Observation` |
| **C ↔ D** | abstract ↔ model | **virtual_vla** (policy) | units `to_model_joint_units` / `to_native_joint_units`; norm ÷`action_std`; padding `pad_nodes_front` / `node_mask` |

Consequences (the invariants to preserve):
- The **policy** never sees qpos, ctrl units, or uncentered proprio — only C. The only unit-space thing it
  owns is the prismatic decimetre (hence it needs `joint_type`).
- The **env** never sees normalized or padded vectors — only C.
- All qpos / ctrl / centering lives at **A ↔ C** (molmospaces); all normalization / padding lives at
  **C ↔ D** (virtual_vla).
- The TCP is a **site**, never re-derived from nodes or from the wrist `c2w` (the wrist camera mount is
  randomized per episode). `Kinematics` injects the catalog's `tcp_frame` op into its own FK model when an
  entry's spec predates it; spec files and replay bundles are never rewritten (exact replay hashes model names).
