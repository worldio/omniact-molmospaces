"""node_spec.json is the typed node interface consumers (virtual_vla among them) read instead of
the MJCF, so its shape is a contract: complete and correct, with nothing extra. These tests pin that
contract on one emitted robot per gripper -- the schema, the tree, the measured calibration, and the
one property that took the longest to get right: driver axes signed toward opening, which jnt_axis
alone cannot express (yam's fingers share an axis and mirror purely through opposite qpos ranges).
"""

import json

import mujoco
import numpy as np
import pytest

from molmo_spaces.embodiment_gen.constants import NODE_SPEC_NAME
from molmo_spaces.embodiment_gen.node_spec import (
    BODY,
    GEOM,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_TCP,
    JOINT_TERMINAL,
    SITE,
    TCP_NODE,
    node_axis_world,
)
from molmo_spaces.embodiment_gen.sampling import sample_params
from molmo_spaces.embodiment_gen.spec_emit import emit
from molmo_spaces.utils.asset_tokens import from_tokens

# One production-library seed per gripper (accepted in the 0..200000 run), NOT merely the first
# seed that samples each gripper. Library-grade matters: the ctrl->qpos calibration settles the
# actuator on the free-hanging robot, which is only meaningful when the jaw can actually move --
# on a screen-reject robot the arm can jam its own gripper (seed 4's yam settles to the same qpos
# at both ctrl extremes, open-closed ~3e-5) and the opening-axis sign then comes from noise. The
# fixture asserts the mapping still holds, so a sampler change fails loudly here rather than as
# three tests quietly exercising the same gripper.
# linear_4310 inherited yam_jaw's slot in the GRIPPERS tuple at generator version 3, so seed
# 96215 samples the same arm with the linear gripper.
GRIPPER_SEEDS = {"robotiq_2f85": 90099, "linear_4310": 96215, "rum": 43}

CALIBRATION_KEYS = {"gripper_ctrl_range", "gripper_qpos_at_ctrl", "gripper_open_ctrl",
                    "gripper_qpos_at_closed", "gripper_qpos_at_open"}
TOP_KEYS = {"nodes", "edges", "joint_type", "n_arm", "driver_qpos", "qpos_center",
            *CALIBRATION_KEYS}


@pytest.fixture(scope="module")
def entries(tmp_path_factory):
    """One emitted entry per gripper, screens disabled: the screens gate acceptance, not schema."""
    out = {}
    for gripper, seed in GRIPPER_SEEDS.items():
        params = sample_params(seed)
        assert params.gripper == gripper, (
            f"seed {seed} no longer samples {gripper}; update GRIPPER_SEEDS")
        directory = tmp_path_factory.mktemp(f"nodes_{gripper}")
        emit(params, directory, max_self_collision=None, min_grasp_feasibility=None)
        spec = json.loads((directory / NODE_SPEC_NAME).read_text())
        model = mujoco.MjModel.from_xml_string(from_tokens((directory / "model.xml").read_text()))   # ${ASSET_CACHE} tokens
        out[gripper] = (params, spec, model)
    return out


@pytest.mark.parametrize("gripper", sorted(GRIPPER_SEEDS))
def test_top_level_keys_are_exactly_the_contract(entries, gripper):
    _, spec, _ = entries[gripper]
    assert set(spec) == TOP_KEYS


@pytest.mark.parametrize("gripper", sorted(GRIPPER_SEEDS))
def test_every_node_is_a_typed_anchor_and_nothing_more(entries, gripper):
    _, spec, _ = entries[gripper]
    for node in spec["nodes"]:
        assert set(node) == {"kind", "name", "centre_offset", "axis"}
        assert node["kind"] in (BODY, GEOM, SITE)
        assert len(node["centre_offset"]) == 3
        if node["axis"] is not None:
            assert len(node["axis"]) == 3
            assert np.linalg.norm(node["axis"]) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("gripper", sorted(GRIPPER_SEEDS))
def test_node_set_is_arm_then_two_drivers_then_the_tcp_then_two_terminals(entries, gripper):
    params, spec, _ = entries[gripper]
    n_arm = len(params.arm.joints)
    assert spec["n_arm"] == n_arm
    assert len(spec["nodes"]) == n_arm + 5
    assert len(spec["driver_qpos"]) == 2
    # BFS from the arm root: the first n_arm nodes are the chain in order, drivers precede the TCP (a sideless
    # child of the wrist, so it sorts after them) which precedes the terminals they carry.
    assert [n["name"] for n in spec["nodes"][:n_arm]] == [f"link{i}" for i in range(1, n_arm + 1)]
    types = spec["joint_type"]
    assert len(types) == len(spec["nodes"])
    assert types[:n_arm] == [JOINT_REVOLUTE] * n_arm
    assert types.count(JOINT_TERMINAL) == 2
    driver_type = JOINT_PRISMATIC if gripper == "linear_4310" else JOINT_REVOLUTE
    drivers = [i for i, t in enumerate(types) if i >= n_arm and t == driver_type]
    terminals = [i for i, t in enumerate(types) if t == JOINT_TERMINAL]
    assert len(drivers) == 2
    assert types.count(JOINT_TCP) == 1
    tcp = types.index(JOINT_TCP)
    assert max(drivers) < tcp < min(terminals)
    assert spec["nodes"][tcp] == dict(TCP_NODE)          # the catalog site, kind site, its axis the approach (-z)
    assert spec["edges"][tcp - 1] == [n_arm - 1, tcp]    # a child of the wrist node


@pytest.mark.parametrize("gripper", sorted(GRIPPER_SEEDS))
def test_edges_form_a_tree_over_all_nodes(entries, gripper):
    _, spec, _ = entries[gripper]
    edges = spec["edges"]
    assert len(edges) == len(spec["nodes"]) - 1
    children = [child for _, child in edges]
    assert sorted(children) == list(range(1, len(spec["nodes"])))  # every non-root exactly once
    for parent, child in edges:
        assert parent < child  # BFS order makes every edge point down the tree


@pytest.mark.parametrize("gripper", sorted(GRIPPER_SEEDS))
def test_driver_axes_point_toward_opening_so_they_oppose(entries, gripper):
    """The two fingers move apart as the jaw opens, so their signed axes must oppose in the world.

    This is the measured-not-declared property: yam's raw jnt_axis vectors are IDENTICAL (mirroring
    lives in reversed qpos ranges), and before signing, both arrows pointed the same way."""
    _, spec, model = entries[gripper]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    n_arm = spec["n_arm"]
    left, right = spec["nodes"][n_arm], spec["nodes"][n_arm + 1]
    dot = float(np.dot(node_axis_world(model, data, left), node_axis_world(model, data, right)))
    assert dot < -0.99, f"{gripper}: driver axes should oppose, dot={dot:.3f}"


@pytest.mark.parametrize("gripper", sorted(GRIPPER_SEEDS))
def test_calibration_is_measured_and_consistent(entries, gripper):
    """ctrl->qpos comes from settling the actuator at both extremes, not from declared ranges --
    an affine map from the declaration is wrong by 0.78 rad on rum and inverted on yam's right
    finger. Consistency check: the open-end qpos per driver matches the qpos_at_ctrl column the
    open ctrl extreme selects."""
    _, spec, _ = entries[gripper]
    lo, hi = spec["gripper_ctrl_range"]
    assert hi > lo
    assert spec["gripper_open_ctrl"] in (lo, hi)
    at_extremes = spec["gripper_qpos_at_ctrl"]  # [qpos per driver at ctrl lo, ... at ctrl hi]
    assert len(at_extremes) == 2 and all(len(row) == 2 for row in at_extremes)
    open_row = 0 if spec["gripper_open_ctrl"] == lo else 1
    assert at_extremes[open_row] == spec["gripper_qpos_at_open"]
    assert at_extremes[1 - open_row] == spec["gripper_qpos_at_closed"]


@pytest.mark.parametrize("gripper", sorted(GRIPPER_SEEDS))
def test_qpos_center_is_midrange_per_limited_joint(entries, gripper):
    """Tolerance is 1e-4, not 1e-6: the spec derives centres from the in-memory float64 model,
    while this test recomputes them from the re-parsed model.xml, whose ranges are text with ~6
    significant digits -- a 2e-6 disagreement is the serialization, not the centre."""
    _, spec, model = entries[gripper]
    center = spec["qpos_center"]
    assert len(center) == model.nq
    for jid in range(model.njnt):
        if model.jnt_limited[jid] and model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                              mujoco.mjtJoint.mjJNT_SLIDE):
            adr = int(model.jnt_qposadr[jid])
            lo, hi = model.jnt_range[jid]
            assert center[adr] == pytest.approx((lo + hi) / 2, abs=1e-4)
