import numpy as np
import pytest

from molmo_spaces.embodiment_gen.constants import GENERATOR_VERSION, GRIPPERS
from molmo_spaces.embodiment_gen.grippers import load_catalog
from molmo_spaces.embodiment_gen.params import SUPPORT
from molmo_spaces.embodiment_gen.sampling import GRAMMARS, sample_params
from molmo_spaces.embodiment_gen.skeleton import (
    chain_length,
    TWISTS,
    axes_at_zero,
    axes_ok,
    build_frames,
    flange_offset,
)

SEEDS = list(range(200))


def test_declared_grippers_match_the_catalog():
    """The sampler draws from an explicit tuple so a new catalog entry cannot silently
    reassign every seed's gripper -- but the tuple must not go stale either. yam_jaw is the
    one deliberate exception: it is the crank our shipped i2rt_yam carries, kept in the
    catalog as the reference-fidelity anchor but unsampled since generator version 3, when
    linear_4310 -- the gripper MolmoAct2's real bimanual-YAM rig runs -- took its slot."""
    assert set(GRIPPERS) <= set(load_catalog())
    assert set(load_catalog()) - set(GRIPPERS) == {"yam_jaw"}


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 999, 123456])
def test_sampling_is_deterministic(seed):
    first, second = sample_params(seed), sample_params(seed)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.version == GENERATOR_VERSION
    assert first.embodiment_id == f"gen_{seed}"


def test_distinct_seeds_give_distinct_robots():
    arms = {sample_params(seed).arm for seed in SEEDS}
    assert len(arms) > 0.95 * len(SEEDS), "sampling is collapsing distinct seeds"


def test_every_sample_is_structurally_valid():
    """The space is shaped so realization cannot fail: no degenerate axis set, DoF in range,
    one joint range per joint, a known gripper and grammar."""
    for seed in SEEDS:
        params = sample_params(seed)
        arm = params.arm
        assert arm.dof in (6, 7, 8)
        assert len(arm.joints) == arm.dof
        assert len(arm.segments) == arm.dof
        assert params.gripper in GRIPPERS
        assert arm.grammar in GRAMMARS
        assert axes_ok(axes_at_zero([s.quat for s in arm.segments])), (
            f"seed {seed}: degenerate axis set escaped rejection sampling")


def test_joint_ranges_and_lengths_are_sane():
    for seed in SEEDS:
        arm = sample_params(seed).arm
        assert 0.5 < arm.pedestal_height < 0.9
        for joint in arm.joints:
            low, high = joint.range
            assert low < high


def test_chain_length_is_independent_of_dof_for_sampled_geometry():
    """For arms whose geometry is sampled, scale is budgeted and then distributed, so adding a DoF
    buys redundancy rather than length -- sampling bones independently made 8-DoF arms ~40% longer.

    Arms that mostly wear REAL link geometry are deliberately excluded: they keep their source's
    natural spacing instead of being rescaled, because rigid link meshes only span the distance
    they were built for. Their length therefore follows the source robot, not the budget."""
    by_dof = {}
    for seed in range(6000):
        params = sample_params(seed)
        # ANY pool link now constrains segment offsets -- its own pose and its successor's, since
        # rigid geometry has to span both -- so only fully-primitive arms are budget-driven.
        if any(link.geom.mode == "pool" for link in params.arm.links):
            continue
        by_dof.setdefault(params.arm.dof, []).append(chain_length(params.arm))

    assert len(by_dof) == 3, sorted(by_dof)
    for dof, lengths in sorted(by_dof.items()):
        assert 0.6 < np.median(lengths) < 1.3, f"{dof}-dof median {np.median(lengths):.2f}"
    medians = [np.median(by_dof[dof]) for dof in sorted(by_dof)]
    assert max(medians) - min(medians) < 0.08, (
        f"chain length still tracks DoF: medians {[round(m, 3) for m in medians]}")


def test_fallback_quats_are_non_degenerate():
    """The rejection-sampling fallback must actually pass the screen, or some seeds yield a
    degenerate arm. Plain "alternating" gives only two distinct axis directions."""
    from molmo_spaces.embodiment_gen.sampling import _fallback_quats

    for dof in range(6, 9):
        assert axes_ok(axes_at_zero(_fallback_quats(dof))), dof


def test_frames_reconstruct_the_declared_axes():
    """build_frames must record the same world axes the degeneracy screen tested, or the
    screen is checking a different robot than the one that gets built."""
    for seed in SEEDS[:50]:
        arm = sample_params(seed).arm
        frames = build_frames(arm)
        assert len(frames) == arm.dof
        expected = axes_at_zero([s.quat for s in arm.segments])
        for frame, axis in zip(frames, expected):
            np.testing.assert_allclose(frame.axis_world, axis, atol=1e-9)
        for frame, segment in zip(frames, arm.segments):
            assert frame.pos == segment.offset


def test_link_geometry_is_a_mixture():
    """Geometry mode lives PER LINK now, not per arm: pool-only would drop the mesh-free
    variety, primitive-only would put every real robot's geometry outside the space."""
    modes = [link.geom.mode for seed in SEEDS for link in sample_params(seed).arm.links]
    assert set(modes) == {"primitive", "pool"}
    pool_fraction = modes.count("pool") / len(modes)
    assert 0.15 < pool_fraction < 0.60, f"pool fraction {pool_fraction:.2f} off target ~0.35"


def test_palette_channels_are_valid_rgba():
    for seed in SEEDS[:50]:
        for role, rgba in sample_params(seed).dressing.palette.items():
            assert len(rgba) == 4 and rgba[3] == 1.0, role
            assert all(0.0 <= channel <= 1.0 for channel in rgba), role
