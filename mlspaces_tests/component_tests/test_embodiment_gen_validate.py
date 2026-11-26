"""Validation is the acceptance gate for generated embodiments, and both of its hard-won rules have
already caught real defects once: T1's spec-load check caught an emit regression that silently
dropped a required kinematics field, and T2's rest-pose contact check caught a robot whose second
link sat 30 mm inside its own pedestal -- invisible to every bare-model screen (IK ignores
collision; that robot had the run's HIGHEST reach score) and fatal to every planning attempt.
These tests pin one live seed on each side of the gate.
"""

import pytest

from molmo_spaces.embodiment_gen.sampling import sample_params
from molmo_spaces.embodiment_gen.spec_emit import emit
from molmo_spaces.embodiment_gen.validate import validate

# Both seeds are from the production 0..200000 range: 2198 is the pedestal-interpenetration robot,
# 90099 is a library entry that planned and produced trajectories.
INTERPENETRATING_SEED = 2198
ACCEPTED_SEED = 90099


def _emitted(tmp_path, seed):
    # Screens disabled: they gate acceptance upstream of validation and cost more than the
    # validation being tested.
    return emit(sample_params(seed), tmp_path / f"gen_{seed}",
                max_self_collision=None, min_grasp_feasibility=None).parent


def test_rest_pose_interpenetration_is_rejected_at_t2(tmp_path):
    """The assembled robot -- pedestal included -- must stand at rest without structural contact.

    The bare build() model has no pedestal, so this is only observable after assembly; seed 2198 is
    the regression case that motivated the check."""
    verdict = validate(_emitted(tmp_path, INTERPENETRATING_SEED))
    assert not verdict
    assert verdict.tier == "T2"
    assert "interpenetrat" in verdict.reason


def test_known_accepted_entry_still_passes(tmp_path):
    """A production library entry passes T1+T2 when re-emitted from its seed.

    This is also the regression test for the emitted spec itself: T1 loads spec.yaml through
    EmbodimentSpec validation, so an emit change that drops a required field (as one did with
    convergence_tolerance) fails here rather than in the next fleet run."""
    verdict = validate(_emitted(tmp_path, ACCEPTED_SEED))
    assert verdict, f"{verdict.tier}: {verdict.reason}"
