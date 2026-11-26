"""CPU implementations of cuRobo's compiled CUDA kernels.

cuRobo is CUDA-only by construction: each of the five extensions in ``curobo/curobolib`` asserts
``x.is_cuda()`` on entry (``curobolib/cpp/lbfgs_step_cuda.cpp:36`` and siblings), and
``curobo/types/base.py:21`` defaults the device to ``cuda:0``. But cuRobo imports those extensions
from only eight sites, all of the form ``from curobo.curobolib.X import <autograd Function>``, and
everything above that line -- ``MotionGen``, ``TrajOptSolver``, ``ArmReacher``, the graph planner,
every cost term -- is pure Python and torch.

So the whole library runs on CPU if these leaf kernels do. Each module here mirrors one ``_cu``
extension: same function names, same argument order, same in-place out-tensor convention, same
return tuples, so it can be substituted at cuRobo's own import guard without touching anything
above it.

**Parity, not identity.** cuRobo does not reproduce its own GPU trajectories bitwise -- measured at
the shipped budget, five replays of one query gave a worst pairwise deviation of 4.7e-4 rad. These
implementations are therefore held to a tolerance, not to equality; see
``scripts/curobo/check_cpu_kernel_parity.py``. Two known systematic differences, both far inside
that envelope:

* the CUDA kinematics uses ``__cosf``/``__sinf`` fast-math intrinsics (~2 ULP), CPU uses IEEE;
* CUDA reduces with ``__shfl`` warp trees, CPU sums sequentially.
"""

from molmo_spaces.planner.curobo_cpu import geom, kinematics_fused, lbfgs_step, line_search, tensor_step

__all__ = ["geom", "kinematics_fused", "lbfgs_step", "line_search", "tensor_step"]
