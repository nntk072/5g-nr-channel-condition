from __future__ import annotations

import math

from experiments.audit_g5_tdl_profile_matrix import (
    ALPHA,
    BOOTSTRAP_REPLICATES,
    _bootstrap_interval,
    _exact_upper,
)
from experiments.g5_independent_channel_receiver import (
    clopper_pearson_upper_bound,
    cluster_bootstrap_interval,
)


def test_independent_bootstrap_audit_matches_retained_cluster_interval() -> None:
    values = [-0.1, 0.0, 0.2, 0.4, 0.1, 0.0, -0.2, 0.3]

    assert _bootstrap_interval(values, seed=934101) == cluster_bootstrap_interval(
        values,
        seed=934101,
        alpha=ALPHA,
        replicates=BOOTSTRAP_REPLICATES,
    )


def test_exact_reference_bound_matches_clopper_pearson_implementation() -> None:
    assert math.isclose(
        _exact_upper(2, 64),
        clopper_pearson_upper_bound(2, 64, alpha=ALPHA),
        rel_tol=0.0,
        abs_tol=2e-14,
    )
