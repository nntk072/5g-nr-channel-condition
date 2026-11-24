"""Pilot sample reports: the audit rejects every altered field, roster and cell rule."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from experiments.pilot_sampling import audit_pilot_outcomes
from tests.test_experiments.test_pilot_sampling_design import _outcomes, _sample


def _cell(report: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return report["cells"][index]


def _selected(report: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return report["selected_candidates"][index]


CASES = [
    pytest.param(
        lambda r: r.update(extra_field=1), "invalid field roster", id="report-extra-field"
    ),
    pytest.param(
        lambda r: r.update(qualification_status="QUALIFIED"),
        "unsupported or qualification-eligible",
        id="report-qualified",
    ),
    pytest.param(
        lambda r: r.update(sampling_algorithm="other_v1"),
        "unsupported pilot sampling algorithm",
        id="algorithm",
    ),
    pytest.param(
        lambda r: r.update(status="SAMPLE_ROSTER_UNKNOWN"),
        "unsupported pilot sample status",
        id="status",
    ),
    pytest.param(lambda r: r.update(sampling_seed=-1), "sampling_seed is invalid", id="seed"),
    pytest.param(
        lambda r: r.update(candidate_count=0),
        "selected more candidates than the roster",
        id="candidate-count",
    ),
    pytest.param(
        lambda r: r.update(selected_candidate_count=10_000),
        "selected more candidates than the roster",
        id="selected-count",
    ),
    pytest.param(
        lambda r: r.update(limitations=[]), "limitations must be a nonempty list", id="limitations"
    ),
    pytest.param(
        lambda r: r.update(selected_candidates=r["selected_candidates"] * 100),
        "selected-candidate count is inconsistent",
        id="over-roster",
    ),
    pytest.param(
        lambda r: _selected(r).update(extra=1),
        "selected-candidate row has an invalid field roster",
        id="selected-row-roster",
    ),
    pytest.param(
        lambda r: r["selected_candidates"].append(copy.deepcopy(_selected(r))),
        "selected-candidate count is inconsistent",
        id="selected-duplicate",
    ),
    pytest.param(
        lambda r: r.update(selected_candidates="not-a-list"),
        "selected candidates/cells are malformed",
        id="selected-not-list",
    ),
    pytest.param(
        lambda r: r.update(cells="not-a-list"),
        "selected candidates/cells are malformed",
        id="cells-not-list",
    ),
    pytest.param(
        lambda r: r["cells"].append(copy.deepcopy(_cell(r))),
        "cell identity is duplicate or inconsistent",
        id="cells-duplicate",
    ),
    pytest.param(
        lambda r: _cell(r).update(extra=1), "cell has an invalid field roster", id="cell-roster"
    ),
    pytest.param(
        lambda r: _cell(r).update(difficulty="impossible"),
        "cell identity is duplicate or inconsistent",
        id="cell-identity",
    ),
    pytest.param(
        lambda r: _cell(r).update(prior_kind="other_prior"),
        "cell identity is duplicate or inconsistent",
        id="cell-factor",
    ),
    pytest.param(
        lambda r: _cell(r).update(requested_independent_groups=-1),
        "counts must be nonnegative integers",
        id="cell-count-type",
    ),
    pytest.param(
        lambda r: _cell(r).update(shortfall=5),
        "cell shortfall is inconsistent",
        id="cell-shortfall",
    ),
    pytest.param(
        lambda r: _cell(r).update(selected_independent_groups=0),
        "cell shortfall is inconsistent",
        id="cell-selected-count",
    ),
    pytest.param(
        lambda r: _cell(r).update(selected_candidate_ids=["unknown|id"] * 2),
        "selected roster is inconsistent",
        id="cell-selected-ids",
    ),
    pytest.param(
        lambda r: _cell(r).update(target_population_probability=1.5),
        "population probability",
        id="cell-probability",
    ),
    pytest.param(
        lambda r: _cell(r).update(target_population_probability=0.5),
        "importance weight is inconsistent",
        id="probabilities-sum",
    ),
    pytest.param(
        lambda r: r.update(selected_unique_independent_group_count=99),
        "unique independent-group count is inconsistent",
        id="unique-group-count",
    ),
    pytest.param(
        lambda r: r.update(
            status="SAMPLE_ROSTER_COMPLETE", selected_candidate_count=0, selected_candidates=[]
        ),
        "cell selected roster is inconsistent",
        id="status-mismatch",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), CASES)
def test_audit_rejects_altered_sample_reports(mutate, message: str) -> None:
    sample = _sample()
    report = copy.deepcopy(sample)
    mutate(report)

    with pytest.raises(ValueError, match=message):
        audit_pilot_outcomes(report, _outcomes(sample))


def test_audit_accepts_the_unaltered_report() -> None:
    sample = _sample()
    assert audit_pilot_outcomes(copy.deepcopy(sample), _outcomes(sample))["status"]
