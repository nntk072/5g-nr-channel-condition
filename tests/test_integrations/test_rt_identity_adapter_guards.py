"""RT identity to UMCD provenance adapter: refuses incomplete, tampered or unbound identities."""

from __future__ import annotations

import copy

import pytest
from integrations.sionna.rt_identity import (
    canonical_hash,
    rt_run_identity,
    umcd_provenance_from_rt_identity,
)
from tests.test_integrations.test_rt_identity import _inputs


def _identity():
    inputs = _inputs()
    inputs["package_source_sha256"] = "b" * 64
    return rt_run_identity(**inputs), inputs


def _convert(identity, **overrides):
    kwargs = {
        "record_id": "rt-generation-guard",
        "subject_layer": "oracle_paths",
        "source_family": "F03-RT-STATIC",
        "producer": "Sionna 2.2 PathSolver",
    }
    kwargs.update(overrides)
    return umcd_provenance_from_rt_identity(identity, **kwargs)


def test_incomplete_identities_cannot_become_provenance() -> None:
    identity, _ = _identity()
    incomplete = copy.deepcopy(dict(identity))
    incomplete["complete"] = False
    with pytest.raises(ValueError, match="only complete RT run identities"):
        _convert(incomplete)


def test_identity_inputs_must_match_their_cache_key() -> None:
    identity, _ = _identity()
    tampered = copy.deepcopy(dict(identity))
    tampered["inputs"]["seed"] = 999999
    with pytest.raises(ValueError, match="do not match their cache key"):
        _convert(tampered)


def test_complete_identity_requires_inputs_and_cache_key() -> None:
    with pytest.raises(ValueError, match="must contain inputs and a cache key"):
        _convert({"schema": "loslab.rt_run_identity.v2", "complete": True})


def test_reproducibility_inputs_must_be_present_and_typed() -> None:
    identity, _ = _identity()
    broken = copy.deepcopy(dict(identity))
    del broken["inputs"]["solver_config"]
    broken["cache_key_sha256"] = canonical_hash(dict(broken["inputs"]))
    with pytest.raises(ValueError, match="missing required reproducibility inputs"):
        _convert(broken)


def test_runtime_versions_must_be_string_valued() -> None:
    identity, _ = _identity()
    numeric = copy.deepcopy(dict(identity))
    numeric["inputs"]["runtime_versions"]["numpy"] = 2
    numeric["cache_key_sha256"] = canonical_hash(dict(numeric["inputs"]))
    with pytest.raises(ValueError, match="must be string-valued"):
        _convert(numeric)


def test_material_override_must_be_a_mapping_and_match_the_bound_identity() -> None:
    identity, inputs = _identity()
    with pytest.raises(TypeError, match="material_config override must be a mapping"):
        _convert(identity, material_config=["not", "a", "mapping"])
    other = dict(inputs["material_config"])
    other["extra-material"] = {"name": "unbound"}
    with pytest.raises(ValueError, match="does not match the cache-key-bound"):
        _convert(identity, material_config=other)
