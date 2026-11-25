from __future__ import annotations

from collections.abc import Sequence

import integrations.sionna.srs.srs_bwp_schedules as schedules
import pytest


def _multicell_arguments() -> dict[str, object]:
    resource = {
        "resource_set_id": 1,
        "resource_type": "aperiodic",
        "aperiodic_trigger": 1,
        "allocation": {
            "c_srs": 0,
            "transmission_comb": 2,
            "b_srs": 0,
            "b_hop": 0,
            "srs_symbol_count": 1,
        },
    }
    return {
        "request_list_index": 0,
        "request_list": ((1,),),
        "serving_cell_ids": ("cell-a",),
        "resource_sets_by_cell": {"cell-a": [resource]},
        "triggering_dci_slot_by_cell": {"cell-a": 12},
        "scheduled_cell_ids": ("cell-a",),
    }


def _stub_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    def selected(*_args: object, scheduled_cell_ids: Sequence[str], **_kwargs: object):
        return tuple((cell, (1,)) for cell in scheduled_cell_ids)

    monkeypatch.setattr(schedules, "nr_srs_multicell_request_resource_set_ids", selected)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"scheduled_cell_ids": "cell-a"}, TypeError, "scheduled_cell_ids must be an ordered"),
        ({"scheduled_cell_ids": ()}, ValueError, "must contain nonempty cell IDs"),
        ({"scheduled_cell_ids": ("ghost",)}, ValueError, "unique subset of serving_cell_ids"),
        (
            {
                "serving_cell_ids": ("cell-a", "cell-b"),
                "resource_sets_by_cell": {"cell-a": [], "cell-b": []},
                "triggering_dci_slot_by_cell": {"cell-b": 1, "cell-a": 1},
                "scheduled_cell_ids": ("cell-b", "cell-a"),
            },
            ValueError,
            "follow serving-cell order",
        ),
        ({"offset_combo_cell_ids": ("",)}, ValueError, "nonempty cell IDs"),
        (
            {
                "serving_cell_ids": ("cell-a", "cell-b"),
                "resource_sets_by_cell": {"cell-a": [], "cell-b": []},
                "triggering_dci_slot_by_cell": {"cell-a": 1},
                "offset_combo_cell_ids": ("cell-b", "cell-a"),
            },
            ValueError,
            "follow serving-cell order",
        ),
    ],
)
def test_multicell_schedule_rejects_cell_roster_guards(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    _stub_selection(monkeypatch)
    arguments = _multicell_arguments()
    arguments.update(changes)

    with pytest.raises(error, match=message):
        schedules.nr_srs_multicell_aperiodic_resource_set_schedules(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("record", "error", "message"),
    [
        ({"resource_set_id": True}, ValueError, "resource_set_id must be an integer"),
        ({"resource_set_id": 1, "allocation": "bad"}, ValueError, "requires an allocation mapping"),
    ],
)
def test_multicell_schedule_rejects_selected_set_record_errors(
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    _stub_selection(monkeypatch)
    arguments = _multicell_arguments()
    arguments["resource_sets_by_cell"] = {"cell-a": [record]}

    with pytest.raises(error, match=message):
        schedules.nr_srs_multicell_aperiodic_resource_set_schedules(**arguments)  # type: ignore[arg-type]


def test_multicell_schedule_skips_nonmapping_unselected_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_selection(monkeypatch)
    arguments = _multicell_arguments()
    arguments["resource_sets_by_cell"] = {
        "cell-a": ["unselected", *_multicell_arguments()["resource_sets_by_cell"]["cell-a"]]
    }

    result = schedules.nr_srs_multicell_aperiodic_resource_set_schedules(**arguments)  # type: ignore[arg-type]

    assert result[0][0] == "cell-a"
    assert result[0][1][0].resource_set_id == 1


def test_aperiodic_frequency_adapter_rejects_unexpected_inactive_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        schedules, "nr_srs_resource_frequency_allocations", lambda *_args, **_kwargs: None
    )
    with pytest.raises(RuntimeError, match="allocation unexpectedly resolved inactive"):
        schedules.nr_srs_aperiodic_resource_schedule(
            10,
            c_srs=0,
            transmission_comb=2,
            b_srs=0,
            b_hop=0,
            srs_symbol_count=1,
        )


class _ChangingSequence(Sequence):
    def __init__(self, first: list[object], second: list[object]) -> None:
        self._values = (first, second)
        self._iteration = 0

    def __getitem__(self, index: int) -> object:
        return self._values[0][index]

    def __len__(self) -> int:
        return len(self._values[0])

    def __iter__(self):
        values = self._values[min(self._iteration, 1)]
        self._iteration += 1
        return iter(values)


@pytest.mark.parametrize(
    ("second_pass_resource", "error", "message"),
    [
        ("not a mapping", TypeError, r"resources\[0\] must be a mapping"),
        ({"resource_id": True}, TypeError, r"resources\[0\].resource_id must be an integer"),
        ({"resource_id": -1}, ValueError, "unique nonnegative integers"),
    ],
)
def test_periodic_resource_builder_rechecks_mutating_resource_roster(
    second_pass_resource: object, error: type[Exception], message: str
) -> None:
    allocation = {
        "c_srs": 0,
        "transmission_comb": 2,
        "b_srs": 0,
        "b_hop": 0,
        "srs_symbol_count": 1,
    }
    resources = _ChangingSequence(
        [{"resource_id": 1, "allocation": allocation}],
        [second_pass_resource],
    )
    with pytest.raises(error, match=message):
        schedules.nr_srs_periodic_resource_set_schedule(
            0, 1, resources, periodicity_slots=1, slot_offset=0
        )


def test_aperiodic_set_builder_skips_nonmapping_unselected_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        schedules, "nr_srs_triggered_resource_set_ids", lambda *_args, **_kwargs: (1,)
    )
    resource = _multicell_arguments()["resource_sets_by_cell"]["cell-a"][0]
    result = schedules.nr_srs_aperiodic_resource_set_schedules(
        10,
        1,
        ["unselected", resource],
        dci_format="0_1",
    )
    assert result[0].resource_set_id == 1
