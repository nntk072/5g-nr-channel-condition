"""Schema paths, size limits and limitation text for acquisition packages."""

from __future__ import annotations

from pathlib import Path

_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "datasets/schema"


SCHEMA_PATHS = {
    "g3.03-acquisition-package.v0.2": _SCHEMA_ROOT / "acquisition-package.v0.2.schema.json",
    "g3.03-acquisition-package.v0.3": _SCHEMA_ROOT / "acquisition-package.v0.3.schema.json",
    "g3.03-acquisition-package.v0.4": _SCHEMA_ROOT / "acquisition-package.v0.4.schema.json",
}


V03_PLUS_SCHEMA_VERSIONS = frozenset(
    {
        "g3.03-acquisition-package.v0.3",
        "g3.03-acquisition-package.v0.4",
    }
)


EVALUATOR_INPUT_SCHEMA_PATHS = {
    "loslab.g3_material_calibration_predictions.v1": _SCHEMA_ROOT
    / "material-calibration-predictions.v1.schema.json",
    "loslab.g3_material_calibration_predictions.v2": _SCHEMA_ROOT
    / "material-calibration-predictions.v2.schema.json",
}


MATERIAL_MAPPING_SCHEMA_PATH = _SCHEMA_ROOT / "material-mapping.v1.schema.json"


MAX_PACKAGE_BYTES = 16 * 1024 * 1024


MAX_SCHEMA_BYTES = 4 * 1024 * 1024


MAX_INVENTORY_FILES = 10_000


MAX_FILE_BYTES = 256 * 1024 * 1024


MAX_TOTAL_FILE_BYTES = 1024 * 1024 * 1024


MAX_JSON_ARTIFACT_BYTES = 32 * 1024 * 1024


CHUNK_BYTES = 1024 * 1024


LIMITATIONS = [
    "Schema-valid proposal fixtures are synthetic examples and are never measurement evidence.",
    "This consistency report cannot establish scientific validity, physical independence, "
    "calibration truth, preregistration timing, legal rights sufficiency, signature authenticity, "
    "or evaluator authorship.",
    "Only v0.3+ semantic checks validate full-sphere angular grids and their solid-angle cells; v0.2 remains a legacy structural proposal format.",
    "For the current v2 material evaluator, reported results are recomputed from the bound scorer input. Legacy v1 evaluator outputs receive structural checks only.",
    "The hash-bound material map checks package-internal specimen and evaluator IDs but cannot authenticate material identity or evidence-file contents.",
    "Acquisition-package v0.4 recomputes the declared RMS of same-grid noise-only CFR estimates from exact bounded NPY bytes, but cannot establish that the capture is representative or calibrated.",
]
