"""End-to-end leakage-protected experiment with exact dataset and run provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.layout import RunLayout
from artifacts.manifest import mark_complete, run_manifest
from artifacts.provenance import (
    atomic_json,
    canonical_hash,
    package_source_sha256,
    sha256_file,
    versions,
)
from configuration.loader import resolve_config
from datasets.channel import ChannelDataset
from evaluation.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    holdout_domain,
    in_domain_split,
    threshold_from_validation,
)
from evaluation.statistics import paired_seed_delta
from learning import fit_coral_mlp, fit_sklearn
from propagation.physics import observable_features
from receivers.pilots import comb_ls_estimate


def _label_semantics_by_source(data: ChannelDataset) -> str:
    semantics: dict[str, set[str]] = {}
    for source, metadata in zip(data.source, data.metadata, strict=True):
        source_name = str(source)
        label_semantic = metadata.get("geometry_label") or metadata.get("label_semantics")
        if label_semantic is None:
            label_semantic = (
                "SYNTHETIC_DEBUG_CONSTRUCTED"
                if source_name == "synthetic_debug"
                else f"UNSPECIFIED({source_name})"
            )
        semantics.setdefault(source_name, set()).add(str(label_semantic))
    if len(semantics) == 1:
        values = next(iter(semantics.values()))
        if len(values) == 1:
            return next(iter(values))
    return "; ".join(
        f"{source}={'|'.join(sorted(values))}" for source, values in sorted(semantics.items())
    )


def _normalized_label_semantics(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "GEOMETRIC_DIRECT_PATH_ORACLE": "geometric_direct_path_oracle",
        "PROFILE_PROXY": "profile_proxy",
        "SYNTHETIC_DEBUG_CONSTRUCTED": "synthetic_debug_direct_component",
        "SYNTHETIC_DEBUG_DIRECT_COMPONENT": "synthetic_debug_direct_component",
        "UNSPECIFIED": "unspecified",
        "UNKNOWN": "unspecified",
        "UNKNOWN_LEGACY": "unspecified",
    }
    return aliases.get(normalized, value.strip().lower().replace(" ", "_"))


def _resolved_label_semantics(data: ChannelDataset) -> str:
    """Resolve one target meaning and refuse to score mixed or unknown legacy labels."""
    row_semantics: set[str] = set()
    has_unresolved_row = False
    for source, metadata in zip(data.source, data.metadata, strict=True):
        raw = metadata.get("geometry_label") or metadata.get("label_semantics")
        if raw is None and str(source) == "synthetic_debug":
            raw = "synthetic_debug_direct_component"
        if raw is None:
            has_unresolved_row = True
        else:
            normalized = _normalized_label_semantics(str(raw))
            if normalized == "unspecified":
                has_unresolved_row = True
            else:
                row_semantics.add(normalized)

    declared = data.label_semantics
    if declared != "unspecified_legacy":
        if row_semantics and row_semantics != {declared}:
            raise ValueError(
                "dataset label_semantics conflicts with per-sample label metadata: "
                f"declared={declared!r}, observed={sorted(row_semantics)}"
            )
        return declared
    if has_unresolved_row or len(row_semantics) != 1:
        raise ValueError(
            "cannot evaluate unspecified legacy labels without one explicit, consistent "
            "label meaning; regenerate the dataset or provide per-sample label metadata"
        )
    return next(iter(row_semantics))


def _precomputed_observation_semantics(data: ChannelDataset) -> str:
    semantics = sorted(
        {
            str(metadata.get("receiver_observation"))
            for metadata in data.metadata
            if metadata.get("receiver_observation")
        }
    )
    return "; ".join(semantics) if semantics else "precomputed channel dataset observations"


def run_experiment(
    paths: Sequence[str | Path],
    output: str | Path,
    *,
    holdout: str | None = None,
    methods: Sequence[str] = ("rf", "mlp", "coral", "coral0"),
    seeds: Sequence[int] = (11, 23, 47, 59, 71, 83, 97, 101, 109, 127),
    bootstrap: int = 200,
    include_phase: bool = False,
    observation: str = "ideal_cfr",
    pilot_spacing: int = 4,
    pilot_snr_db: float = 12.0,
    observation_seed: int = 2026,
    device: str = "cpu",
    coral_penalty: float = 0.1,
) -> dict:
    if not paths:
        raise ValueError("No datasets supplied")
    data = ChannelDataset.concatenate([ChannelDataset.load(p) for p in paths])
    resolved_label_semantics = _resolved_label_semantics(data)
    if observation == "ideal_cfr":
        observable = data.h
    elif observation == "comb_ls":
        observable = comb_ls_estimate(
            data.h, spacing=pilot_spacing, snr_db=pilot_snr_db, seed=observation_seed
        )
    elif observation == "precomputed":
        observable = data.h
    else:
        raise ValueError("observation must be ideal_cfr, comb_ls, or precomputed")
    x = observable_features(observable, include_phase=include_phase)
    layout = RunLayout.at(output)
    layout.ensure()
    output = layout.root
    config = {
        "holdout": holdout,
        "methods": list(methods),
        "seeds": list(seeds),
        "bootstrap": bootstrap,
        "include_phase": include_phase,
        "device": device,
        "coral_penalty": coral_penalty,
        "observation": observation,
        "pilot_spacing": pilot_spacing,
        "pilot_snr_db": pilot_snr_db,
        "observation_seed": observation_seed,
        "feature_version": "logmag+pdp+moments-v1",
    }
    dependency_versions = versions()
    manifest = run_manifest(
        output.name,
        config,
        legacy_schema="loslab.run.v1",
        config_sha256=canonical_hash(config),
        inputs=[{"path": str(p), "sha256": sha256_file(p)} for p in paths],
        dependencies=dependency_versions,
        source_revision=dependency_versions.get("git_sha"),
        package_source_sha256=package_source_sha256(),
        n_samples=len(data.y),
        source_counts={str(s): int(np.sum(data.source == s)) for s in np.unique(data.source)},
        domain_counts={str(s): int(np.sum(data.domain == s)) for s in np.unique(data.domain)},
        label_semantics=resolved_label_semantics,
        label_semantics_by_source=_label_semantics_by_source(data),
        observation_semantics={
            "ideal_cfr": "ideal channel, diagnostic upper bound",
            "comb_ls": "generic noisy comb pilot estimate, NOT 3GPP DMRS",
            "precomputed": _precomputed_observation_semantics(data),
        }[observation],
        study_status=(
            "DEBUG_ONLY" if "synthetic_debug" in data.source else "SIMULATION_ONLY_UNVALIDATED"
        ),
    )
    atomic_json(layout.config / "resolved.json", resolve_config(config))
    atomic_json(output / "manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = (
            holdout_domain(data, holdout, seed=seed)
            if holdout
            else in_domain_split(data, seed=seed)
        )
        assert not (set(data.group[train]) & set(data.group[val]))
        assert not (set(data.group[test]) & (set(data.group[train]) | set(data.group[val])))
        for method in methods:
            if method in ("rf", "mlp", "logreg"):
                clf = fit_sklearn(x[train], data.y[train], seed=seed, method=method)
                extra = {}
            elif method in ("coral", "coral0"):
                clf, extra = fit_coral_mlp(
                    x[train],
                    data.y[train],
                    data.domain[train],
                    x[val],
                    data.y[val],
                    seed=seed,
                    penalty=coral_penalty if method == "coral" else 0.0,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown method {method}")
            val_scores = clf.predict_proba(x[val])[:, 1]
            threshold = threshold_from_validation(data.y[val], val_scores)
            checkpoint = (
                layout.models / f"{method}_seed{seed}.pt"
                if method.startswith("coral")
                else layout.models / f"{method}_seed{seed}.joblib"
            )
            if method.startswith("coral"):
                clf.save(checkpoint, threshold=threshold, include_phase=include_phase)
            else:
                import joblib

                joblib.dump(clf, checkpoint)
                atomic_json(
                    layout.models / f"{method}_seed{seed}_modelinfo.json",
                    {
                        "format": "loslab.sklearn.v1",
                        "threshold": threshold,
                        "include_phase": include_phase,
                        "method": method,
                    },
                )
            prob = clf.predict_proba(x[test])[:, 1]
            metrics = binary_metrics(data.y[test], prob, threshold=threshold)
            ci = block_bootstrap_auc(
                data.y[test], prob, data.group[test], seed=seed, draws=bootstrap
            )
            result = {
                "seed": int(seed),
                "method": method,
                "metrics": metrics,
                "confidence": ci,
                "split": {
                    "indices_sha256": canonical_hash(
                        {
                            "train_idx": train.tolist(),
                            "validation_idx": val.tolist(),
                            "test_idx": test.tolist(),
                        }
                    ),
                    "train_size": len(train),
                    "val_size": len(val),
                    "test_size": len(test),
                    "train_groups": len(set(data.group[train])),
                    "validation_groups": len(set(data.group[val])),
                    "test_groups": len(set(data.group[test])),
                    "train_domains": sorted(set(data.domain[train].tolist())),
                    "test_domains": sorted(set(data.domain[test].tolist())),
                },
                "training": extra,
                "checkpoint_path": str(checkpoint),
            }
            results.append(result)
            # Keep split indices and probabilities for independently reproducing every metric.
            np.savez_compressed(
                layout.predictions / f"{method}_seed{seed}.npz",
                train_idx=train,
                val_idx=val,
                test_idx=test,
                y_test=data.y[test],
                y_val=data.y[val],
                test_group=np.asarray(data.group[test], dtype=str),
                test_domain=np.asarray(data.domain[test], dtype=str),
                p_test=prob,
                p_val=val_scores,
                threshold=np.float64(threshold),
            )
        atomic_json(output / "results.json", results)
        atomic_json(layout.evaluation / "results.json", results)
    aggregated = {}
    for method in methods:
        seq = [row["metrics"]["auroc"] for row in results if row["method"] == method]
        aggregated[method] = {
            "seeds": len(seq),
            "mean_test_auroc": float(np.mean(seq)),
            "std_test_auroc": float(np.std(seq, ddof=1)) if len(seq) > 1 else None,
        }
    comparisons = {}
    method_pairs = (
        ("rf", "mlp"),
        ("logreg", "mlp"),
        ("mlp", "coral"),
        ("coral0", "coral"),
        ("mlp", "coral0"),
    )
    for control, treatment in method_pairs:
        if control in methods and treatment in methods:
            comparisons[f"{treatment}_versus_{control}"] = paired_seed_delta(
                results, control, treatment, draws=5000
            )
    summary = {
        "schema": "loslab.summary.v1",
        "evidence_grade": manifest["study_status"],
        "paired_method_comparisons": comparisons,
        "aggregate_across_fixed_splits": aggregated,
        "warning": "Do not report synthetic-debug results or profile-proxy labels as real LOS measurements.",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(layout.evaluation / "summary.json", summary)
    complete_manifest = mark_complete(manifest)
    atomic_json(layout.manifest, complete_manifest)
    atomic_json(
        layout.complete,
        {
            "run_id": output.name,
            "manifest_schema": complete_manifest["schema"],
            "config_identity": complete_manifest["config_identity"],
        },
    )
    return summary
