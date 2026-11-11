"""Measure process RSS for a bounded synthetic NR store write/read pass.

This is an engineering diagnostic, not a corpus-scale memory guarantee or
qualification run. The process builds records, writes them through the public
content-addressed store API, and consumes every deterministic shard before
reporting Linux/macOS process high-water RSS.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from datasets.nr import NRChannelRecord, NRObservation, TargetDefinition
from datasets.nr_store import iter_nr_record_shards, write_nr_record
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _record(sample_id: str, *, symbols: int, subcarriers: int) -> NRChannelRecord:
    shape = (1, 1, 1, 1, 1, symbols, subcarriers)
    observation = NRObservation(
        sample_id=sample_id,
        received_grid=np.zeros((1, 1, 1, symbols, subcarriers), dtype=np.complex64),
        channel_estimate=NRChannelEstimate(
            h_hat=np.zeros(shape, dtype=np.complex64),
            error_variance=np.zeros(shape, dtype=np.float32),
            pilot_mask=np.zeros((1, 1, symbols, subcarriers), dtype=bool),
            metadata={"receiver": "bounded-synthetic-rss-diagnostic"},
        ),
        fft_bin_indices=tuple(range(subcarriers)),
        receiver_config=NRPUSCHConfig(n_size_bwp=max(1, (subcarriers + 11) // 12)),
    )
    return NRChannelRecord(
        observation=observation,
        targets={"synthetic_scalar": np.asarray(1.0, dtype=np.float32)},
        target_masks={},
        provenance={"source": "synthetic_memory_diagnostic", "production": False},
        target_definitions={
            "synthetic_scalar": TargetDefinition.versioned(
                semantic="synthetic memory diagnostic scalar",
                units="unitless",
                source="synthetic_memory_diagnostic",
                reference="fixed value used only to exercise serialization",
                aggregation_rule="one scalar per record",
                definition_version="synthetic-memory-diagnostic.v1",
            )
        },
    )


def _current_rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("Linux process status does not report VmRSS")
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class _RssSampler:
    def __init__(self) -> None:
        self.baseline = _current_rss_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.002):
            self.peak = max(self.peak, _current_rss_bytes())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> tuple[int, int]:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _current_rss_bytes())
        return self.baseline, self.peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--symbols", type=int, default=14)
    parser.add_argument("--subcarriers", type=int, default=24)
    parser.add_argument("--shard-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if min(args.records, args.symbols, args.subcarriers, args.shard_bytes) <= 0:
        parser.error("all numeric arguments must be positive")

    rss = _RssSampler()
    rss.start()
    with tempfile.TemporaryDirectory(prefix="nr-store-rss-") as temporary:
        root = Path(temporary)
        for index in range(args.records):
            write_nr_record(
                root,
                _record(
                    f"synthetic-{index:08d}",
                    symbols=args.symbols,
                    subcarriers=args.subcarriers,
                ),
            )
        read_count = sum(
            len(shard) for shard in iter_nr_record_shards(root, max_shard_bytes=args.shard_bytes)
        )
        stored_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    baseline_rss_bytes, process_peak_rss_bytes = rss.stop()

    print(
        json.dumps(
            {
                "schema": "synthetic.nr_store_rss_measurement.v1",
                "production": False,
                "records_written": args.records,
                "records_read": read_count,
                "symbols": args.symbols,
                "subcarriers": args.subcarriers,
                "shard_budget_bytes": args.shard_bytes,
                "store_disk_bytes": stored_bytes,
                "process_rss_before_store_bytes": baseline_rss_bytes,
                "process_peak_rss_bytes": process_peak_rss_bytes,
                "incremental_peak_rss_bytes": max(0, process_peak_rss_bytes - baseline_rss_bytes),
                "platform": platform.platform(),
                "scope": "one-process public NR store serialization and full deterministic read pass",
            },
            sort_keys=True,
        )
    )
    return 0 if read_count == args.records else 1


if __name__ == "__main__":
    raise SystemExit(main())
