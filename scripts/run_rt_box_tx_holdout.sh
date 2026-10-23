#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SIONNA_PYTHON:-python}"
output_dir="${1:-/tmp/loslab-pilot/rt-box-tx-holdout-20261022}"
observation_device="${OBSERVATION_DEVICE:-cuda:0}"
if [[ "${output_dir}" != /* ]]; then
  output_dir="${repo_root}/${output_dir}"
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite transmitter-holdout run: ${output_dir}" >&2
  exit 2
fi
if ! "${python_bin}" -c 'import sionna.rt' >/dev/null 2>&1; then
  echo "Set SIONNA_PYTHON to a Python environment with Sionna RT 2.2 installed." >&2
  exit 2
fi
mkdir -p "${output_dir}"
cd "${repo_root}"

datasets=()
run_tx() {
  local transmitter_id="$1"
  local tx_x="$2"
  local tx_y="$3"
  local seed="$4"
  local dataset="${output_dir}/${transmitter_id}.npz"
  "${python_bin}" -m loslab rt-data --pilot \
    --output "${dataset}" \
    --seed "${seed}" \
    --scene box_one_screen \
    --tx-id "${transmitter_id}" \
    --tx "${tx_x}" "${tx_y}" 2.5 \
    --center 2 0 2.5 \
    --radius 4 \
    --per-class 20 \
    --max-attempts 2000 \
    --samples-per-src 20000 \
    --block-size 2 \
    --max-depth 2 \
    --mitsuba-variant llvm_ad_mono_polarized
  datasets+=("${dataset}")
}

# Three distinct points on the same side of the screen, with identical RX
# sampling support and solver settings. This isolates TX-location sensitivity
# within one controlled geometry; it is not cross-scene or field validation.
run_tx west_x3 -3 0 30301
run_tx west_x2 -2 0 30302
run_tx west_x1 -1 0 30303

observation="${output_dir}/nr-dmrs-observations.npz"
"${python_bin}" -m loslab nr-observe --pilot \
  --input "${datasets[@]}" \
  --output "${observation}" \
  --snr-db 12 \
  --seed 20261022 \
  --device "${observation_device}" \
  --n-size-bwp 6 \
  --subcarrier-spacing-khz 30

for transmitter_id in west_x3 west_x2 west_x1; do
  "${python_bin}" -m loslab experiment \
    --input "${observation}" \
    --output "${output_dir}/holdout-${transmitter_id}" \
    --holdout "rt_box_one_screen_${transmitter_id}" \
    --methods logreg rf raw_mlp \
    --seeds 11 23 47 59 71 83 97 \
    --bootstrap 200 \
    --device cpu \
    --observation precomputed
done

echo "Saved three-transmitter LOSO datasets, DMRS observations, and evaluations to ${output_dir}"
