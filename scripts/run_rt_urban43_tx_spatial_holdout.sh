#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_python="${repo_root}/.venv/bin/python3"
if [[ ! -x "${default_python}" ]]; then
  default_python="python3"
fi
python_bin="${SIONNA_PYTHON:-${default_python}}"
output_dir="${1:-outputs/rt-urban43-tx-spatial-holdout-20261022-v3}"
observation_device="${OBSERVATION_DEVICE:-cuda:0}"
if [[ "${output_dir}" != /* ]]; then
  output_dir="${repo_root}/${output_dir}"
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite TX spatial-holdout run: ${output_dir}" >&2
  exit 2
fi
if ! "${python_bin}" -c 'import sionna.rt' >/dev/null 2>&1; then
  echo "Set SIONNA_PYTHON to a Python environment with Sionna RT 2.2 installed." >&2
  exit 2
fi
mkdir -p "${output_dir}"
cd "${repo_root}"

scene="${repo_root}/scenes/generated/research_scenes/city43/urban_00043.xml"
datasets=()
run_tx() {
  local transmitter_id="$1"
  local tx_x="$2"
  local rx_x="$3"
  local seed="$4"
  local dataset="${output_dir}/${transmitter_id}.npz"
  "${python_bin}" -m loslab rt-data \
    --output "${dataset}" \
    --seed "${seed}" \
    --scene "${scene}" \
    --tx-id "${transmitter_id}" \
    --tx "${tx_x}" -55 18 \
    --center "${rx_x}" 0 1.5 \
    --radius 25 \
    --per-class 20 \
    --max-attempts 2000 \
    --samples-per-src 20000 \
    --block-size 20 \
    --max-depth 3 \
    --mitsuba-variant llvm_ad_mono_polarized
  datasets+=("${dataset}")
}

# A low-cost native preflight verified two examples per class for all three
# configurations with the default LOS-multipath filter at 20k rays/source.
# RX disks have 30 m gaps (centers 80 m apart, radius 25 m), eliminating
# cross-domain receiver proximity below the disk-separation distance. TX and RX
# regions vary together, so this tests a joint location shift, not TX alone.
run_tx west_x3 -60 -80 43111
run_tx center_x0 0 0 43112
run_tx east_x3 60 80 43113

observation="${output_dir}/nr-dmrs-observations.npz"
"${python_bin}" -m loslab nr-observe \
  --input "${datasets[@]}" \
  --output "${observation}" \
  --snr-db 12 \
  --seed 20261022 \
  --device "${observation_device}" \
  --n-size-bwp 6 \
  --subcarrier-spacing-khz 30

for transmitter_id in west_x3 center_x0 east_x3; do
  "${python_bin}" -m loslab experiment \
    --input "${observation}" \
    --output "${output_dir}/holdout-${transmitter_id}" \
    --holdout "rt_urban_00043_${transmitter_id}" \
    --methods logreg rf raw_mlp \
    --seeds 11 23 47 59 71 83 97 \
    --bootstrap 200 \
    --device cpu \
    --observation precomputed
done

"${python_bin}" scripts/summarize_rt_tx_holdout.py \
  "${output_dir}" "${output_dir}/tx-holdout-summary-audited.json"
echo "Saved three-TX, disjoint-RX-region experiment to ${output_dir}"
