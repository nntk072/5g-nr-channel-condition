#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SIONNA_PYTHON:-python}"
samples_per_src_override="${SAMPLES_PER_SRC:-}"
output_dir="${1:-/tmp/loslab-pilot/rt-six-scene-holdout-20261021}"
if [[ "${output_dir}" != /* ]]; then
  output_dir="${repo_root}/${output_dir}"
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing six-scene run: ${output_dir}" >&2
  exit 2
fi
if [[ -n "${samples_per_src_override}" && ! "${samples_per_src_override}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLES_PER_SRC override must be a positive integer." >&2
  exit 2
fi
if ! "${python_bin}" -c 'import sionna.rt' >/dev/null 2>&1; then
  echo "Set SIONNA_PYTHON to a Python environment with Sionna RT 2.2 installed." >&2
  exit 2
fi
mkdir -p "${output_dir}"
cd "${repo_root}"

run_scene() {
  local output_name="$1"
  local scene="$2"
  local seed="$3"
  local tx_x="$4" tx_y="$5" tx_z="$6"
  local center_x="$7" center_y="$8" center_z="$9"
  local radius="${10}" block_size="${11}"
  local scene_samples_per_src="${12}"
  if [[ -n "${samples_per_src_override}" ]]; then
    scene_samples_per_src="${samples_per_src_override}"
  fi
  if [[ ! "${scene_samples_per_src}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Scene ray budget must be a positive integer: ${scene_samples_per_src}" >&2
    exit 2
  fi
  "${python_bin}" -m loslab rt-data --pilot \
    --output "${output_dir}/${output_name}.npz" \
    --seed "${seed}" --scene "${scene}" \
    --tx "${tx_x}" "${tx_y}" "${tx_z}" \
    --center "${center_x}" "${center_y}" "${center_z}" \
    --radius "${radius}" --per-class 20 --max-attempts 800 \
    --samples-per-src "${scene_samples_per_src}" --block-size "${block_size}" --max-depth 2 \
    --mitsuba-variant llvm_ad_mono_polarized
}

run_scene box_one_screen box_one_screen 42 -2 0 2.5 0 0 2.5 4 5 20000
run_scene simple_street_canyon simple_street_canyon 142 0 0 25 0 0 1.5 30 20 1000000
run_scene etoile etoile 1021 232.8 -95.5 17 211 -180 1.5 60 20 5000000

scene_root="${repo_root}/scenes/generated/research_scenes"
run_scene urban_00041 "${scene_root}/city41/urban_00041.xml" 41001 0 -55 18 0 0 1.5 115 55 20000
run_scene urban_00042 "${scene_root}/city42/urban_00042.xml" 42001 0 -55 18 0 0 1.5 115 55 100000
run_scene urban_00043 "${scene_root}/city43/urban_00043.xml" 43001 0 -55 18 0 0 1.5 115 55 20000

datasets=(
  "${output_dir}/box_one_screen.npz"
  "${output_dir}/simple_street_canyon.npz"
  "${output_dir}/etoile.npz"
  "${output_dir}/urban_00041.npz"
  "${output_dir}/urban_00042.npz"
  "${output_dir}/urban_00043.npz"
)
domains=(
  rt_box_one_screen
  rt_simple_street_canyon
  rt_etoile
  rt_urban_00041
  rt_urban_00042
  rt_urban_00043
)
for holdout in "${domains[@]}"; do
  "${python_bin}" -m loslab experiment \
    --input "${datasets[@]}" \
    --output "${output_dir}/holdout-${holdout}" \
    --holdout "${holdout}" \
    --methods rf mlp logreg \
    --seeds 11 23 47 59 71 83 97 \
    --bootstrap 200 --device cpu --observation comb_ls \
    --pilot-spacing 4 --pilot-snr-db 12 --observation-seed 20261026
done

echo "Saved six-scene datasets and held-out evaluations to ${output_dir}"
