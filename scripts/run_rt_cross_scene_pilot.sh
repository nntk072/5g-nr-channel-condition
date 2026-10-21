#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-outputs/rt-cross-scene-repro-20261021}"
if [[ "${output_dir}" != /* ]]; then
  output_dir="${repo_root}/${output_dir}"
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing pilot output: ${output_dir}" >&2
  exit 2
fi
mkdir -p "${output_dir}"
cd "${repo_root}"

python -m loslab rt-data \
  --output "${output_dir}/box_one_screen.npz" \
  --seed 42 --scene box_one_screen --tx -2 0 2.5 --center 0 0 2.5 \
  --radius 4 --per-class 20 --max-attempts 600 --samples-per-src 200 \
  --block-size 5 --max-depth 2 --mitsuba-variant llvm_ad_mono_polarized

python -m loslab rt-data \
  --output "${output_dir}/simple_street_canyon.npz" \
  --seed 142 --scene simple_street_canyon --tx 0 0 25 --center 0 0 1.5 \
  --radius 30 --per-class 20 --max-attempts 600 --samples-per-src 200 \
  --block-size 20 --max-depth 2 --mitsuba-variant llvm_ad_mono_polarized

python -m loslab rt-data \
  --output "${output_dir}/etoile.npz" \
  --seed 1021 --scene etoile --tx 232.8 -95.5 17 --center 211 -180 1.5 \
  --radius 60 --per-class 20 --max-attempts 600 --samples-per-src 200 \
  --block-size 20 --max-depth 2 --mitsuba-variant llvm_ad_mono_polarized

datasets=(
  "${output_dir}/box_one_screen.npz"
  "${output_dir}/simple_street_canyon.npz"
  "${output_dir}/etoile.npz"
)
for holdout in rt_box_one_screen rt_simple_street_canyon rt_etoile; do
  python -m loslab experiment \
    --input "${datasets[@]}" \
    --output "${output_dir}/run-holdout-${holdout}" \
    --holdout "${holdout}" \
    --methods rf mlp logreg \
    --seeds 11 23 47 59 71 83 97 \
    --bootstrap 200 --device cpu --observation comb_ls \
    --pilot-spacing 4 --pilot-snr-db 12 --observation-seed 2026
done
