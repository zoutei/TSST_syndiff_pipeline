#!/usr/bin/env bash
# After irregular/square bundles exist, CPU-measure VRAM vs stamp_chunk.
# Does NOT submit to Colab.
set -euo pipefail
cd /home/kshukawa/syndiff_pipeline
eval "$(conda shell.bash hook)"
conda activate syndiff
export PYTHONPATH=/home/kshukawa/syndiff_pipeline
export PYTHONUNBUFFERED=1

LOGDIR=dev/forward_epsf_wcs/output/logs
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/measure_vram_sweep.status"
say() { echo "[$(date -Is)] $*" | tee -a "$STATUS"; }

BUNDLES=(
  "dev/forward_epsf_wcs/output/bundles/fullccd_mag710_5x5"
  "dev/forward_epsf_wcs/output/bundles/orbit1_half_mag710_irregular"
  "dev/forward_epsf_wcs/output/bundles/fullccd_mag710_irreg_295"
  "dev/forward_epsf_wcs/output/bundles/fullccd_mag711_irreg_295"
  "dev/forward_epsf_wcs/output/bundles/fullccd_mag712_irreg_295"
  "dev/forward_epsf_wcs/output/bundles/fullccd_mag710_irreg_590"
)

# Skip unchunked (0) on full-CCD — host compile can be huge; T4 needs remat anyway.
for b in "${BUNDLES[@]}"; do
  name=$(basename "$b")
  if [[ ! -f "$b/fit_bundle.npz" ]]; then
    say "WAIT missing $name"
    continue
  fi
  out="$LOGDIR/measure_${name}.log"
  if [[ -f "$out" ]] && grep -q 'Summary (est_T4' "$out"; then
    say "SKIP measured $name"
    continue
  fi
  say "MEASURE $name -> $out"
  # Prefer 30,15,5; drop 0 for large T
  chunks=30,15,5
  python -m dev.forward_epsf_wcs._tmp_measure_full_step \
    --bundle "$b" \
    --stamp-chunks "$chunks" \
    --repeats 1 \
    --stage 3 \
    2>&1 | tee "$out"
  say "MEASURE done $name"
done
say "MEASURE_SWEEP_DONE"
