#!/usr/bin/env bash
# Pack lean train-only code + FitBundle for Google Colab (jax/optax/numpy only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUNDLE_DIR="${1:-$ROOT/dev/forward_epsf_wcs/output/bundles/orbit1_half_mag710}"
OUT_ZIP="${2:-$ROOT/dev/forward_epsf_wcs/output/colab/colab_forward_epsf_wcs_train.zip}"
if [[ "$OUT_ZIP" != /* ]]; then
  OUT_ZIP="$ROOT/$OUT_ZIP"
fi

BUNDLE_NPZ="$BUNDLE_DIR/fit_bundle.npz"
BUNDLE_META="$BUNDLE_DIR/fit_bundle_meta.json"
if [[ ! -f "$BUNDLE_NPZ" ]]; then
  echo "missing $BUNDLE_NPZ — run export_fit_bundle first" >&2
  exit 1
fi
if [[ ! -f "$BUNDLE_META" ]]; then
  echo "missing $BUNDLE_META" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$STAGE/dev/forward_epsf_wcs"
touch "$STAGE/dev/__init__.py"
# Train-path modules only: no workspace preparation, astronomy catalogs, PRF,
# pandas, scipy, astropy, or image I/O.
for f in \
  __init__.py _bootstrap.py \
  train_from_bundle.py isolated_stage_runner.py gpu_job_runner.py training_state.py gpu_preflight.py train_loop.py fit_bundle.py fit.py loss.py \
  epsf_model.py cheb_wcs.py groups.py flux_solve.py temporal.py runtime.py \
  packed_support.py stamp_reject.py
do
  src="$ROOT/dev/forward_epsf_wcs/$f"
  if [[ ! -f "$src" ]]; then
    echo "missing required train module: $src" >&2
    exit 1
  fi
  cp "$src" "$STAGE/dev/forward_epsf_wcs/$f"
done

mkdir -p "$STAGE/bundle"
cp "$BUNDLE_NPZ" "$STAGE/bundle/fit_bundle.npz"
cp "$BUNDLE_META" "$STAGE/bundle/fit_bundle_meta.json"
if [[ -f "$BUNDLE_DIR/release_provenance.json" ]]; then
  cp "$BUNDLE_DIR/release_provenance.json" "$STAGE/bundle/release_provenance.json"
fi

# Optional warm params must be explicitly supplied.  Never silently package a
# checkpoint from a different region, magnitude cut, or frame geometry.
PARAMS_LATEST="${PARAMS_LATEST:-}"
if [[ -n "$PARAMS_LATEST" && -f "$PARAMS_LATEST" ]]; then
  cp "$PARAMS_LATEST" "$STAGE/bundle/params_latest.npz"
fi

printf '%s\n' 'numpy==2.4.6' 'jax==0.9.2' 'jaxlib==0.9.2' 'optax==0.2.8' \
  > "$STAGE/RUNTIME_REQUIREMENTS.txt"
GPU_SCRIPT="${GPU_SCRIPT:-$ROOT/dev/forward_epsf_wcs/scripts/run_gpu_fullccd_mag711_orbit.sh}"
if [[ ! -f "$GPU_SCRIPT" ]]; then
  echo "missing GPU script: $GPU_SCRIPT" >&2
  exit 1
fi
cp "$GPU_SCRIPT" "$STAGE/RUN_GPU.sh"
chmod +x "$STAGE/RUN_GPU.sh"
# Durable payload inventory for upload verification, including the launcher.
(cd "$STAGE" && sha256sum bundle/* dev/forward_epsf_wcs/*.py \
  RUNTIME_REQUIREMENTS.txt RUN_GPU.sh > SHA256SUMS)

rm -f "$OUT_ZIP"
(cd "$STAGE" && zip -qr "$OUT_ZIP" dev bundle SHA256SUMS RUNTIME_REQUIREMENTS.txt RUN_GPU.sh)
unzip -tq "$OUT_ZIP"
ls -lh "$OUT_ZIP"
echo "packed -> $OUT_ZIP"
