#!/bin/bash
# Activate syndiff conda env and exec the stage command on Condor execute nodes.
set -eo pipefail

# Condor submit uses getenv=false, so HOME is often unset on execute nodes.
if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6)"
  export HOME
fi

CONDA_ENV="${SYNDIFF_CONDA_ENV:-syndiff}"
CONDA_SH="${SYNDIFF_CONDA_SH:-}"

if [[ -z "${CONDA_SH}" ]]; then
  for candidate in \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      CONDA_SH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
  echo "condor_wrapper: cannot find conda.sh (SYNDIFF_CONDA_SH unset; HOME=${HOME:-<unset>})" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
exec "$@"
