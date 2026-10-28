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

# Python's tempfile silently falls back to /tmp (4.9 GB, shared) when TMPDIR
# does not exist, so make sure the scratch dirs set by the submit file exist.
for _d in "${TMPDIR:-}" "${JOBLIB_TEMP_FOLDER:-}" "${MPLCONFIGDIR:-}" "${DASK_TEMPORARY_DIRECTORY:-}"; do
  [[ -n "${_d}" ]] && mkdir -p "${_d}" || true
done

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
exec "$@"
