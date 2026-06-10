#!/bin/bash
# Activate syndiff conda env and exec the stage command on Condor execute nodes.
set -eo pipefail

# Condor submit uses getenv=false, so HOME is often unset on execute nodes.
if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6)"
  export HOME
fi
if [[ -z "${HOME:-}" || ! -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  echo "condor_wrapper: cannot find miniforge3 under HOME=${HOME:-<unset>}" >&2
  exit 1
fi

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate syndiff
exec "$@"
