#!/usr/bin/env bash
# Sample host memory/load and write a JSON heartbeat to shared NFS.
set -euo pipefail

INTERVAL_S=60
OUT_DIR="${HOST_STATS_DIR:-/home/kshukawa/.syndiff/host_stats}"

usage() {
  cat <<'EOF'
Usage: host_sampler.sh [--out-dir DIR] [--interval SECONDS]

Run a loop that samples MemAvailable, MemTotal, and load averages, then
writes one JSON file per host under OUT_DIR (atomic replace).

Options:
  --out-dir DIR       Output directory (default: /home/kshukawa/.syndiff/host_stats)
  --interval SECONDS  Sleep between samples (default: 60)
  -h, --help          Show this help

Duplicate prevention is handled by launch_monitors.sh (pgrep before start).
EOF
}

log() {
  printf '%s host_sampler: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

condor_hostname() {
  local host
  host="$(hostname -f 2>/dev/null || hostname)"
  if [[ "${host}" =~ ^science([0-9]+)\. ]]; then
    printf 'plscience%s.stsci.edu' "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "${host}" =~ ^science([0-9]+)$ ]]; then
    printf 'plscience%s.stsci.edu' "${BASH_REMATCH[1]}"
    return
  fi
  printf '%s' "${host}"
}

read_mem_kb() {
  local key="$1"
  awk -v key="${key}" '$1 == key ":" { print $2; exit }' /proc/meminfo
}

ensure_out_dir() {
  local dir="$1"
  local fstype

  # /astro is autofs on the science cluster. mkdir before the mount is active
  # creates a local directory tree that is NOT visible on other hosts.
  if [[ "${dir}" == /astro/* ]]; then
    local attempt fstype
    for attempt in $(seq 1 15); do
      ls /astro/armin/koji >/dev/null 2>&1 || true
      if mountpoint -q /astro/armin 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if ! mountpoint -q /astro/armin 2>/dev/null; then
      log "ERROR: /astro/armin is not mounted after 15s (autofs). Refusing to mkdir ${dir} on local disk."
      log "DEBUG: df -T /astro 2>&1: $(df -T /astro 2>&1 | tr '\n' '; ')"
      return 1
    fi
  fi

  mkdir -p "${dir}"

  if [[ "${dir}" == /astro/* ]]; then
    fstype="$(findmnt -no FSTYPE -T "${dir}" 2>/dev/null || true)"
    if [[ -z "${fstype}" ]]; then
      fstype="$(df -T "${dir}" 2>/dev/null | awk 'NR==2 {print $2}')"
    fi
    case "${fstype}" in
      nfs|nfs4) ;;
      *)
        log "ERROR: ${dir} is not on NFS (fstype=${fstype:-unknown}). Refusing to write."
        log "DEBUG: df -T ${dir}: $(df -T "${dir}" 2>&1 | tr '\n' '; ')"
        return 1
        ;;
    esac
  fi
  return 0
}

sample_once() {
  local host ts mem_avail_kb mem_total_kb load1 load5 load15 out tmp
  host="$(condor_hostname)"
  ts="$(date +%s)"
  mem_avail_kb="$(read_mem_kb MemAvailable)"
  mem_total_kb="$(read_mem_kb MemTotal)"
  read -r load1 load5 load15 _ < /proc/loadavg

  if ! ensure_out_dir "${OUT_DIR}"; then
    return 1
  fi
  out="${OUT_DIR}/${host}.json"
  tmp="${out}.$$.tmp"

  # shellcheck disable=SC2016
  cat > "${tmp}" <<EOF
{
  "hostname": "${host}",
  "login_hostname": "$(hostname -f 2>/dev/null || hostname)",
  "timestamp": ${ts},
  "mem_available_mb": $((mem_avail_kb / 1024)),
  "mem_total_mb": $((mem_total_kb / 1024)),
  "load1": ${load1},
  "load5": ${load5},
  "load15": ${load15}
}
EOF
  mv -f "${tmp}" "${out}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --interval)
      INTERVAL_S="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${INTERVAL_S}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_S}" -lt 1 ]]; then
  echo "interval must be a positive integer (seconds)" >&2
  exit 2
fi

if ! ensure_out_dir "${OUT_DIR}"; then
  exit 1
fi

log "starting on $(hostname -f 2>/dev/null || hostname) -> ${OUT_DIR}/$(condor_hostname).json every ${INTERVAL_S}s"
if [[ "${OUT_DIR}" == /astro/* ]]; then
  log "DEBUG: mountpoint /astro/armin=$(mountpoint -q /astro/armin 2>/dev/null && echo yes || echo no) df=$(df -T "${OUT_DIR}" 2>/dev/null | awk 'NR==2{print $1,$2,$6}' || echo missing)"
fi

while true; do
  if sample_once; then
    :
  else
    log "sample failed; will retry"
  fi
  sleep "${INTERVAL_S}"
done
