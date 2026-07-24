#!/usr/bin/env bash
# Start/stop/status host_sampler.sh across the STScI science cluster.
#
# Assumes host_sampler.sh is already installed on shared home NFS, e.g.:
#   /home/kshukawa/.syndiff/bin/host_sampler.sh
set -euo pipefail

REMOTE_SAMPLER="${REMOTE_SAMPLER:-/home/kshukawa/.syndiff/bin/host_sampler.sh}"
STATS_DIR="${HOST_STATS_DIR:-/home/kshukawa/.syndiff/host_stats}"
INTERVAL_S="${HOST_SAMPLER_INTERVAL:-60}"
MAX_HEARTBEAT_AGE_S="${HOST_SAMPLER_MAX_HEARTBEAT_AGE_S:-120}"
FORCE_START=0
VERBOSE="${HOST_MONITOR_VERBOSE:-1}"
INSTALL_SAMPLERS=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SAMPLER="${LOCAL_SAMPLER:-${SCRIPT_DIR}/host_sampler.sh}"
HOSTS=(
  science{1..15}.stsci.edu
)

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|ls|debug> [--force] [--install] [host]

Start host_sampler.sh on all science hosts via SSH. The sampler script and
heartbeat JSON files live on shared home NFS (/home/kshukawa/...).

Defaults:
  REMOTE_SAMPLER=${REMOTE_SAMPLER}
  HOST_STATS_DIR=${STATS_DIR}

Options:
  --install   scp host_sampler.sh from this repo to REMOTE_SAMPLER first
  --force     kill and restart even when a fresh heartbeat exists
  --quiet     less output (HOST_MONITOR_VERBOSE=0)

Examples:
  $(basename "$0") start
  $(basename "$0") debug science1.stsci.edu
  $(basename "$0") status
EOF
}

dbg() { [[ "${VERBOSE}" == "1" ]] && printf '  | %s\n' "$*" || true; }

remote_run() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}" bash -l -s -- "$@"
}

remote_install_sampler() {
  local host="$1"
  if [[ ! -f "${LOCAL_SAMPLER}" ]]; then
    echo "LOCAL_SAMPLER not found: ${LOCAL_SAMPLER}" >&2
    return 1
  fi
  dbg "install: scp -> ${host}:${REMOTE_SAMPLER}"
  remote_run "${host}" "mkdir -p $(dirname "${REMOTE_SAMPLER}")"
  scp -o BatchMode=yes -o ConnectTimeout=15 "${LOCAL_SAMPLER}" "${host}:${REMOTE_SAMPLER}"
  remote_run "${host}" "chmod +x '${REMOTE_SAMPLER}'"
}

remote_start_one() {
  local host="$1"
  echo "${host}"
  if [[ "${INSTALL_SAMPLERS}" -eq 1 ]]; then
    remote_install_sampler "${host}" || return 1
  fi
  remote_run "${host}" \
    "${REMOTE_SAMPLER}" "${STATS_DIR}" "${INTERVAL_S}" "${FORCE_START}" \
    "${MAX_HEARTBEAT_AGE_S}" "${VERBOSE}" <<'REMOTE' || return 1
set -uo pipefail
sampler="$1"
stats_dir="$2"
interval="$3"
force="$4"
max_age="$5"
verbose="$6"

say() { [[ "$verbose" == "1" ]] && printf '  | %s\n' "$*" || true; }

condor_host() {
  local h
  h="$(hostname -f 2>/dev/null || hostname)"
  if [[ "$h" =~ ^science([0-9]+)\. ]]; then printf 'plscience%s.stsci.edu' "${BASH_REMATCH[1]}"; return; fi
  if [[ "$h" =~ ^science([0-9]+)$ ]]; then printf 'plscience%s.stsci.edu' "${BASH_REMATCH[1]}"; return; fi
  printf '%s' "$h"
}

json="${stats_dir}/$(condor_host).json"
sampler_pgrep='[h]ost_sampler.sh --out-dir '"${stats_dir}"

say "hostname=$(hostname -f) condor=$(condor_host) user=$(whoami)"
say "sampler=${sampler} stats_dir=${stats_dir}"

if [[ ! -x "$sampler" ]]; then
  say "RESULT: FAILED — sampler not executable: ${sampler}"
  exit 1
fi

mkdir -p "${stats_dir}"
say "stats_dir listing: $(ls -la "${stats_dir}" 2>&1 | tr '\n' '; ')"

heartbeat_ok() {
  [[ -f "$json" ]] || return 1
  python3 - "$json" "$max_age" <<'PY'
import json, sys, time
path, max_age = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as fh:
    ts = int(json.load(fh)["timestamp"])
sys.exit(0 if time.time() - ts <= max_age else 1)
PY
}

if pgrep -f "${sampler_pgrep}" >/dev/null 2>&1; then
  pid="$(pgrep -f "${sampler_pgrep}" | head -n1)"
  if [[ "$force" == "1" ]]; then
    say "killing pid ${pid} (--force)"
    pkill -f "${sampler_pgrep}" 2>/dev/null || true
    sleep 1
  elif heartbeat_ok; then
    mem="$(python3 -c "import json; print(json.load(open('${json}'))['mem_available_mb'])" 2>/dev/null || echo '?')"
    say "RESULT: already running pid ${pid} (heartbeat ok, mem_avail=${mem}MB)"
    exit 0
  else
    say "killing stale pid ${pid} (no fresh heartbeat at ${json})"
    pkill -f "${sampler_pgrep}" 2>/dev/null || true
    sleep 1
  fi
elif pgrep -f '[h]ost_sampler.sh --out-dir' >/dev/null 2>&1; then
  say "killing sampler on different stats path"
  pkill -f '[h]ost_sampler.sh --out-dir' 2>/dev/null || true
  sleep 1
fi

say "nohup ${sampler} --out-dir ${stats_dir} --interval ${interval}"
# setsid + nohup: survive SSH disconnect (new session, no controlling tty)
setsid nohup "${sampler}" --out-dir "${stats_dir}" --interval "${interval}" \
  >> "${stats_dir}/sampler.log" 2>&1 < /dev/null &
disown -h "$!" 2>/dev/null || true
sleep 2

if ! pgrep -f "${sampler_pgrep}" >/dev/null 2>&1; then
  say "RESULT: FAILED — process died"
  tail -n 15 "${stats_dir}/sampler.log" 2>/dev/null | sed 's/^/  |   /' || say "(no sampler.log)"
  exit 1
fi

pid="$(pgrep -f "${sampler_pgrep}" | head -n1)"
if [[ -f "$json" ]]; then
  mem="$(python3 -c "import json; print(json.load(open('${json}'))['mem_available_mb'])" 2>/dev/null || echo '?')"
  say "RESULT: started pid ${pid} json=${json} mem_avail=${mem}MB"
else
  say "RESULT: started pid ${pid} (no json yet; check sampler.log)"
  tail -n 5 "${stats_dir}/sampler.log" 2>/dev/null | sed 's/^/  |   /' || true
fi
REMOTE
}

remote_stop_one() {
  local host="$1"
  echo "${host}"
  remote_run "${host}" <<'REMOTE' || return 1
set -euo pipefail
if pgrep -af '[h]ost_sampler.sh --out-dir' >/dev/null 2>&1; then
  pgrep -af '[h]ost_sampler.sh --out-dir' | sed 's/^/  | before: /'
  pkill -f '[h]ost_sampler.sh --out-dir' 2>/dev/null || true
  sleep 1
  echo "  | RESULT: stopped"
else
  echo "  | RESULT: not running"
fi
REMOTE
}

remote_debug_one() {
  local host="$1"
  echo "${host}"
  remote_run "${host}" "${REMOTE_SAMPLER}" "${STATS_DIR}" <<'REMOTE'
set -uo pipefail
sampler="$1"
stats_dir="$2"
condor_host() {
  local h; h="$(hostname -f 2>/dev/null || hostname)"
  if [[ "$h" =~ ^science([0-9]+)\. ]]; then printf 'plscience%s.stsci.edu' "${BASH_REMATCH[1]}"; return; fi
  printf '%s' "$h"
}
say() { printf '  | %s\n' "$*"; }
json="${stats_dir}/$(condor_host).json"
say "hostname=$(hostname -f) home=${HOME}"
say "sampler=${sampler} executable=$([[ -x "$sampler" ]] && echo yes || echo NO)"
say "stats_dir=${stats_dir}"
say "pgrep: $(pgrep -af host_sampler || echo none)"
say "ls stats_dir: $(ls -la "$stats_dir" 2>&1 | tr '\n' '; ')"
say "json=${json} exists=$([[ -f "$json" ]] && echo yes || echo no)"
[[ -f "${stats_dir}/sampler.log" ]] && tail -n 10 "${stats_dir}/sampler.log" | sed 's/^/  | log: /'
REMOTE
}

remote_status_one() {
  local host="$1"
  remote_run "${host}" "${STATS_DIR}" "${MAX_HEARTBEAT_AGE_S}" <<'REMOTE'
set -euo pipefail
stats_dir="$1"; max_age="$2"
condor_host() {
  local h; h="$(hostname -f 2>/dev/null || hostname)"
  if [[ "$h" =~ ^science([0-9]+)\. ]]; then printf 'plscience%s.stsci.edu' "${BASH_REMATCH[1]}"; return; fi
  printf '%s' "$h"
}
json="${stats_dir}/$(condor_host).json"
if pgrep -f "[h]ost_sampler.sh --out-dir ${stats_dir}" >/dev/null 2>&1; then
  proc="RUNNING pid=$(pgrep -f "[h]ost_sampler.sh --out-dir ${stats_dir}" | head -n1)"
else proc="DEAD"; fi
if [[ -f "$json" ]]; then
  hb="$(python3 - "$json" "$max_age" <<'PY'
import json,sys,time
p,m=sys.argv[1],int(sys.argv[2])
d=json.load(open(p))
age=int(time.time())-int(d["timestamp"])
print(f"{age}s mem={d['mem_available_mb']}MB load1={d['load1']} {'STALE' if age>m else 'ok'}")
PY
)"
else hb="no file ${json}"; fi
printf '%s\t%s\t%s\n' "$(condor_host)" "$proc" "$hb"
REMOTE
}

run_all() {
  local fn="$1" failed=0 host
  for host in "${HOSTS[@]}"; do
    if ! "${fn}" "${host}"; then failed=$((failed + 1)); fi
    echo ""
  done
  echo "=== ${failed} failed / ${#HOSTS[@]} hosts ==="
  [[ "${failed}" -eq 0 ]]
}

cmd="${1:-}"; shift || true
DEBUG_HOST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE_START=1; shift ;;
    --install) INSTALL_SAMPLERS=1; shift ;;
    --quiet) VERBOSE=0; shift ;;
    science*.stsci.edu) DEBUG_HOST="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${cmd}" in
  start)
    echo "sampler: ${REMOTE_SAMPLER}"
    echo "stats:   ${STATS_DIR}"
    run_all remote_start_one
    ;;
  stop)
    run_all remote_stop_one
    ;;
  restart)
    FORCE_START=1
    echo "=== stop ==="
    run_all remote_stop_one || true
    echo "=== start ==="
    echo "sampler: ${REMOTE_SAMPLER}"
    echo "stats:   ${STATS_DIR}"
    run_all remote_start_one
    ;;
  status)
    printf '%-28s %-24s %s\n' "HOST" "PROCESS" "HEARTBEAT"
    for host in "${HOSTS[@]}"; do
      remote_status_one "${host}" 2>/dev/null || printf '%-28s %-24s %s\n' "${host}" "SSH_FAIL" "-"
    done
    ;;
  debug)
    remote_debug_one "${DEBUG_HOST:-science1.stsci.edu}"
    ;;
  ls)
    echo "stats: ${STATS_DIR}"
    ls -la "${STATS_DIR}" 2>/dev/null || echo "(not visible on this machine — run on plscience5)"
    ;;
  -h|--help|"")
    usage; exit 0 ;;
  *)
    echo "Unknown command: ${cmd}" >&2; usage >&2; exit 2 ;;
esac
