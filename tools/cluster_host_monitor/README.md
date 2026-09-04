# Cluster Host Monitor

Reads live host memory and load on the STScI science cluster (`plscience1`–`plscience15`)
straight from HTCondor's own ClassAds. Syndiff queries this at every `condor_submit` to
exclude execute hosts with no live data and rank survivors by lowest load (`LOAD15`), and
to filter live on real available memory / load via Requirements clauses.

**Primary CLI:** `syndiff cluster` (see [Cluster host snapshot](../../docs/markdown/syndiff_cli.md#cluster-host-snapshot)).

## How it fits together

```text
plscienceN execute host  --STARTD_CRON (every 30s)-->  MemAvailableMB ClassAd attribute
                                                                  |
                                                          condor_status collector
                                                                  |
              syndiff cluster (query)  <───────────────  condor_submit (filter + rank)
                    |
              Discord bot (message contains "cluster")
```

`MemAvailableMB` is published by an HTCondor `STARTD_CRON` job installed on every plscience
execute host (STARS ticket RITM0202207, baked into Ansible so it survives host rebuilds):
a small script reads `/proc/meminfo MemAvailable` every 30s and merges it into the machine's
ClassAd. `DetectedMemory` (stable total physical RAM) and `LoadAvg` (native Condor load) round
out the three live attributes syndiff reads. There is no separate sampler daemon, heartbeat
file, or NFS directory to maintain — a host's data is exactly as fresh as the last
`condor_status` query.

| Component | Path / command |
|-----------|----------------|
| Live query + policy | `syndiff_pipeline/common/orchestration/host_stats.py` (`query_condor_host_samples`, `apply_host_stats_policy`) |
| Human-readable table | `syndiff cluster` → `host_stats_cli.py` |
| Legacy placement check | `read_host_stats.py` (= `syndiff cluster --check`) |

## Verify the Condor-side attribute directly

```bash
condor_config_val STARTD_CRON_JOBLIST                              # expect: MEM_AVAILABLE
condor_status -af Machine DetectedMemory MemAvailableMB LoadAvg
grep MemAvailable /proc/meminfo                                    # compare on a given host
```

## Reading cluster status

### `syndiff cluster` (preferred)

**Status mode** (default) — live snapshot, no pass/fail:

```bash
syndiff cluster
```

**Placement check** — preview Condor exclusions/live clauses for a stage class:

```bash
syndiff cluster --check --preset 500gb       # ps1_process (300 GB min available)
syndiff cluster --check --preset 128gb       # mapping / remap (128 GB min available)
syndiff cluster --check --site config/ --stage ps1_process
syndiff cluster --check --site config/ --stage diff
```

Example status output (fixed-width columns; widths grow to fit values like `361.7GB`):

```text
HOST                   SLOT   AVAIL LOAD15
--------------------- ----- ------- ------
plscience4.stsci.edu  515GB 361.7GB  37.90
plscience5.stsci.edu  515GB 423.7GB   4.75
plscience7.stsci.edu      ?       ?      ?
```

#### Column reference

| Column | Meaning |
|--------|---------|
| `HOST` | Execute hostname (`plscienceN.stsci.edu`) |
| `SLOT` | Total RAM from `DetectedMemory` (stable physical total, not the fluctuating claimable `Memory`) |
| `AVAIL` | Available RAM from `MemAvailableMB` |
| `LOAD15` | `LoadAvg` — ranking key among hosts with live data at submit |
| `VERDICT` | Only with `--check`: `OK` or `EXCLUDE (reason, ...)` — shown for any host that currently fails a threshold, even though only a `?` (no data at all) host is hard-excluded by name in the generated Requirements |

With `--check`, a footer prints thresholds, excluded/OK counts, a Condor `requirements`
snippet for genuinely-missing hosts, and the live `MemAvailableMB`/`LoadAvg` clauses that
filter everything else (re-evaluated by the negotiator every cycle, not frozen at submit time).

Machine-readable output (for scripting) — note these only ever list hosts with **no live
data at all**, since low-mem/high-load hosts are filtered live, not by name:

```bash
syndiff cluster --format requirements --check --preset 500gb
syndiff cluster --format bad-machines --check --site config/ --stage mapping
syndiff cluster --format hosts --check --preset 128gb
```

### `read_host_stats.py` (legacy)

```bash
python3 tools/cluster_host_monitor/read_host_stats.py --preset 500gb
```

Thin wrapper: calls `syndiff cluster --check` with VERDICT on by default. Prefer
`syndiff cluster` for day-to-day use.

### Discord bot

When the in-process Discord status bot is enabled (`notifications.bot.enabled`), post any
message whose text **contains the word `cluster`** (case-insensitive) in the configured
channel. The bot replies with the **status-mode** table (no `VERDICT`), in a fenced code
block with header `**syndiff cluster**` — same format as `condor_q` replies.

Examples: `cluster`, `how is the cluster?`, `syndiff cluster`.

Exact-match Condor commands (`condor_q`, `condor_qn`, `condor_status`, `condor_status -tla`)
take precedence when the message is only that command.

## Config knobs (what `--check` evaluates)

Template stages (`pipeline.yaml` → `stages.*`):

| Key | Meaning | Example |
|-----|---------|---------|
| `host_stats_min_mem_mb` | Live `MemAvailableMB >= ...` Requirements floor | `300000` for `ps1_process` |
| `host_stats_max_load15` | Live `LoadAvg <= ...` Requirements ceiling | `10.0` |

Diff / star / photometry: same keys under `condor:` in `diff_config.yaml`, `star_config.yaml`,
`photometry_config.yaml`.

| Concept | Role |
|---------|------|
| `condor_request_memory` | HTCondor cgroup **claim** (`Memory >= …` in requirements) |
| `host_stats_min_mem_mb` | Live `MemAvailableMB` **filter** (Requirements clause, re-evaluated every cycle) |
| `host_stats_max_load15` | Live `LoadAvg` **filter** (Requirements clause) and **rank** (lowest wins among hosts with live data) |

If `condor_status` returns no usable data at submit time (collector unreachable, or
`MemAvailableMB` not yet published anywhere), syndiff falls back to `Memory >= request_memory`
and `rank = -LoadAvg`.

Reactive per-run exclusions from `{stage}.condor.bad_machines` (eviction/memory holds)
still merge on top of host-stats exclusions at submit time.

## Manual inspection before a big run

```bash
condor_status -af Name State Activity LoadAv Mem
syndiff cluster
syndiff cluster --check --preset 500gb
syndiff cluster --check --min-mem-mb 300000 --max-load15 10
```

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `?` for a host in `syndiff cluster` | Host down, decommissioned, or STARTD_CRON not yet rolled out on it | `condor_status -af Machine MemAvailableMB` for that host directly; check with IT if it's a live host that should be reporting |
| Submit warns, uses `-LoadAvg` rank | `condor_status` unreachable or returned nothing usable | Check `condor_status` works from the submit host; check collector connectivity |
| Discord shows table but CLI empty | Bot runs on submit host with Condor CLI access | Run `syndiff cluster` on the same host |
| Discord replies **N identical** cluster/status tables | N Discord bot processes with the same token (orphans after daemon restarts) | `pgrep -af orchestration.discord_bot` on every science host that has run the supervisor; `pkill -f 'template_creation.orchestration.discord_bot'`; single `syndiff daemon start`. After the lease fix, bots hold `control/discord_bot.lease.json` so a second instance exits before connecting. |

## History

Before the `MemAvailableMB` STARTD_CRON attribute existed (STARS ticket RITM0202207,
completed 2026-09-02), this directory ran a home-grown sampler: an SSH-launched bash loop
(`host_sampler.sh`) on each host writing JSON heartbeats to a shared NFS directory
(`launch_monitors.sh` to deploy/manage it). That had real operational costs — autofs/NFS
mount races, no survival across host reboots/patches, manual babysitting, orphaned-process
bugs — all now moot since Condor publishes the same data live. Those scripts were removed
once this migration landed; see git history if you need to reference them.
