# Cluster Host Monitor

Samples real host memory and load on the STScI science cluster (`plscience1`–`plscience15`).
Syndiff reads these heartbeats at every `condor_submit` to exclude bad execute hosts and
rank survivors by lowest 15-minute load (`load15`).

**Primary CLI:** `syndiff cluster` (see [Cluster host snapshot](../../docs/markdown/syndiff_cli.md#cluster-host-snapshot)).

## How it fits together

```text
plscienceN  →  host_sampler.sh (every 60s)  →  ~/.syndiff/host_stats/plscienceN.stsci.edu.json
                                                          ↓
              syndiff cluster (read)  ←────────  condor_submit (filter + rank)
                    ↓
              Discord bot (message contains "cluster")
```

| Component | Path / command |
|-----------|----------------|
| Sampler script (on each host) | `~/.syndiff/bin/host_sampler.sh` |
| Heartbeat JSON | `HOST_STATS_DIR` / `plscienceN.stsci.edu.json` (default `~/.syndiff/host_stats`) |
| Submit-time selection | `syndiff_pipeline/common/orchestration/host_stats.py` |
| Human-readable table | `syndiff cluster` → `host_stats_cli.py` |
| Legacy placement check | `read_host_stats.py` (= `syndiff cluster --check`) |

## Layout (shared home NFS)

```text
/home/kshukawa/.syndiff/
  bin/host_sampler.sh          # shared across all hosts
  host_stats/
    plscience1.stsci.edu.json  # one heartbeat per host, updated every 60s
    ...
    sampler.log
```

Each JSON file contains (among other fields): `hostname`, `timestamp`, `mem_total_mb`,
`mem_available_mb`, `load1`, `load5`, `load15`.

## Deploy samplers (from your Mac)

Copy `launch_monitors.sh` to your Mac (or run from a machine with SSH to science hosts).

```bash
./launch_monitors.sh start      # run ~/.syndiff/bin/host_sampler.sh on all hosts
./launch_monitors.sh start science5            # one host only (also: plscience5, science5.stsci.edu)
./launch_monitors.sh start --install --force plscience12
./launch_monitors.sh status
./launch_monitors.sh status science5
./launch_monitors.sh debug science1.stsci.edu   # troubleshoot one host
./launch_monitors.sh stop
./launch_monitors.sh stop science12
```

No `scp` / install step unless you pass `--install` to push an updated sampler.

Each host is started with `setsid nohup ... &` so the sampler **keeps running after SSH
disconnects**. It does **not** auto-restart after a host reboot or monthly patch — re-run
`launch_monitors.sh start` after maintenance.

## Reading sampler JSON

### `syndiff cluster` (preferred)

**Status mode** (default) — live snapshot, no pass/fail:

```bash
syndiff cluster
```

**Placement check** — preview Condor exclusions for a stage class:

```bash
syndiff cluster --check --preset 500gb       # ps1_process (300 GB min available)
syndiff cluster --check --preset 128gb       # mapping / remap (128 GB min available)
syndiff cluster --check --site config/ --stage ps1_process
syndiff cluster --check --site config/ --stage diff
```

Example status output (fixed-width columns; widths grow to fit values like `361.7GB`):

```text
HOST                   SLOT   AVAIL LOAD15 AGE
--------------------- ----- ------- ------ ---
plscience4.stsci.edu  515GB 361.7GB  37.90  8s
plscience5.stsci.edu  515GB 423.7GB   4.75  0s
plscience7.stsci.edu      ?       ?      ?   ?
```

#### Column reference

| Column | Meaning |
|--------|---------|
| `HOST` | Execute hostname (`plscienceN.stsci.edu`) |
| `SLOT` | Total RAM rounded to Condor `Memory` buckets (`128GB`, `515GB`, …) from `mem_total_mb` |
| `AVAIL` | Available RAM from `mem_available_mb` (decimal GB) |
| `LOAD15` | 15-minute load average — **sole ranking key** among eligible hosts at submit |
| `AGE` | Seconds since last heartbeat; stale if >300 s (excluded at submit) |
| `VERDICT` | Only with `--check`: `OK` or `EXCLUDE (reason, …)` |

With `--check`, a footer prints thresholds, excluded/OK counts, and a Condor `requirements`
snippet (`Machine != "plscience4.stsci.edu" || …`).

Machine-readable output (for scripting):

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
| `host_stats_min_mem_mb` | Exclude if `mem_available_mb` below this | `300000` for `ps1_process` |
| `host_stats_max_load15` | Exclude if 15-min load at or above this | `10.0` |

Diff / star / photometry: same keys under `condor:` in `diff_config.yaml`, `star_config.yaml`,
`photometry_config.yaml`.

| Concept | Role |
|---------|------|
| `condor_request_memory` | HTCondor cgroup **claim** (`Memory >= …` in requirements) |
| `host_stats_min_mem_mb` | Real `MemAvailable` **filter** from sampler JSON (not ranking) |
| `load15` | **Filter** (< `host_stats_max_load15`) and **rank** (lowest wins) |

If no usable sampler JSON exists at submit time, syndiff falls back to
`Memory >= request_memory` and `rank = -LoadAvg`.

Reactive per-run exclusions from `{stage}.condor.bad_machines` (eviction/memory holds)
still merge on top of host-stats exclusions at submit time.

## Environment overrides

```bash
export HOST_STATS_DIR=/home/kshukawa/.syndiff/host_stats   # default
export REMOTE_SAMPLER=/home/kshukawa/.syndiff/bin/host_sampler.sh
```

## Manual inspection before a big run

```bash
condor_status -af Name State Activity LoadAv Mem
syndiff cluster
syndiff cluster --check --preset 500gb
syndiff cluster --check --min-mem-mb 300000 --max-load15 10
ls -la /home/kshukawa/.syndiff/host_stats/
```

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `?` for a host in `syndiff cluster` | Sampler not running or stale JSON | `./launch_monitors.sh debug plscienceN.stsci.edu` |
| All hosts `EXCLUDE (missing)` | Wrong `HOST_STATS_DIR` or NFS mount | Check `stats_dir:` line from `syndiff cluster` |
| Submit warns, uses `-LoadAvg` rank | No fresh heartbeats | Restart samplers; verify JSON age <300 s |
| Discord shows table but CLI empty | Bot runs on submit host with NFS access | Run `syndiff cluster` on same host |
| Discord replies **N identical** cluster/status tables | N Discord bot processes with the same token (orphans after daemon restarts) | `pgrep -af orchestration.discord_bot` on every science host that has run the supervisor; `pkill -f 'template_creation.orchestration.discord_bot'`; single `syndiff daemon start`. After the lease fix, bots hold `control/discord_bot.lease.json` so a second instance exits before connecting. |

