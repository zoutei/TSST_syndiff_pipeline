# Cluster Host Monitor

Samples real host memory/load on the science cluster and recommends Condor
exclusions before submit.

## Layout (shared home NFS)

```text
/home/kshukawa/.syndiff/
  bin/host_sampler.sh          # already installed; shared across all hosts
  host_stats/
    plscience1.stsci.edu.json  # one heartbeat per host, updated every 60s
    ...
    sampler.log
```

## From your Mac (SSH to science1–15)

Copy only `launch_monitors.sh` to your Mac (or run from repo on a machine with SSH).

```bash
./launch_monitors.sh start      # run existing ~/.syndiff/bin/host_sampler.sh on all hosts
./launch_monitors.sh status
./launch_monitors.sh debug science1.stsci.edu   # troubleshoot one host
./launch_monitors.sh stop
```

No `scp` / install step unless you pass `--install` to push an updated sampler.

Each host is started with `setsid nohup ... &` so the sampler **keeps running after SSH
disconnects**. It does **not** auto-restart after a host reboot or monthly patch — re-run
`launch_monitors.sh start` after maintenance.

## On plscience5 (read results)

```bash
ls -la /home/kshukawa/.syndiff/host_stats/
python3 tools/cluster_host_monitor/read_host_stats.py --preset 500gb
```

## Overrides

```bash
export HOST_STATS_DIR=/home/kshukawa/.syndiff/host_stats   # default
export REMOTE_SAMPLER=/home/kshukawa/.syndiff/bin/host_sampler.sh
```

## Pre-submit

```bash
condor_status -af Name State Activity LoadAv Mem
python3 tools/cluster_host_monitor/read_host_stats.py --preset 500gb
```

Paste excluded hosts into `{stage}.condor.bad_machines` or Condor requirements.
