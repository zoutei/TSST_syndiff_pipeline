"""Stage job launchers: local subprocess or HTCondor."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import List, Protocol

from syndiff_pipeline.template_creation.orchestration import condor
from syndiff_pipeline.template_creation.orchestration.runner_config import RunnerConfig


class StageJobHandle(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


@dataclass
class LocalJobHandle:
    proc: subprocess.Popen

    def poll(self) -> int | None:
        return self.proc.poll()

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()


@dataclass
class CondorJobHandle:
    cluster_id: int
    submit_epoch: float

    def poll(self) -> int | None:
        return condor.poll_cluster(self.cluster_id, submitted_at=self.submit_epoch)

    def terminate(self) -> None:
        condor.remove_cluster(self.cluster_id)


@dataclass(frozen=True)
class LaunchDescriptor:
    executor: str
    native_id: int
    launch_token: str
    submit_epoch: float | None = None
    handle: StageJobHandle | None = None


def _condor_resources(cfg: RunnerConfig, stage: str) -> condor.CondorResourceRequest:
    if stage == "mapping":
        params = cfg.stages.mapping
    elif stage == "ps1_process":
        params = cfg.stages.ps1_process
    else:
        raise ValueError(f"No Condor resource profile for stage {stage!r}")
    return condor.CondorResourceRequest(
        request_cpus=params.condor_request_cpus,
        request_memory_mb=params.condor_request_memory,
        requirements=params.condor_requirements,
        rank=params.condor_rank,
    )


def launch_stage(
    cmd: List[str],
    *,
    cfg: RunnerConfig,
    stage: str,
    runs_root: str,
    run_id: str,
    target_label: str,
    launch_token: str,
) -> LaunchDescriptor:
    """Launch a stage locally or on Condor; return durable descriptor."""
    if cfg.stage_executor(stage) == "condor":
        resources = _condor_resources(cfg, stage)
        cluster_id, submit_epoch = condor.submit_job(
            cmd,
            runs_root,
            run_id,
            target_label,
            stage,
            resources=resources,
        )
        handle: StageJobHandle = CondorJobHandle(cluster_id, submit_epoch)
        return LaunchDescriptor(
            executor="condor",
            native_id=cluster_id,
            launch_token=launch_token,
            submit_epoch=submit_epoch,
            handle=handle,
        )

    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return LaunchDescriptor(
        executor="local",
        native_id=proc.pid,
        launch_token=launch_token,
        submit_epoch=time.time(),
        handle=LocalJobHandle(proc),
    )
