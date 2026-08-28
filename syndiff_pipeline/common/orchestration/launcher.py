"""Stage job launchers: local subprocess or HTCondor."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List, Optional, Protocol

from syndiff_pipeline.common.orchestration import condor

if TYPE_CHECKING:
    from syndiff_pipeline.template_creation.orchestration.runner_config import RunnerConfig


class StageJobHandle(Protocol):
    """StageJobHandle (Protocol)."""

    def poll(self) -> int | None:
        """Return exit code if the job finished, else ``None``."""
        ...

    def terminate(self) -> None:
        """Signal the job to stop."""
        ...


@dataclass
class LocalJobHandle:
    """LocalJobHandle."""
    proc: subprocess.Popen

    def poll(self) -> int | None:
        """Poll.
        
        Returns
        -------
        int | None"""
        return self.proc.poll()

    def terminate(self) -> None:
        """Terminate."""
        if self.proc.poll() is None:
            self.proc.terminate()


@dataclass
class CondorJobHandle:
    """CondorJobHandle."""
    cluster_id: int
    submit_epoch: float

    def poll(self) -> int | None:
        """Poll.
        
        Returns
        -------
        int | None"""
        return condor.poll_cluster(self.cluster_id, submitted_at=self.submit_epoch)

    def terminate(self) -> None:
        """Terminate."""
        condor.remove_cluster(self.cluster_id)


@dataclass(frozen=True)
class LaunchDescriptor:
    """LaunchDescriptor."""
    executor: str
    native_id: int
    launch_token: str
    submit_epoch: float | None = None
    handle: StageJobHandle | None = None


def launch_stage(
    cmd: List[str],
    *,
    cfg: "RunnerConfig",
    stage: str,
    runs_root: str,
    run_id: str,
    target_label: str,
    launch_token: str,
    resources_override: Optional["condor.CondorResourceRequest"] = None,
    priority: int = 0,
) -> LaunchDescriptor:
    """Launch a stage locally or on Condor; return durable descriptor.

    ``resources_override``, when given, replaces the stage's usual static
    ``condor_resources(cfg)`` profile entirely (e.g. a per-target
    preflight-sized small job for ``ps1_process``). ``priority`` (CSV-row-order
    derived; see ``PipelineState.create_run``) is applied as a final override
    on whichever resource profile was selected, so it covers both the static
    per-stage profile and any per-target override uniformly.
    """
    if cfg.stage_executor(stage) == "condor":
        if resources_override is not None:
            resources = resources_override
        else:
            from syndiff_pipeline.pipeline_spec import get_syndiff_pipeline

            stage_spec = get_syndiff_pipeline().require(stage)
            resources = stage_spec.condor_resources(cfg)
        if resources is None:
            raise ValueError(f"No Condor resource profile for stage {stage!r}")
        resources = replace(resources, priority=priority)
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
