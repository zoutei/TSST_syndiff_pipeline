"""Parse pipeline list entries, including preamble dicts without ``kind``."""

from __future__ import annotations

from typing import Any

from syndiff_pipeline.difference_imaging.support.workspace_inherit import (
    WorkspaceInheritSpec,
)

EXTERNAL_WORKSPACES_KEY = "external_workspaces"
WORKSPACE_INHERIT_KEY = "workspace_inherit"
_EXTERNAL_ENTRY_ALLOWED_KEYS = frozenset({EXTERNAL_WORKSPACES_KEY})
_WORKSPACE_INHERIT_ALLOWED_KEYS = frozenset({"from", "labels", "root_artifacts"})


def is_external_workspaces_entry(entry: Any) -> bool:
    """True when *entry* is a preamble dict with ``external_workspaces`` and no ``kind``."""
    if not isinstance(entry, dict):
        return False
    if "kind" in entry:
        return False
    return EXTERNAL_WORKSPACES_KEY in entry


def is_workspace_inherit_entry(entry: Any) -> bool:
    """True when *entry* is a preamble dict with ``workspace_inherit`` and no ``kind``."""
    if not isinstance(entry, dict):
        return False
    if "kind" in entry:
        return False
    return WORKSPACE_INHERIT_KEY in entry


def is_pipeline_preamble_entry(entry: Any) -> bool:
    return is_external_workspaces_entry(entry) or is_workspace_inherit_entry(entry)


def validate_external_entry_keys(entry: dict, idx: int) -> None:
    extra = set(entry.keys()) - _EXTERNAL_ENTRY_ALLOWED_KEYS
    if extra:
        raise ValueError(
            f"pipeline[{idx}] external_workspaces: unexpected key(s) {sorted(extra)!r}"
        )


def parse_external_workspace_labels(entry: dict, idx: int) -> list[str]:
    validate_external_entry_keys(entry, idx)
    raw = entry.get(EXTERNAL_WORKSPACES_KEY)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"pipeline[{idx}] external_workspaces: must be a list, got {type(raw).__name__}"
        )
    labels: list[str] = []
    for j, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"pipeline[{idx}] external_workspaces[{j}]: must be a non-empty string, "
                f"got {item!r}"
            )
        labels.append(item.strip())
    return labels


def _parse_string_list(raw: Any, *, pipeline_idx: int, field: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"pipeline[{pipeline_idx}] workspace_inherit.{field}: must be a list, "
            f"got {type(raw).__name__}"
        )
    out: list[str] = []
    for j, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"pipeline[{pipeline_idx}] workspace_inherit.{field}[{j}]: "
                f"must be a non-empty string, got {item!r}"
            )
        out.append(item.strip())
    return out


def parse_workspace_inherit_spec(entry: dict, idx: int) -> WorkspaceInheritSpec:
    extra = set(entry.keys()) - {WORKSPACE_INHERIT_KEY}
    if extra:
        raise ValueError(
            f"pipeline[{idx}] workspace_inherit: unexpected key(s) {sorted(extra)!r}"
        )
    raw = entry.get(WORKSPACE_INHERIT_KEY)
    if not isinstance(raw, dict):
        raise ValueError(
            f"pipeline[{idx}] workspace_inherit: must be a mapping, "
            f"got {type(raw).__name__}"
        )
    unknown = set(raw.keys()) - _WORKSPACE_INHERIT_ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"pipeline[{idx}] workspace_inherit: unknown key(s) {sorted(unknown)!r}; "
            f"allowed: {sorted(_WORKSPACE_INHERIT_ALLOWED_KEYS)!r}"
        )
    from_run = raw.get("from")
    if not isinstance(from_run, str) or not from_run.strip():
        raise ValueError(
            f"pipeline[{idx}] workspace_inherit.from: non-empty string required"
        )
    labels = _parse_string_list(
        raw.get("labels"), pipeline_idx=idx, field="labels"
    )
    root_artifacts = _parse_string_list(
        raw.get("root_artifacts"), pipeline_idx=idx, field="root_artifacts"
    )
    if not labels and not root_artifacts:
        raise ValueError(
            f"pipeline[{idx}] workspace_inherit: at least one of "
            f"'labels' or 'root_artifacts' is required"
        )
    return WorkspaceInheritSpec(
        from_run_id=from_run.strip(),
        labels=tuple(labels),
        root_artifacts=tuple(root_artifacts),
    )


def split_pipeline(
    pipeline: list,
) -> tuple[list[str], list[WorkspaceInheritSpec], list[tuple[int, dict]]]:
    """
    Split a config ``pipeline`` list into preambles and executable stages.

    Returns ``(external_labels, inherit_specs, [(original_index, stage_dict), ...])``.

    Preamble entries must appear before the first ``kind:`` stage.
    """
    if not isinstance(pipeline, list):
        raise ValueError("pipeline must be a list")

    external_labels: list[str] = []
    inherit_specs: list[WorkspaceInheritSpec] = []
    stages: list[tuple[int, dict]] = []
    seen_executable = False

    for idx, entry in enumerate(pipeline):
        if not isinstance(entry, dict):
            raise ValueError(
                f"pipeline[{idx}] must be a mapping, got {type(entry).__name__}"
            )

        if is_external_workspaces_entry(entry):
            if seen_executable:
                raise ValueError(
                    f"pipeline[{idx}] external_workspaces: must appear before the first "
                    f"'kind:' stage"
                )
            external_labels.extend(parse_external_workspace_labels(entry, idx))
            continue

        if is_workspace_inherit_entry(entry):
            if seen_executable:
                raise ValueError(
                    f"pipeline[{idx}] workspace_inherit: must appear before the first "
                    f"'kind:' stage"
                )
            inherit_specs.append(parse_workspace_inherit_spec(entry, idx))
            continue

        if "kind" not in entry:
            raise ValueError(f"pipeline[{idx}]: missing required key 'kind'")

        if EXTERNAL_WORKSPACES_KEY in entry:
            raise ValueError(
                f"pipeline[{idx}]: entry cannot set both 'kind' and 'external_workspaces'"
            )
        if WORKSPACE_INHERIT_KEY in entry:
            raise ValueError(
                f"pipeline[{idx}]: entry cannot set both 'kind' and 'workspace_inherit'"
            )

        seen_executable = True
        stages.append((idx, entry))

    return external_labels, inherit_specs, stages


def inherited_workspace_labels(inherit_specs: list[WorkspaceInheritSpec]) -> list[str]:
    """Workspace label names made available by inherit preambles."""
    return [label for spec in inherit_specs for label in spec.labels]
