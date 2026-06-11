"""Parse pipeline list entries, including ``external_workspaces`` preambles."""

from __future__ import annotations

from typing import Any

EXTERNAL_WORKSPACES_KEY = "external_workspaces"
_EXTERNAL_ENTRY_ALLOWED_KEYS = frozenset({EXTERNAL_WORKSPACES_KEY})


def is_external_workspaces_entry(entry: Any) -> bool:
    """True when *entry* is a preamble dict with ``external_workspaces`` and no ``kind``."""
    if not isinstance(entry, dict):
        return False
    if "kind" in entry:
        return False
    return EXTERNAL_WORKSPACES_KEY in entry


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


def split_pipeline(
    pipeline: list,
) -> tuple[list[str], list[tuple[int, dict]]]:
    """
    Split a config ``pipeline`` list into merged external labels and executable stages.

    Returns ``(external_labels, [(original_index, stage_dict), ...])``.

    ``external_workspaces`` entries must appear before the first ``kind:`` stage.
    """
    if not isinstance(pipeline, list):
        raise ValueError("pipeline must be a list")

    external_labels: list[str] = []
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

        if "kind" not in entry:
            raise ValueError(f"pipeline[{idx}]: missing required key 'kind'")

        if EXTERNAL_WORKSPACES_KEY in entry:
            raise ValueError(
                f"pipeline[{idx}]: entry cannot set both 'kind' and 'external_workspaces'"
            )

        seen_executable = True
        stages.append((idx, entry))

    return external_labels, stages
