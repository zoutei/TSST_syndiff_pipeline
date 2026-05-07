"""
Validate config ``pipeline`` lists before execution.
"""

from __future__ import annotations

from typing import Any

from .config import SynDiffConfig
from .subtract_expr import parse_subtract_expression

STAGE_KINDS = frozenset(
    {
        "wcs_grouping",
        "shared_mask",
        "hotpants",
        "epsf",
        "sat_template",
        "subtract",
        "background_rough",
        "background_adaptive",
        "background_estimate",
        "forced_photometry",
    }
)


def _outputs_for_stage(stage: dict[str, Any]) -> list[str]:
    kind = stage.get("kind")
    if kind == "hotpants":
        o = stage.get("output") or {}
        labels = [o["diffs"], o["convolved"]]
        if o.get("bkg"):
            labels.append(o["bkg"])
        return labels
    if kind in (
        "epsf",
        "sat_template",
        "subtract",
        "background_rough",
        "background_adaptive",
        "background_estimate",
        "forced_photometry",
    ):
        return [stage["output"]]
    return []


def _inputs_refs(stage: dict[str, Any], idx: int) -> list[str]:
    kind = stage.get("kind")
    inp = stage.get("inputs") or {}
    refs = []
    if kind == "hotpants":
        if inp.get("bkg"):
            refs.append(inp["bkg"])
        if inp.get("convolved"):
            refs.append(inp["convolved"])
    elif kind == "epsf":
        d = inp.get("diffs")
        if d is not None and str(d).strip():
            refs.append(str(d).strip())
    elif kind == "sat_template":
        for key in ("diffs", "epsf"):
            v = inp.get(key)
            if v is not None and str(v).strip():
                refs.append(str(v).strip())
    elif kind == "subtract":
        ex = inp.get("expression")
        if isinstance(ex, str) and ex.strip():
            try:
                for _, lab in parse_subtract_expression(ex):
                    refs.append(lab)
            except ValueError as e:
                raise ValueError(
                    f"pipeline[{idx}] subtract: invalid inputs.expression: {e}"
                ) from e
        else:
            for key in ("science", "template"):
                v = inp.get(key)
                if v is not None and str(v).strip():
                    refs.append(str(v).strip())
    elif kind == "background_rough":
        for key in ("diffs", "bkg"):
            v = inp.get(key)
            if v is not None and str(v).strip():
                refs.append(str(v).strip())
    elif kind == "background_adaptive":
        for key in ("rough", "diffs", "bkg"):
            v = inp.get(key)
            if v is not None and str(v).strip():
                refs.append(str(v).strip())
    elif kind == "background_estimate":
        for key in ("diffs", "bkg"):
            v = inp.get(key)
            if v is not None and str(v).strip():
                refs.append(str(v).strip())
    elif kind == "forced_photometry":
        d = inp.get("diffs")
        if d is not None and str(d).strip():
            refs.append(str(d).strip())
        if inp.get("epsf"):
            refs.append(str(inp["epsf"]).strip())
    return refs


def validate_pipeline(cfg: SynDiffConfig) -> None:
    if not cfg.pipeline:
        raise ValueError(
            "Config has an empty ``pipeline`` list. Add a ``pipeline:`` section "
            "to your YAML (see syndiff_pipeline/example/recipe_*.yaml)."
        )

    available: set[str] = set()
    ext = getattr(cfg, "pipeline_external_workspace_labels", None) or []
    for lab in ext:
        if isinstance(lab, str) and lab.strip():
            available.add(lab.strip())

    for idx, stage in enumerate(cfg.pipeline):
        if not isinstance(stage, dict):
            raise ValueError(f"pipeline[{idx}] must be a mapping, got {type(stage).__name__}")
        kind = stage.get("kind")
        if kind not in STAGE_KINDS:
            raise ValueError(f"pipeline[{idx}]: unknown kind {kind!r}")

        for ref in _inputs_refs(stage, idx):
            # ``ffi`` in subtract expressions is virtual: cropped science from manifest paths,
            # not ``ws/ffi/``.
            if kind == "subtract" and ref == "ffi":
                continue
            if ref not in available:
                raise ValueError(
                    f"pipeline[{idx}] ({kind}): input label {ref!r} is not produced "
                    f"by an earlier stage (available: {sorted(available)!r})."
                )

        if kind == "hotpants":
            o = stage.get("output") or {}
            for req in ("diffs", "convolved"):
                if req not in o or not str(o[req]).strip():
                    raise ValueError(
                        f"pipeline[{idx}] hotpants: output.{req} is required (non-empty string label)."
                    )
            sci = str(stage.get("science", "ffi")).strip()
            if sci != "ffi" and sci not in available:
                raise ValueError(
                    f"pipeline[{idx}] hotpants: science workspace label {sci!r} is not available "
                    f"(available: {sorted(available)!r}). Use science: ffi for raw cropped FFIs."
                )

        if kind == "epsf":
            inp = stage.get("inputs") or {}
            if "diffs" not in inp:
                raise ValueError(f"pipeline[{idx}] epsf: inputs.diffs required")
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] epsf: output label required")

        if kind == "sat_template":
            inp = stage.get("inputs") or {}
            for req in ("diffs", "epsf"):
                if req not in inp:
                    raise ValueError(f"pipeline[{idx}] sat_template: inputs.{req} required")
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] sat_template: output label required")

        if kind == "subtract":
            inp = stage.get("inputs") or {}
            ex = inp.get("expression")
            has_expr = isinstance(ex, str) and ex.strip()
            if not has_expr:
                for req in ("science", "template"):
                    if req not in inp:
                        raise ValueError(
                            f"pipeline[{idx}] subtract: inputs.{req} required "
                            "(or use inputs.expression for a linear combination)"
                        )
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] subtract: output label required")

        if kind == "background_rough":
            inp = stage.get("inputs") or {}
            for req in ("diffs", "bkg"):
                if req not in inp:
                    raise ValueError(f"pipeline[{idx}] background_rough: inputs.{req} required")
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] background_rough: output label required")

        if kind == "background_adaptive":
            inp = stage.get("inputs") or {}
            for req in ("rough", "diffs", "bkg"):
                if req not in inp:
                    raise ValueError(f"pipeline[{idx}] background_adaptive: inputs.{req} required")
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] background_adaptive: output label required")

        if kind == "background_estimate":
            inp = stage.get("inputs") or {}
            for req in ("diffs", "bkg"):
                if req not in inp:
                    raise ValueError(f"pipeline[{idx}] background_estimate: inputs.{req} required")
            if "mode" not in stage:
                raise ValueError(f"pipeline[{idx}] background_estimate: mode required")
            if stage["mode"] != "rough_then_adaptive":
                raise ValueError(
                    f"pipeline[{idx}] background_estimate: only mode 'rough_then_adaptive' is implemented"
                )
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] background_estimate: output label required")

        if kind == "forced_photometry":
            inp = stage.get("inputs") or {}
            if "diffs" not in inp:
                raise ValueError(f"pipeline[{idx}] forced_photometry: inputs.diffs required")
            use_prf = str(stage.get("psf", "")).lower() == "prf"
            use_epsf = bool(inp.get("epsf"))
            if use_prf == use_epsf:
                raise ValueError(
                    f"pipeline[{idx}] forced_photometry: set exactly one of "
                    f"'psf: prf' or 'inputs.epsf: <label>'."
                )
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] forced_photometry: output workspace label required")

        for lab in _outputs_for_stage(stage):
            available.add(lab)
