"""
Validate config ``pipeline`` lists before execution.
"""

from __future__ import annotations

from typing import Any

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import (
    inherited_workspace_labels,
    split_pipeline,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import validate_stage_for_kind
from syndiff_pipeline.difference_imaging.support.subtract import parse_subtract_expression

STAGE_KINDS = frozenset(
    {
        "shared_mask",
        "hotpants",
        "kernel_fit",
        "convolved_templates",
        "kernel_subtract",
        "epsf",
        "sat_template",
        "subtract",
        "background",
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
    if kind == "kernel_fit":
        out = stage.get("output")
        return [str(out).strip()] if out and str(out).strip() else []
    if kind == "convolved_templates":
        return [stage["output"]]
    if kind == "kernel_subtract":
        o = stage.get("output") or {}
        labels = [o["diffs"]]
        if o.get("phot_bkg"):
            labels.append(o["phot_bkg"])
        return labels
    if kind in (
        "epsf",
        "sat_template",
        "subtract",
        "background",
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
    elif kind == "convolved_templates":
        v = (inp or {}).get("kernel_fit")
        if v is not None and str(v).strip():
            refs.append(str(v).strip())
    elif kind == "kernel_subtract":
        v = (inp or {}).get("convolved")
        if v is not None and str(v).strip():
            refs.append(str(v).strip())
    elif kind == "background":
        for key in ("diffs", "bkg", "bkg_in"):
            v = inp.get(key)
            if v is not None and str(v).strip():
                refs.append(str(v).strip())
        steps = stage.get("steps") or {}
        if isinstance(steps, dict):
            for step_name in ("spatial", "temporal", "strap"):
                step = steps.get(step_name) or {}
                if isinstance(step, dict):
                    save = step.get("save")
                    if save is not None and str(save).strip():
                        refs.append(str(save).strip())
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
            "to your YAML (see config/example/diff_config_*.yaml)."
        )

    available: set[str] = set()
    ext = getattr(cfg, "pipeline_external_workspace_labels", None) or []
    for lab in ext:
        if isinstance(lab, str) and lab.strip():
            available.add(lab.strip())

    preamble_labels, inherit_specs, executable_stages = split_pipeline(cfg.pipeline)
    for lab in preamble_labels:
        available.add(lab)
    for lab in inherited_workspace_labels(inherit_specs):
        available.add(lab)

    if not executable_stages:
        raise ValueError(
            "Config pipeline has no executable stages (only preamble entries). "
            "Add at least one stage with a 'kind:' key."
        )

    for idx, stage in executable_stages:
        kind = stage.get("kind")
        if kind == "wcs_grouping":
            raise ValueError(
                f"pipeline[{idx}]: wcs_grouping is not a differencing stage. "
                "Run the template pipeline (wcs_grouping → downsample) first; "
                "diff loads WCS handoff from cluster_template_job.json and "
                "syndiff_ffi_frames.csv in output_dir."
            )
        if kind not in STAGE_KINDS:
            raise ValueError(f"pipeline[{idx}]: unknown kind {kind!r}")

        validate_stage_for_kind(stage, idx, kind)

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

        if kind == "kernel_fit":
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] kernel_fit: output workspace label required")

        if kind == "convolved_templates":
            inp = stage.get("inputs") or {}
            if "kernel_fit" not in inp or not str(inp["kernel_fit"]).strip():
                raise ValueError(
                    f"pipeline[{idx}] convolved_templates: inputs.kernel_fit required"
                )
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(
                    f"pipeline[{idx}] convolved_templates: output workspace label required"
                )

        if kind == "kernel_subtract":
            inp = stage.get("inputs") or {}
            if "convolved" not in inp or not str(inp["convolved"]).strip():
                raise ValueError(
                    f"pipeline[{idx}] kernel_subtract: inputs.convolved required"
                )
            o = stage.get("output") or {}
            if "diffs" not in o or not str(o["diffs"]).strip():
                raise ValueError(
                    f"pipeline[{idx}] kernel_subtract: output.diffs required"
                )

        if kind == "background":
            inp = stage.get("inputs") or {}
            steps = stage.get("steps") or {}
            if not isinstance(steps, dict):
                raise ValueError(f"pipeline[{idx}] background: steps must be a mapping")
            spatial_on = bool((steps.get("spatial") or {}).get("enabled", True))
            temporal_on = bool((steps.get("temporal") or {}).get("enabled", True))
            strap_on = bool((steps.get("strap") or {}).get("enabled", True))
            bkg_in = str(inp.get("bkg_in") or "").strip()
            diffs_in = str(inp.get("diffs") or "").strip()
            if not spatial_on and not bkg_in:
                raise ValueError(
                    f"pipeline[{idx}] background: inputs.bkg_in required when "
                    "spatial step is disabled"
                )
            if spatial_on and not diffs_in and not bkg_in:
                raise ValueError(
                    f"pipeline[{idx}] background: inputs.diffs required when "
                    "spatial step is enabled"
                )
            if strap_on and not diffs_in:
                raise ValueError(
                    f"pipeline[{idx}] background: inputs.diffs required when "
                    "strap step is enabled"
                )
            if temporal_on and not spatial_on and not bkg_in:
                raise ValueError(
                    f"pipeline[{idx}] background: inputs.bkg_in required for "
                    "temporal step when spatial is disabled"
                )
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] background: output label required")

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

        if kind == "forced_photometry":
            inp = stage.get("inputs") or {}
            if "diffs" not in inp:
                raise ValueError(f"pipeline[{idx}] forced_photometry: inputs.diffs required")
            if "psf_type" in stage:
                raise ValueError(
                    f"pipeline[{idx}] forced_photometry: top-level 'psf_type' is no "
                    "longer supported; use a 'methods' list (see config/README.md)."
                )
            methods = stage.get("methods")
            if not methods or not isinstance(methods, list):
                raise ValueError(
                    f"pipeline[{idx}] forced_photometry: required non-empty 'methods' list"
                )
            seen: set[str] = set()
            needs_epsf = False
            for mi, entry in enumerate(methods):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"pipeline[{idx}] forced_photometry methods[{mi}]: must be a mapping"
                    )
                name = str(entry.get("name", "")).strip().lower()
                if not name:
                    raise ValueError(
                        f"pipeline[{idx}] forced_photometry methods[{mi}]: 'name' required"
                    )
                if name in seen:
                    raise ValueError(
                        f"pipeline[{idx}] forced_photometry: duplicate method name {name!r}"
                    )
                seen.add(name)
                mtype = str(entry.get("type", "")).strip().lower()
                if mtype == "psf":
                    pt = str(entry.get("psf_type", "")).strip().lower()
                    if pt not in ("epsf", "prf"):
                        raise ValueError(
                            f"pipeline[{idx}] forced_photometry methods[{mi}]: "
                            f"psf_type must be 'epsf' or 'prf'"
                        )
                    if pt == "epsf":
                        meth_inp = entry.get("inputs") or {}
                        if not (isinstance(meth_inp, dict) and meth_inp.get("epsf")):
                            if not inp.get("epsf"):
                                needs_epsf = True
                    elif pt == "prf":
                        meth_inp = entry.get("inputs") or {}
                        if isinstance(meth_inp, dict) and meth_inp.get("epsf"):
                            raise ValueError(
                                f"pipeline[{idx}] forced_photometry methods[{mi}]: "
                                "psf_type 'prf' must not set inputs.epsf"
                            )
                elif mtype == "aperture":
                    pass
                else:
                    raise ValueError(
                        f"pipeline[{idx}] forced_photometry methods[{mi}]: "
                        f"type must be 'psf' or 'aperture', got {entry.get('type')!r}"
                    )
            if needs_epsf and not inp.get("epsf"):
                raise ValueError(
                    f"pipeline[{idx}] forced_photometry: at least one psf method uses "
                    "psf_type 'epsf' but inputs.epsf workspace label is missing"
                )
            if "output" not in stage or not str(stage["output"]).strip():
                raise ValueError(f"pipeline[{idx}] forced_photometry: output workspace label required")

        for lab in _outputs_for_stage(stage):
            available.add(lab)
