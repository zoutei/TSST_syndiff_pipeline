"""Record applied zeroing operations without changing the image algorithm.

An operation's support is not the same as its finite changed pixels. Operations
may overlap; the first finite change gets exactly one owner. Footprint IDs are
cell-local; ledger IDs include the immutable input identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

SCHEMA_VERSION = 1
REASONS = {"background": 1, "catalog": 2, "saturation": 3}


def array_digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


# Snapshot at import: concurrent development must not label an already-running
# process with a newer file version read only when it finishes.
IMPLEMENTATION = {"cell": file_digest(__file__),
                  "band_utils": file_digest(Path(__file__).parents[1]/'band_utils.py')}


def exact_id(value):
    """Never recover a 64-bit catalogue ID from a rounded binary64 value."""
    if value is None or value is pd.NA: return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value): return None
        if abs(value) > 2**53: raise ValueError("Catalogue identifier arrived as lossy float")
    out = str(int(value))
    return None if out == "-1" else out


class CellLedger:
    def __init__(self, raw, identity):
        if np.asarray(raw).ndim != 2: raise ValueError("2D cell required")
        if not identity.get("cell") or not identity.get("combined_fingerprint"):
            raise ValueError("Cell and immutable combined fingerprint are required")
        self.identity = dict(identity)
        self.shape = tuple(raw.shape)
        self.input_digest = array_digest(raw)
        self.prefix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        self.regions = []
        self.geometry = {}
        self.selected_stage = np.zeros(self.shape, dtype=np.uint8)
        self.changed_stage = np.zeros(self.shape, dtype=np.uint8)
        self.changed_operation = np.zeros(self.shape, dtype=np.uint32)
        self.labels = None
        self.output_digest = None
        self.replay_verified = False

    def observe(self, before, support, *, reason, component, origin=(0, 0), trigger=None):
        mask = np.asarray(support, dtype=bool)
        if mask.shape != before.shape: raise ValueError("Mask/image shape mismatch")
        if reason not in REASONS: raise ValueError(reason)
        y, x = map(int, origin); h, w = before.shape
        if y < 0 or x < 0 or y+h > self.shape[0] or x+w > self.shape[1]:
            raise ValueError("Operation lies outside its cell")
        n = len(self.regions) + 1
        finite = mask & np.isfinite(before)
        changed = finite & (before != 0)
        nonfinite = mask & ~np.isfinite(before)
        target = np.s_[y:y+h, x:x+w]
        if np.any(self.changed_operation[target][changed]):
            raise ValueError("A finite pixel would be charged twice")
        self.changed_operation[target][changed] = n
        self.changed_stage[target][changed] = REASONS[reason]
        first = mask & (self.selected_stage[target] == 0)
        self.selected_stage[target][first] = REASONS[reason]
        vals = np.asarray(before[changed], dtype=np.float64)
        triggers = {} if trigger is None else dict(trigger)
        row = dict(region_id=f"{self.prefix}:{n}", operation=n, component=int(component), reason=reason,
                   y0=y, x0=x, height=h, width=w, support_pixels=int(mask.sum()),
                   changed_pixels=int(changed.sum()), nonfinite_pixels=int(nonfinite.sum()),
                   zero_flux_pixels=int((finite & (before == 0)).sum()),
                   signed_flux_change=float(vals.sum()), absolute_flux_change=float(np.abs(vals).sum()),
                   trigger_gaia_id=exact_id(triggers.get("source_id")),
                   trigger_ra=triggers.get("ra"), trigger_dec=triggers.get("dec"))
        self.regions.append(row)
        self.geometry[f"support_{n}"] = np.packbits(mask.ravel())

    def support(self, operation):
        row = self.regions[operation-1]
        return np.unpackbits(self.geometry[f"support_{operation}"],
                             count=row["height"]*row["width"]).reshape(row["height"], row["width"]).astype(bool)

    def finish(self, raw, result, expected=None):
        if raw.shape != self.shape or result.shape != self.shape: raise ValueError("Shape changed")
        if array_digest(raw) != self.input_digest: raise ValueError("Input was mutated")
        finite_change = np.isfinite(raw) & (raw != 0) & (result == 0)
        if not np.array_equal(finite_change, self.changed_operation > 0):
            raise ValueError("Missing or spurious finite pixel change")
        retained = self.selected_stage == 0
        if not np.array_equal(raw[retained], result[retained], equal_nan=True):
            raise ValueError("Unrecorded image mutation")
        if not np.all(result[~retained] == 0): raise ValueError("Selected pixels were not zeroed")
        if expected is not None and not np.array_equal(result, expected, equal_nan=True):
            raise ValueError("Replay differs from immutable published image")
        self.replay_verified = expected is not None
        self.output_digest = array_digest(result)

    def associate_centres(self, sources, *, support_radius_px=None, support_radius_column=None):
        """Associate catalogue centres and declared circular support with operations.

        A radius is a documented candidate-support convention, not a measurement
        of a star's full PSF. Unknown support remains unknown. No centre is clipped
        onto a cell edge, and no pixel flux is assigned to every associated star.
        """
        required = {"source_key", "pixel_x", "pixel_y"}
        if not required <= set(sources): raise ValueError(f"Missing {required-set(sources)}")
        if sources.source_key.duplicated().any(): raise ValueError("Duplicate source identity")
        src = sources.copy().reset_index(drop=True)
        src["centre_status"] = "outside"
        src["centre_changed_stage"] = 0
        src["support_status"] = "unknown" if support_radius_px is None and support_radius_column is None else "candidate_circle"
        links = []
        supports = {r["operation"]: self.support(r["operation"]) for r in self.regions}
        statuses = ["outside"] * len(src)
        stages = np.zeros(len(src), dtype=np.uint8)
        for source_index, row in enumerate(src.itertuples(index=False)):
            x, y = float(row.pixel_x), float(row.pixel_y)
            if not np.isfinite(x+y): continue
            ix, iy = int(np.rint(x)), int(np.rint(y))
            inside = 0 <= x < self.shape[1] and 0 <= y < self.shape[0]
            # Match historical rounding only for in-cell centres; never map an
            # outside catalogue source onto a boundary pixel.
            if inside:
                ix=min(ix,self.shape[1]-1); iy=min(iy,self.shape[0]-1)
                st = int(self.selected_stage[iy,ix]); ch = int(self.changed_stage[iy,ix])
                statuses[source_index] = (
                    "centre_removed" if ch in (2,3) else "background_zeroed" if ch==1 else
                    "selected_no_finite_change" if st else "unaffected")
                stages[source_index] = ch
            chosen_radius = (getattr(row,support_radius_column) if support_radius_column else support_radius_px)
            radius = 0.0 if chosen_radius is None else float(chosen_radius)
            if radius < 0 or not np.isfinite(radius): raise ValueError("Invalid support radius")
            for region in self.regions:
                if region["reason"] == "background" and not inside: continue
                x0,y0=region["x0"],region["y0"]
                if x+radius < x0 or y+radius < y0 or x-radius >= x0+region["width"] or y-radius >= y0+region["height"]:continue
                mask=supports[region["operation"]]
                centre_hit=bool(inside and x0<=ix<x0+mask.shape[1] and y0<=iy<y0+mask.shape[0] and mask[iy-y0,ix-x0])
                count=0;support_count=None
                if chosen_radius is not None:
                    lo_x=max(x0,int(np.floor(x-radius)));hi_x=min(x0+mask.shape[1],int(np.ceil(x+radius))+1)
                    lo_y=max(y0,int(np.floor(y-radius)));hi_y=min(y0+mask.shape[0],int(np.ceil(y+radius))+1)
                    if lo_x<hi_x and lo_y<hi_y:
                        yy,xx=np.ogrid[lo_y:hi_y,lo_x:hi_x]
                        disc=(xx-x)**2+(yy-y)**2 <= radius**2
                        count=int(np.count_nonzero(mask[lo_y-y0:hi_y-y0,lo_x-x0:hi_x-x0]&disc))
                        support_count=int(disc.sum())
                if not centre_hit and not count:continue
                links.append(dict(source_key=row.source_key,region_id=region["region_id"],reason=region["reason"],
                                  centre_in_support=centre_hit,candidate_overlap_pixels=count,
                                  support_radius_px=chosen_radius,support_in_region_bbox_pixels=support_count,
                                  association_status="centre_membership" if centre_hit else "possible_partial_support",
                                  stellar_flux_change=None))
        src["centre_status"] = statuses
        src["centre_changed_stage"] = stages
        return src, pd.DataFrame(links, columns=["source_key","region_id","reason","centre_in_support",
            "candidate_overlap_pixels","support_radius_px","support_in_region_bbox_pixels","association_status","stellar_flux_change"])

    def publish(self, root, *, sources=None, associations=None, catalogue_manifests=(), metadata=None, extra_tables=None):
        if self.output_digest is None: raise ValueError("Ledger not validated")
        params=dict(schema=SCHEMA_VERSION,implementation=IMPLEMENTATION,identity=self.identity,input_digest=self.input_digest,
                    output_digest=self.output_digest,catalogues=list(catalogue_manifests),metadata=metadata or {})
        # Tables participate in the ledger identity, independently of image IDs.
        for key,table in (("sources",sources),("associations",associations)):
            params[key+"_digest"]=None if table is None else hashlib.sha256(table.to_json(orient="table",index=False).encode()).hexdigest()
        extra_tables = extra_tables or {}
        for name,table in extra_tables.items():
            if not name.replace('_','').isalnum() or name in ('sources','associations','regions'):
                raise ValueError('Unsafe or reserved table name')
        params['extra_tables']={name:hashlib.sha256(table.to_json(orient='table',index=False).encode()).hexdigest()
                                for name,table in sorted(extra_tables.items())}
        key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:24]
        root=Path(root);root.mkdir(parents=True,exist_ok=True);dest=root/key
        with FileLock(str(root/(key+".lock"))):
            if dest.exists():
                validate_published(dest)
                return dest
            tmp=Path(tempfile.mkdtemp(prefix=".ledger-",dir=root))
            try:
                pd.DataFrame(self.regions).to_parquet(tmp/'regions.parquet',index=False)
                arrays=dict(self.geometry,selected_stage=self.selected_stage,changed_stage=self.changed_stage,
                            changed_operation=self.changed_operation)
                if self.labels is not None:arrays['footprint_labels']=self.labels
                np.savez_compressed(tmp/'geometry.npz',**arrays)
                if sources is not None:sources.to_parquet(tmp/'sources.parquet',index=False)
                if associations is not None:associations.to_parquet(tmp/'associations.parquet',index=False)
                for name,table in extra_tables.items():table.to_parquet(tmp/(name+'.parquet'),index=False)
                params.update(fingerprint=key,shape=list(self.shape),status="complete_cell_pixels",
                              source_accounting_status="not_supplied" if sources is None else "catalogue_scoped",
                              replay_verified=self.replay_verified,
                              files={p.name:file_digest(p) for p in tmp.iterdir()})
                (tmp/'manifest.json').write_text(json.dumps(params,indent=2,sort_keys=True))
                os.replace(tmp,dest)
            except BaseException:
                shutil.rmtree(tmp,ignore_errors=True);raise
        return dest


def validate_published(path):
    path=Path(path)
    m=json.loads((path/'manifest.json').read_text())
    if m.get('schema') != SCHEMA_VERSION or m.get('status') != 'complete_cell_pixels':
        raise ValueError("Incomplete/unknown ledger schema")
    for name,digest in m['files'].items():
        if Path(name).name!=name or file_digest(path/name)!=digest:raise ValueError("Corrupt ledger payload")
    return m


def recover_segmentation_union(raw, background_suppressed):
    """Recover saved SEP-union information only when it is identifiable.

    Outside the union the stage writes zero; inside it leaves the original
    pixel unchanged, including NaN. An originally zero pixel is ambiguous and
    must not be guessed: use a saved union or replay SEP for that cell instead.
    """
    if raw.shape != background_suppressed.shape:raise ValueError('Shape mismatch')
    if np.any(raw == 0):raise ValueError('Originally zero pixels make union recovery ambiguous')
    if np.isinf(raw).any() or np.isinf(background_suppressed).any():raise ValueError('Infinite input unsupported')
    union=background_suppressed != 0  # NaN != 0 preserves kept invalid pixels.
    predicted=np.where(union,raw,0)
    if not np.array_equal(predicted,background_suppressed,equal_nan=True):
        raise ValueError('Cache is not pure background zeroing')
    return union


def replay_cell(raw, uncert, mask, trigger_catalogue, identity, *, expected=None, cached_union=None,
                segmentation=None, bright_star_mag_threshold=13.0):
    """Reuse exact saved union or SEP result; execute original footprint_v1 logic.

    footprint_v1 uses only the union of SEP-positive and bright-mask pixels.
    Reconstructing that union is sufficient; it does not invent SEP objects.
    """
    from .. import band_utils as bu
    if cached_union is not None:
        if segmentation is not None:raise ValueError("Specify one cached segmentation representation")
        union=np.asarray(cached_union,dtype=bool)
        if union.shape!=raw.shape:raise ValueError("Cached segmentation shape mismatch")
        segmentation=bu.SepBackgroundResult(objects=np.zeros(0),segmap=union,mask_bright_stars=np.zeros_like(union))
    ledger=CellLedger(raw,identity)
    result,legacy=bu._remove_background_footprint_v1(raw.copy(),uncert,2.5,50,mask,True,trigger_catalogue,
        bright_star_mag_threshold,segmentation=segmentation,recorder=ledger)
    ledger.finish(raw,result,expected)
    return result,legacy,ledger
