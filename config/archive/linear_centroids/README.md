# Linear templates → centroids (phase 1)

Site configs for the **round-1 WCS bootstrap** path: simple linear templates
anchored to TESS FFI WCS, then kernel-fit diff through star centroids.

Full pipeline description: [docs/markdown/linear_centroids_pipeline.md](../../docs/markdown/linear_centroids_pipeline.md).

| File | Role |
|------|------|
| `pipeline_templates.yaml` | Build `templates_linear/` (`remap` point drift + `downsample` linear) |
| `diff_config.yaml` | Diff recipe: kernel_fit → hotpants → epsf → centroids on `diff_linear/` |
| `pipeline.yaml` | Condor wrapper for diff-only submit |

```bash
# 1. Templates (once per SCC, after mapping + ps1_process)
syndiff template submit \
  --config config/linear_centroids/pipeline_templates.yaml \
  --scc config/scc_my_lanes.csv \
  --stages remap,downsample \
  --run-id my_linear_templates

# 2. Diff through centroids (any SCC with templates_linear ready)
syndiff diff submit \
  --config config/linear_centroids/pipeline.yaml \
  --scc config/scc_my_lanes.csv \
  --stages diff \
  --run-id my_linear_centroids
```

Phase 2 (temporally varying WCS from centroids) is the next development task.
Phase 3 field **diff** already exists — see
[`pipeline_field_c3_k3_os1.yaml`](../pipeline_field_c3_k3_os1.yaml) and
[`diff_config_scc_c3_k3_multi_hp_epsf_os1.yaml`](../diff_config_scc_c3_k3_multi_hp_epsf_os1.yaml).
Phase 2 wires centroids into the remap drift input that those pipelines consume.
