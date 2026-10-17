"""Plot the measured temporal trace association, without fitting a temporal law."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path("/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016")
m=pd.read_csv(R/"shape_response_metrics.csv")
f=pd.read_csv(R/"inputs/frames.csv")
d=m.merge(f[["frame_index","jitter_trace"]],left_on="frame",right_on="frame_index")
ref=d[d.frame==602].iloc[0]
r=np.corrcoef(d.jitter_trace,d.trace)[0,1]
fig,ax=plt.subplots(figsize=(7,5))
for orb,g in d.groupby("orbit"):
    ax.scatter((g.jitter_trace-ref.jitter_trace)*1000,(g.trace-ref.trace)*1000,label=f"Orbit {orb}")
ax.set(xlabel="Camera jitter trace change [10⁻³ pixel²]",ylabel="Equivalent PSF trace response change [10⁻³ pixel²]",title=f"24 snapshots: Pearson r = {r:.3f}\nAssociation; no temporal law fitted")
ax.legend();fig.tight_layout();fig.savefig(R/"figures/jitter_association.png",dpi=160)
