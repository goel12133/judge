"""Named-feeling channel: inter-reader reliability of per-segment suspense (Sandman).

Writes results/sandman_ceiling.json/.md. Run: python scripts/run_sandman_ceiling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from judge import ceiling, sandman

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    s = sandman.load()
    c = ceiling.split_half_ceiling(s.long, "reader", "segment", "y", n_splits=500)
    R = s.ratings.values
    loo = np.array([np.corrcoef(R[i], np.delete(R, i, 0).mean(0))[0, 1] for i in range(R.shape[0])])
    m = s.mean_rating()
    d = np.sign(np.diff(m))
    turning = int(((d[1:] * d[:-1]) < 0).sum())
    out = {
        "dataset": "Lehne & Koelsch 2015, The Sandman (shortened German), 23 readers x 65 segments, suspense 1-10",
        "group_ceiling_r": {"median": c["median"], "ci95": c["ci95"], "method": "split-half over readers, Spearman-Brown, 500 splits"},
        "single_reader_vs_rest_r": {"median": float(np.median(loo)), "min": float(loo.min()), "max": float(loo.max())},
        "mean_curve_turning_points": turning,
        "per_segment_sd_mean": float(R.std(0).mean()),
        "contamination": "Hoffmann 1816 on Gutenberg (de/en); shortened edition likely partially memorised",
        "command": "python scripts/run_sandman_ceiling.py",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "sandman_ceiling.json").write_text(json.dumps(out, indent=2))
    md = f"""# Sandman — named-feeling channel ceiling

- 23 readers × 65 segments, suspense 1–10 (Lehne & Koelsch 2015, CC BY 4.0)
- Group ceiling (split-half, Spearman–Brown): r = {c['median']:.3f} (95% {c['ci95'][0]:.3f}–{c['ci95'][1]:.3f})
- Single reader vs. rest: median r = {np.median(loo):.2f} (range {loo.min():.2f}–{loo.max():.2f})
- Turning points in the mean curve: {turning} of 63 possible
- Mean per-segment SD across readers: {R.std(0).mean():.2f} rating points

Reading: suspense is a highly reliable crowd signal and a noisy individual one. A model
that predicts the group curve at r ≈ 0.9 would be at ceiling; the suspense-replication
paper's prompted models could not be compared this way because it had only aggregates.
Contamination: likely partial (Gutenberg); label in every table.
"""
    (OUT / "sandman_ceiling.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
