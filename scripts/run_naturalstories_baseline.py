"""First real numbers for the processing channel (Natural Stories, self-paced reading).

Computes:
  1. split-half noise ceiling of per-word mean log RT across 180 subjects
  2. variance explained by nuisance regressors (surprisal, frequency, length, position,
     with 2-word spillover), fit leave-one-story-out
  3. the residual ceiling: how much reliable variance is left for a probe to explain
Writes results/naturalstories_baseline.json and .md. Run: python scripts/run_naturalstories_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from judge import baselines, ceiling, naturalstories as ns

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def main() -> None:
    t = ns.load()
    words = ns.mean_rt(t)
    words = baselines.add_lags(words)
    long = t.rts.copy()
    long["y"] = np.log(long["RT"])
    long["key"] = long["item"].astype(str) + ":" + long["zone"].astype(str)

    ceil = ceiling.split_half_ceiling(long, "WorkerId", "key", "y", n_splits=100)

    # leave-one-story-out nuisance fit
    gkf = GroupKFold(n_splits=10)
    res_all = np.zeros(len(words))
    r2_folds = []
    for tr, te in gkf.split(words, groups=words["item"]):
        r = baselines.Residualiser().fit(words.iloc[tr], "y")
        pred_res = r.residuals(words.iloc[te], "y")
        res_all[te] = pred_res
        r2_folds.append(1 - pred_res.var() / words.iloc[te]["y"].var())
    r2_oos = float(1 - res_all.var() / words["y"].var())

    # surprisal-only, for the record
    surp_only = []
    for tr, te in gkf.split(words, groups=words["item"]):
        r = baselines.Residualiser(["surprisal", "surprisal_lag1", "surprisal_lag2"]).fit(words.iloc[tr], "y")
        surp_only.append(r.r2(words.iloc[te], "y"))

    # residual reliability: split-half on residualised per-subject data is expensive; use the
    # approximation ceiling_resid ≈ sqrt(max(ceil_r2 - r2_oos, 0)) where ceil_r2 = ceiling^2
    # (reliable variance minus the part nuisance explains, under the assumption nuisance
    # explains reliable variance first). Report as approximate.
    ceil_r2 = ceil["median"] ** 2
    resid_ceiling_r = float(np.sqrt(max(ceil_r2 - r2_oos, 0.0)))

    out = {
        "dataset": "Natural Stories (Futrell et al. 2021), self-paced reading, log RT",
        "n_words": int(len(words)),
        "n_subjects": int(long["WorkerId"].nunique()),
        "n_stories": int(words["item"].nunique()),
        "noise_ceiling_r": {"median": ceil["median"], "ci95": ceil["ci95"],
                            "method": "split-half over subjects, Spearman-Brown, 100 splits"},
        "nuisance_r2_out_of_sample": r2_oos,
        "nuisance_r2_per_story": [float(x) for x in r2_folds],
        "surprisal_only_r2_per_story": [float(x) for x in surp_only],
        "residual_ceiling_r_approx": resid_ceiling_r,
        "nuisance_regressors": baselines.nuisance_columns(),
        "notes": "GPT-3 davinci surprisal from the corpus probs file; no LLM hidden states yet.",
    }
    (OUT / "naturalstories_baseline.json").write_text(json.dumps(out, indent=2))
    md = f"""# Natural Stories — processing channel baseline

- Words: {out['n_words']}, subjects: {out['n_subjects']}, stories: {out['n_stories']}
- Noise ceiling (split-half, Spearman–Brown): r = {ceil['median']:.3f} (95% {ceil['ci95'][0]:.3f}–{ceil['ci95'][1]:.3f})
- Nuisance regressors (surprisal + 2-word spillover, frequency, length, position), leave-one-story-out R² = {r2_oos:.3f}
- Surprisal alone (with spillover), per-story R² median = {np.median(surp_only):.3f}
- Approximate residual ceiling for a probe: r ≈ {resid_ceiling_r:.3f}

Reading: the reliable per-word signal is r ≈ {ceil['median']:.2f}; nuisance already explains R² ≈ {r2_oos:.2f}.
Whatever a probe adds has to be measured against the residual ceiling, not against 1.0.
"""
    (OUT / "naturalstories_baseline.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
