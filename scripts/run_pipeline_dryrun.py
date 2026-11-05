"""End-to-end dry run of the probe pipeline on Natural Stories with stand-in features.

No language model is available in the build sandbox, so the "representation" here is a
small hand-built feature set from the corpus itself (n-gram conditional log-probs of
orders 1–4 plus their lags, and a bag of 50 most frequent words). This is NOT a result
about LLMs; it verifies that blocked folds, residualisation, controls and bootstrap all
run on the real data and that the controls sit at ~0 while a weak real signal does not.
Writes results/pipeline_dryrun.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from judge import baselines, naturalstories as ns, probe

ROOT = ns.DEFAULT_ROOT
OUT = Path(__file__).resolve().parents[1] / "results"


def ngram_feats(words: pd.DataFrame) -> np.ndarray:
    cols = []
    for n in (2, 3, 4):
        f = pd.read_csv(ROOT / "freqs" / f"freqs-{n}.tsv", sep="\t", header=None,
                        names=["code", "n", "token", "count", "ctx"], quoting=3,
                        keep_default_na=False, na_values=["NA"])
        f = f[f["code"].str.endswith(".word")].copy()
        parts = f["code"].str.split(".", expand=True)
        f["item"] = parts[0].astype(int)
        f["zone"] = parts[1].astype(int)
        f[f"lp{n}"] = np.log((f["count"].astype(float) + 1) / (f["ctx"].astype(float) + 1e3))
        words = words.merge(f[["item", "zone", f"lp{n}"]], on=["item", "zone"], how="left")
        cols.append(f"lp{n}")
    words = baselines.add_lags(words, cols=tuple(cols), k=2)
    cols += [f"{c}_lag{l}" for c in cols for l in (1, 2)]
    top = words["word"].str.lower().str.strip(".,;:!?\"'").value_counts().index[:50]
    bag = np.stack([(words["word"].str.lower().str.strip(".,;:!?\"'") == t).astype(float).values
                    for t in top], 1)
    X = words[cols].astype(float)
    X = X.fillna(X.mean()).values
    return np.hstack([X, bag])


def main() -> None:
    t = ns.load()
    words = baselines.add_lags(ns.mean_rt(t))
    X = ngram_feats(words)
    real = probe.blocked_probe(words, X, "y")
    shuf = probe.shuffled_label_control(words, X, "y")
    rand = probe.random_feature_control(words, X, "y")
    boot = probe.bootstrap_over_groups(words, X, "y", n_boot=50)
    out = {
        "note": "Stand-in features (n-gram log-probs + 50-word bag), NOT an LLM result. Pipeline verification only.",
        "n_features": int(X.shape[1]),
        "real": {"r_resid": real.r_resid, "r_raw": real.r_raw, "per_fold": real.per_fold, "alpha": real.alpha},
        "shuffled_label_control": {"r_resid": shuf.r_resid},
        "random_feature_control": {"r_resid": rand.r_resid},
        "bootstrap_over_stories": boot,
        "command": "python scripts/run_pipeline_dryrun.py",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "pipeline_dryrun.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "real"}, indent=2))
    print("real r_resid", round(real.r_resid, 3), "per fold", [round(x, 3) for x in real.per_fold])


if __name__ == "__main__":
    main()
