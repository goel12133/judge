"""First human-vs-model comparison on Sandman (named-feeling channel).

    python scripts/run_sandman_model.py --hs results/hs_sandman_qwen25_7b.npz --name qwen25_7b

Inputs: the Colab extraction file (per-segment hidden states at every layer, last-token and
mean; per-segment surprisal; per-token top-64 next-token log-probs).
Outputs (results/sandman_{name}.json/.md and a PNG):
  1. Surprisal-only baselines: corr(segment surprisal mean/sum, human mean suspense).
  2. Token-level belief surprise (judge.beliefs.token_surprise_from_logits on the top-64
     renormalised distribution), averaged per segment, vs human suspense and vs human
     per-segment surprise proxy (|diff| of mean suspense).
  3. Per-layer linear probe (leave-k-segments-out blocked CV in 5 contiguous folds, since
     65 points is too few for a story split) from hidden states to human mean suspense,
     with shuffled-label and random-feature controls and the group ceiling 0.919 beside it.
  4. Timescale: cross-correlation lag between model token-surprise and human suspense.
  5. Reader variance (H4): probe to per-segment SD across readers.
Every number is exploratory (65 points; text likely in pretraining; English translation of
a German stimulus). None of this is the pre-registered test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from judge import beliefs, sandman

OUT = Path(__file__).resolve().parents[1] / "results"


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def blocked_probe_small(X, y, n_folds=5, alphas=(1, 10, 100, 1000, 10000, 100000)):
    """Contiguous-block CV over segments (no shuffling: autocorrelation)."""
    kf = KFold(n_splits=n_folds, shuffle=False)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m = RidgeCV(alphas=alphas).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    return corr(pred, y), pred


def xcorr_lag(a, b, max_lag=5):
    """Lag at which corr(a[t], b[t+lag]) peaks; positive = a leads b."""
    best = (0, -np.inf)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = corr(a[: len(a) - lag], b[lag:])
        else:
            r = corr(a[-lag:], b[: len(b) + lag])
        out[lag] = r
        if np.isfinite(r) and r > best[1]:
            best = (lag, r)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    s = sandman.load()
    R = s.ratings.values                      # (23, 65)
    y = R.mean(0)                             # human mean suspense
    y_sd = R.std(0)                           # reader disagreement
    y_surprise = np.abs(np.diff(y, prepend=y[0]))  # crude human surprise proxy

    z = np.load(args.hs)
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.endswith("_last"))

    # 1. surprisal baselines
    surp_mean, surp_sum = z["surprisal_mean"], z["surprisal_sum"]
    base = {"surprisal_mean_vs_suspense": corr(surp_mean, y),
            "surprisal_sum_vs_suspense": corr(surp_sum, y),
            "n_tokens_vs_suspense": corr(z["n_tokens"], y)}

    # 2. belief surprise from top-64 next-token distributions
    logp = z["tok_top64_logp"].astype(np.float32)        # (T, 64)
    p = np.exp(logp); p /= p.sum(1, keepdims=True)
    # distribution shift between consecutive positions over the union of ids is not
    # directly comparable (different support); use the proxy: entropy and its change
    ent = -(p * np.log(p + 1e-12)).sum(1)
    seg = z["tok_segment"]
    seg_ent = np.array([ent[seg == i].mean() if (seg == i).any() else np.nan for i in range(65)])
    seg_dent = np.abs(np.diff(seg_ent, prepend=seg_ent[0]))
    belief = {"entropy_vs_suspense": corr(seg_ent, y),
              "entropy_change_vs_human_surprise_proxy": corr(seg_dent, y_surprise),
              "surprisal_mean_vs_human_surprise_proxy": corr(surp_mean, y_surprise)}

    # 3. per-layer probes
    per_layer = {}
    for l in layers:
        for kind in ("last", "mean"):
            X = np.nan_to_num(z[f"layer_{l}_{kind}"].astype(np.float32))
            r, _ = blocked_probe_small(X, y)
            per_layer[f"{l}_{kind}"] = r
    best_key = max(per_layer, key=lambda k: np.nan_to_num(per_layer[k], nan=-9))
    bl, bkind = best_key.split("_")
    Xb = np.nan_to_num(z[f"layer_{bl}_{bkind}"].astype(np.float32))
    r_best, pred_best = blocked_probe_small(Xb, y)
    r_shuf, _ = blocked_probe_small(Xb, rng.permutation(y))
    r_rand, _ = blocked_probe_small(rng.standard_normal(Xb.shape), y)
    # inner-selected layer (choose layer on folds' training halves) is overkill at n=65;
    # report best-layer as exploratory and the median across layers as a conservative number
    r_median_layers = float(np.nanmedian(list(per_layer.values())))

    # 4. timescale
    (lag, lag_r), lag_table = xcorr_lag(surp_mean, y)

    # 5. reader variance
    r_sd, _ = blocked_probe_small(Xb, y_sd)
    r_sd_shuf, _ = blocked_probe_small(Xb, rng.permutation(y_sd))

    out = {
        "model": args.name, "n_segments": 65, "n_readers": 23,
        "human_group_ceiling_r": 0.919,
        "baselines": base, "belief_dynamics_proxies": belief,
        "probe_per_layer_r": per_layer,
        "probe_best": {"layer": int(bl), "kind": bkind, "r": r_best, "shuffled_label": r_shuf,
                       "random_feature": r_rand, "median_over_layers": r_median_layers},
        "timescale": {"best_lag_model_surprisal_leads_human": lag, "r_at_best_lag": lag_r,
                      "lag_table": {str(k): v for k, v in lag_table.items()}},
        "reader_variance_H4": {"r": r_sd, "shuffled": r_sd_shuf},
        "caveats": ["exploratory: 65 points, contiguous 5-fold CV, no story split possible",
                    "English machine translation of German stimulus",
                    "Hoffmann likely in pretraining; memorisation test pending"],
        "command": f"python scripts/run_sandman_model.py --hs {args.hs} --name {args.name}",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"sandman_{args.name}.json").write_text(json.dumps(out, indent=2))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        x = np.arange(1, 66)
        ax[0].plot(x, y, "k-", lw=2, label="human mean suspense (23 readers)")
        ax[0].fill_between(x, y - R.std(0) / np.sqrt(23), y + R.std(0) / np.sqrt(23), color="k", alpha=.15)
        ax[0].plot(x, pred_best, "r--", label=f"probe, layer {bl} {bkind} (r={r_best:.2f})")
        ax[0].legend(loc="upper left"); ax[0].set_ylabel("suspense")
        zs = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
        ax[1].plot(x, zs(y), "k-", lw=2, label="human suspense (z)")
        ax[1].plot(x, zs(surp_mean), "b-", alpha=.7, label=f"model mean surprisal (z), r={base['surprisal_mean_vs_suspense']:.2f}")
        ax[1].plot(x, zs(seg_ent), "g-", alpha=.7, label=f"model next-token entropy (z), r={belief['entropy_vs_suspense']:.2f}")
        ax[1].legend(loc="upper left"); ax[1].set_xlabel("segment"); ax[1].set_ylabel("z")
        fig.suptitle(f"Sandman: human vs {args.name} (exploratory; ceiling r=0.92)")
        fig.tight_layout()
        fig.savefig(OUT / f"sandman_{args.name}.png", dpi=150)
    except Exception as e:  # matplotlib optional
        print("no figure:", e)

    md = f"""# Sandman × {args.name} (exploratory)

- Human group ceiling r = 0.919; 65 segments, 23 readers.
- Surprisal baselines vs suspense: mean {base['surprisal_mean_vs_suspense']:.3f}, sum {base['surprisal_sum_vs_suspense']:.3f}
- Next-token entropy vs suspense: {belief['entropy_vs_suspense']:.3f}; entropy change vs human surprise proxy: {belief['entropy_change_vs_human_surprise_proxy']:.3f}
- Best probe: layer {bl} ({bkind}), r = {r_best:.3f}; shuffled-label {r_shuf:.3f}; random-feature {r_rand:.3f}; median over layers {r_median_layers:.3f}
- Timescale: model surprisal best lag = {lag} segments (positive = model leads), r = {lag_r:.3f}
- Reader variance (H4): r = {r_sd:.3f} (shuffled {r_sd_shuf:.3f})

Caveats: 65 points; contiguous-block CV; translated stimulus; likely pretraining contamination.
"""
    (OUT / f"sandman_{args.name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
