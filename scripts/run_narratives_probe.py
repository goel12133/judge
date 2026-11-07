"""Engagement channel: probe LLM hidden states against Narratives fMRI parcel time series.

    python scripts/run_narratives_probe.py --narr data/narr --hs data/hs/hs_narratives_llama31_8b.npz \
        --name llama31_8b [--layer 16] [--dev pieman]

Per story the target is the group-mean Schaefer-400 parcel time series (n_TR, 400), z-scored.
Features: per-word hidden states at one layer, placed at word onsets, convolved with the
canonical HRF and sampled at TRs (judge.narratives.words_to_trs), plus the nuisance set
convolved the same way: word rate (1 per word), model surprisal, word length. Residual
targets are what remains after the nuisance regressors (fit on training stories only).
Probe: ridge (RidgeCV, alphas fixed), multi-output over parcels, leave-one-story-out.
Ceiling: split-half ISC per parcel (Spearman-Brown). Headline: ISC-weighted mean held-out
r across parcels, reported next to the ISC-weighted ceiling; per-network table; shuffled
control (circular shift of the target within story by a random lag >= 20 TR, which keeps
autocorrelation) and random-feature control. HRF lag chosen on the dev story over
{-1.5, 0, +1.5, +3} s and then fixed.
Everything exploratory (H2); nothing here is the pre-registered test.
Writes results/narratives_{name}_layer{L}.json and .md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression

from judge import narratives as nr

OUT = Path(__file__).resolve().parents[1] / "results"
ALPHAS = (1, 10, 100, 1000, 10000, 100000)


def corr_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (a - a.mean(0)) / (a.std(0) + 1e-9); b = (b - b.mean(0)) / (b.std(0) + 1e-9)
    return (a * b).mean(0)


def build(story: nr.Story, hs: dict, layer: int, lag: float):
    key = f"{story.name}/layer_{layer}"
    X = np.nan_to_num(hs[key].astype(np.float32))
    surp = np.nan_to_num(hs[f"{story.name}/surprisal"].astype(np.float32))
    on = story.start_s
    n = min(len(on), len(X))
    on, X, surp = on[:n], X[:n], surp[:n]
    length = np.array([len(w) for w in story.words[:n]], float)
    Xtr = nr.words_to_trs(on, X, story.n_tr, story.tr, lag_s=lag)
    N = np.c_[nr.words_to_trs(on, np.ones(n), story.n_tr, story.tr, lag_s=lag),
              nr.words_to_trs(on, surp, story.n_tr, story.tr, lag_s=lag),
              nr.words_to_trs(on, length, story.n_tr, story.tr, lag_s=lag)]
    Y = story.group_mean
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    return Xtr, N, Y


def loso(stories, feats, nuis, targets, rng=None, shift=False, random_feats=False):
    """Leave-one-story-out; returns per-parcel held-out r (mean over stories, weighted by TRs)."""
    rs, ws = [], []
    for k in range(len(stories)):
        tr_idx = [i for i in range(len(stories)) if i != k]
        Xtr = np.vstack([feats[i] for i in tr_idx]); Ntr = np.vstack([nuis[i] for i in tr_idx])
        Ytr = np.vstack([targets[i] for i in tr_idx])
        Xte, Nte, Yte = feats[k], nuis[k], targets[k]
        if random_feats:
            Xtr = rng.standard_normal(Xtr.shape); Xte = rng.standard_normal(Xte.shape)
        if shift:
            s = rng.integers(20, len(Yte) - 20); Yte = np.roll(Yte, s, axis=0)
        lin = LinearRegression().fit(Ntr, Ytr)
        Rtr, Rte = Ytr - lin.predict(Ntr), Yte - lin.predict(Nte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, Rtr)
        pred = m.predict((Xte - mu) / sd)
        rs.append(corr_cols(pred, Rte)); ws.append(len(Yte))
    rs, ws = np.array(rs), np.array(ws, float)
    return (rs * ws[:, None]).sum(0) / ws.sum(), rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narr", required=True)
    ap.add_argument("--hs", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default: middle of the available layers")
    ap.add_argument("--dev", default="pieman")
    ap.add_argument("--n_boot", type=int, default=500)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    hs = dict(np.load(a.hs))
    stories = [nr.load_story(p) for p in sorted(Path(a.narr).glob("narr_*.npz"))]
    stories = [s for s in stories if f"{s.name}/surprisal" in hs]
    layers = sorted({int(k.split("layer_")[1]) for k in hs if "layer_" in k})
    layer = a.layer if a.layer is not None else layers[len(layers) // 2]
    labels = stories[0].labels
    isc = {s.name: nr.split_half_isc(s.bold) for s in stories}
    ceil = np.mean([isc[s.name] for s in stories], 0)          # per-parcel reliability of the group mean
    w = np.clip(ceil, 0, None); w = w / w.sum()

    # HRF lag on the dev story (within-story 5-fold contiguous CV, fit on train blocks)
    dev = [s for s in stories if s.name == a.dev]
    lag_table = {}
    if dev:
        d = dev[0]
        for lag in (-1.5, 0.0, 1.5, 3.0):
            X, N, Y = build(d, hs, layer, lag)
            folds = np.array_split(np.arange(d.n_tr), 5); rr = []
            for te in folds:
                tr = np.setdiff1d(np.arange(d.n_tr), te)
                lin = LinearRegression().fit(N[tr], Y[tr]); R = Y - lin.predict(N)
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
                m = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, R[tr])
                rr.append((corr_cols(m.predict((X[te] - mu) / sd), R[te]) * w).sum())
            lag_table[lag] = float(np.mean(rr))
        lag = max(lag_table, key=lag_table.get)
    else:
        lag = 0.0
    feats, nuis, targets = zip(*[build(s, hs, layer, lag) for s in stories])
    r_par, r_story = loso(stories, feats, nuis, targets)
    r_shift, _ = loso(stories, feats, nuis, targets, rng=rng, shift=True)
    r_rand, _ = loso(stories, feats, nuis, targets, rng=rng, random_feats=True)
    head = float((r_par * w).sum()); head_ceil = float((ceil * w).sum())
    # bootstrap over stories of the weighted headline
    boots = []
    for _ in range(a.n_boot):
        pick = rng.integers(0, len(stories), len(stories))
        boots.append(float((r_story[pick].mean(0) * w).sum()))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    nets = {}
    for net in sorted({nr.network_of(l) for l in labels}):
        m = np.array([nr.network_of(l) == net for l in labels])
        nets[net] = {"r": float(r_par[m].mean()), "ceiling": float(ceil[m].mean()), "n_parcels": int(m.sum())}
    out = {"model": a.name, "layer": layer, "lag_s": lag, "lag_table": lag_table,
           "stories": [s.name for s in stories], "n_subjects": {s.name: int(s.bold.shape[0]) for s in stories},
           "headline_isc_weighted_r": head, "headline_ci95_stories": ci, "headline_isc_weighted_ceiling": head_ceil,
           "fraction_of_ceiling": head / head_ceil if head_ceil else None,
           "controls": {"circular_shift": float((r_shift * w).sum()), "random_features": float((r_rand * w).sum())},
           "per_network": nets, "per_parcel_r": r_par.tolist(), "per_parcel_ceiling": ceil.tolist(),
           "command": f"python scripts/run_narratives_probe.py --narr {a.narr} --hs {a.hs} --name {a.name} --layer {layer}"}
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / f"narratives_{a.name}_layer{layer}.json", "w"), indent=1)
    md = [f"# Narratives (engagement) × {a.name}, layer {layer} (exploratory, H2)", "",
          f"- Stories: {', '.join(s.name for s in stories)}; subjects per story {out['n_subjects']}",
          f"- HRF lag chosen on {a.dev}: {lag:+.1f} s (table {lag_table})",
          f"- Headline ISC-weighted held-out r = {head:.3f} (95% over stories {ci[0]:.3f}–{ci[1]:.3f}); ISC-weighted ceiling {head_ceil:.3f}; fraction {head / head_ceil if head_ceil else float('nan'):.2f}",
          f"- Controls: circular-shift {out['controls']['circular_shift']:.3f}, random features {out['controls']['random_features']:.3f}",
          "", "| network | r | ceiling | parcels |", "|---|---|---|---|"]
    md += [f"| {k} | {v['r']:.3f} | {v['ceiling']:.3f} | {v['n_parcels']} |" for k, v in nets.items()]
    md += ["", "Nuisance (fit on training stories): HRF-convolved word rate, model surprisal, word length.",
           "Targets: group-mean parcel series; ceiling = split-half ISC of the group mean. Command in JSON."]
    (OUT / f"narratives_{a.name}_layer{layer}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
