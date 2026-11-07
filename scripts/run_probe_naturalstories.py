"""Probe LLM hidden states against Natural Stories reading times, every layer, with controls.

    python scripts/run_probe_naturalstories.py --hs results/hs_llama31_8b.npz --name llama31_8b

Per layer: residual r (leave-one-story-out), raw r, shuffled-label and random-feature
controls, bootstrap over stories for the best layer by inner selection. The model's own
surprisal replaces the corpus GPT-3 surprisal in the nuisance set so the residual is
model-consistent (DECISIONS.local.md). Writes results/probe_{name}.json and .md.

Layer selection for the pre-registered test: the layer with the best mean inner-fold
r_resid on training stories; here we report all layers (exploratory) and the inner-selected
layer's held-out number separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from judge import baselines, naturalstories as ns, probe

OUT = Path(__file__).resolve().parents[1] / "results"


def load_hs(path: str):
    z = np.load(path)
    keys = pd.DataFrame({"item": z["item"], "zone": z["zone"], "model_surprisal": z["surprisal"]})
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    return keys, LazyLayers(z, layers)


class LazyLayers:
    """Dict-like view that decompresses one layer at a time (29 layers x 10k x 3.6k float32
    is 4 GB; holding them all OOM-kills a small box)."""

    def __init__(self, z, layers):
        self.z, self.layers = z, layers

    def __getitem__(self, l):
        return self.z[f"layer_{l}"].astype(np.float32)

    def items(self):
        for l in self.layers:
            yield l, self[l]

    def keys(self):
        return list(self.layers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--inner_stride", type=int, default=4, help="candidate layers for inner selection: every k-th layer (compute budget)")
    ap.add_argument("--inner_splits", type=int, default=3)
    args = ap.parse_args()

    t = ns.load()
    words = ns.mean_rt(t)
    keys, layers = load_hs(args.hs)
    words = words.merge(keys, on=["item", "zone"], how="inner")
    # model-consistent surprisal
    words["surprisal"] = words["model_surprisal"]
    words = baselines.add_lags(words)
    ok = np.isfinite(words["surprisal"].values)
    words = words[ok].reset_index(drop=True)
    idx = words.index.values

    per_layer = {}
    for l, X in layers.items():
        X = X[np.isin(np.arange(len(keys)), idx)] if len(keys) != len(words) else X
        X = np.nan_to_num(X)
        r = probe.blocked_probe(words, X, "y")
        per_layer[l] = {"r_resid": r.r_resid, "r_raw": r.r_raw, "per_fold": r.per_fold, "alpha": r.alpha}
        print(f"layer {l:2d}: r_resid={r.r_resid:.3f} r_raw={r.r_raw:.3f}")

    # inner-selected layer: choose by training-fold performance within each outer fold
    groups = words["item"].values
    gkf = GroupKFold(n_splits=words["item"].nunique())
    outer_preds, outer_resid, chosen = np.zeros(len(words)), np.zeros(len(words)), []
    for tr, te in gkf.split(words, groups=groups):
        best_l, best = None, -np.inf
        for l in layers.keys()[::args.inner_stride]:
            X = np.nan_to_num(layers[l])
            inner = probe.blocked_probe(words.iloc[tr].reset_index(drop=True), X[tr], "y",
                                        n_splits=args.inner_splits)
            if inner.r_resid > best:
                best, best_l = inner.r_resid, l
        chosen.append(best_l)
        X = np.nan_to_num(layers[best_l])
        res = baselines.Residualiser().fit(words.iloc[tr], "y")
        ytr, yte = res.residuals(words.iloc[tr], "y"), res.residuals(words.iloc[te], "y")
        from sklearn.linear_model import RidgeCV
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        m = RidgeCV(alphas=(0.1, 1, 10, 100, 1000, 10000)).fit((X[tr] - mu) / sd, ytr)
        outer_preds[te] = m.predict((X[te] - mu) / sd)
        outer_resid[te] = yte
    r_sel = float(np.corrcoef(outer_preds, outer_resid)[0, 1])
    sel_layer = int(np.median(chosen))
    Xs = np.nan_to_num(layers[sel_layer])
    shuf = probe.shuffled_label_control(words, Xs, "y")
    rand = probe.random_feature_control(words, Xs, "y")
    boot = probe.bootstrap_over_groups(words, Xs, "y", n_boot=args.n_boot)

    out = {
        "model": args.name, "n_words": int(len(words)), "n_stories": int(words["item"].nunique()),
        "per_layer": per_layer,
        "inner_selected": {"r_resid_heldout": r_sel, "layers_chosen_per_fold": chosen,
                           "median_layer": sel_layer},
        "controls_at_selected_layer": {"shuffled_label": shuf.r_resid, "random_feature": rand.r_resid},
        "bootstrap_over_stories": boot,
        "nuisance": baselines.nuisance_columns(),
        "command": f"python scripts/run_probe_naturalstories.py --hs {args.hs} --name {args.name}",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"probe_{args.name}.json").write_text(json.dumps(out, indent=2))
    best_layer = max(per_layer, key=lambda l: per_layer[l]["r_resid"])
    md = f"""# Probe: {args.name} → Natural Stories log RT (residual after model surprisal + spillover, freq, length, position)

- Words {len(words)}, stories {words['item'].nunique()}; noise ceiling r = 0.861, approx residual ceiling 0.54 (see naturalstories_baseline.md)
- Inner-selected layer (pre-registered rule): held-out r_resid = {r_sel:.3f}; layers chosen per fold {chosen}
- Bootstrap over stories at layer {sel_layer}: median {boot['median']:.3f}, 95% {boot['ci95'][0]:.3f}–{boot['ci95'][1]:.3f}
- Controls at layer {sel_layer}: shuffled labels {shuf.r_resid:.3f}, random features {rand.r_resid:.3f}
- Best single layer (exploratory): {best_layer}, r_resid = {per_layer[best_layer]['r_resid']:.3f}
- Fraction of approx residual ceiling: {r_sel/0.544:.2f}
"""
    (OUT / f"probe_{args.name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
