"""Memorisation signature from the model's own surprisal (DESIGN.md, divergence test 2).

    python scripts/run_memorisation.py --sandman data/hs/hs_sandman_qwen25_7b.npz \
        --naturalstories data/hs/hs_naturalstories_qwen25_7b.npz --name qwen25_7b

A model that has memorised a text predicts long stretches of it almost exactly: many
tokens with surprisal near zero, and long runs of them. We report, per text:
  mean surprisal, median, fraction of positions with surprisal < 0.05 nats (p > 0.95),
  fraction < 0.5 nats, and the longest run of consecutive positions below 0.5 nats.
Sandman is per token (English machine translation of Hoffmann's German, so a verbatim
copy of the Gutenberg English translation would not match either); Natural Stories is per
word (sum over the word's tokens), which inflates the unit relative to tokens, so the two
columns are not directly comparable. The comparison that matters is within a unit: a
memorised text shows a run length far above the others. Verbatim prefix-completion remains
the complementary test and needs a generation pass (not done here).
Writes results/memorisation_{name}.json and .md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results"


def stats(s: np.ndarray, unit: str) -> dict:
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    low = s < 0.5
    best = run = 0
    for v in low:
        run = run + 1 if v else 0
        best = max(best, run)
    return {"unit": unit, "n": int(len(s)), "mean": float(s.mean()), "median": float(np.median(s)),
            "frac_lt_0.05": float((s < 0.05).mean()), "frac_lt_0.5": float(low.mean()),
            "longest_run_lt_0.5": int(best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandman", required=True)
    ap.add_argument("--naturalstories", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = {"model": a.name}
    z = np.load(a.sandman)
    tok = z["tok_surprisal"]
    out["sandman_tokens"] = stats(tok[1:], "token")        # position 0 has no prediction
    n = np.load(a.naturalstories)
    out["naturalstories_words"] = stats(n["surprisal"], "word")
    per_story = {}
    for it in np.unique(n["item"]):
        m = n["item"] == it
        per_story[int(it)] = stats(n["surprisal"][m], "word")
    out["naturalstories_per_story"] = per_story
    # Sandman by thirds of the story: memorised text often shows surprisal falling as the
    # model "locks on"; a fresh text does not.
    T = len(tok) - 1
    out["sandman_by_third"] = [stats(tok[1 + i * T // 3: 1 + (i + 1) * T // 3], "token") for i in range(3)]
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / f"memorisation_{a.name}.json", "w"), indent=1)
    s, w = out["sandman_tokens"], out["naturalstories_words"]
    md = f"""# Memorisation signature × {a.name}

| text | unit | n | mean surprisal | median | frac < 0.05 | frac < 0.5 | longest run < 0.5 |
|---|---|---|---|---|---|---|---|
| Sandman (EN translation) | token | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | {s['frac_lt_0.05']:.3f} | {s['frac_lt_0.5']:.3f} | {s['longest_run_lt_0.5']} |
| Natural Stories | word | {w['n']} | {w['mean']:.2f} | {w['median']:.2f} | {w['frac_lt_0.05']:.3f} | {w['frac_lt_0.5']:.3f} | {w['longest_run_lt_0.5']} |

Sandman by thirds (mean surprisal): {', '.join(f"{t['mean']:.2f}" for t in out['sandman_by_third'])}
Natural Stories longest run per story: {', '.join(str(v['longest_run_lt_0.5']) for v in per_story.values())}

Reading: a verbatim-memorised text would show a long run of near-zero surprisal (tens of
tokens) and a falling mean across thirds. Units differ between rows (token vs word), so
compare run lengths within a text family, not across. Command: see docstring.
"""
    (OUT / f"memorisation_{a.name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
