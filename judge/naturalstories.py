"""Loader for the Natural Stories corpus (Futrell et al. 2021).

Repository: https://github.com/languageMIT/naturalstories (CC BY-NC-SA 4.0).
We use:
  naturalstories_RTS/processed_RTs.tsv   per-subject, per-word self-paced reading times
  probs/all_stories_gpt3.csv             GPT-3 davinci token log-probs (surprisal regressor)
  freqs/freqs-1.tsv                      Google Books unigram frequencies
  words.tsv                              canonical token codes

The processing channel for the paper: reading time per word, 181 subjects, 10 stories.
Alignment is by (story, position) which both RT and prob files carry (item/zone in the
RT file; story/offset ordering in the probs file). Punctuation-attached tokens in the
probs file are merged to the RT-file 'zone' granularity by cumulative character offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "naturalstories"


@dataclass
class WordTable:
    """One row per (story, zone) word with nuisance regressors."""
    words: pd.DataFrame      # columns: item, zone, word, length, position, surprisal, log_freq
    rts: pd.DataFrame        # columns: WorkerId, item, zone, RT (filtered)


def _load_words(root: Path) -> pd.DataFrame:
    w = pd.read_csv(root / "words.tsv", sep="\t", header=None, names=["code", "token"],
                    quoting=3, keep_default_na=False)
    parts = w["code"].str.split(".", expand=True)
    w["item"] = parts[0].astype(int)
    w["zone"] = parts[1].astype(int)
    w["kind"] = parts[2]
    whole = w[w["kind"] == "whole"][["item", "zone", "token"]].rename(columns={"token": "word"})
    whole = whole.reset_index(drop=True)
    # words.tsv puts a space before attached punctuation ("England ,"); the GPT-3 probs
    # were computed on the RT distribution text (all_stories.tok, "England,"), so take the
    # surface form from there for character-offset alignment.
    tok = pd.read_csv(root / "naturalstories_RTS" / "all_stories.tok", sep="\t",
                      quoting=3, keep_default_na=False)
    tok = tok.rename(columns={"word": "surface"})
    whole = whole.merge(tok[["item", "zone", "surface"]], on=["item", "zone"], how="left")
    whole["word"] = whole["surface"].fillna(whole["word"])
    whole = whole.drop(columns="surface")
    whole["length"] = whole["word"].str.replace(r"[^A-Za-z']", "", regex=True).str.len()
    whole["position"] = whole.groupby("item").cumcount()
    return whole


def _load_freqs(root: Path) -> pd.DataFrame:
    f = pd.read_csv(root / "freqs" / "freqs-1.tsv", sep="\t", header=None,
                    names=["code", "n", "token", "count", "ctx"], quoting=3,
                    keep_default_na=False, na_values=["NA"])
    f = f[f["code"].str.endswith(".word")].copy()
    parts = f["code"].str.split(".", expand=True)
    f["item"] = parts[0].astype(int)
    f["zone"] = parts[1].astype(int)
    f["log_freq"] = np.log(f["count"].astype(float).clip(lower=1.0))
    return f[["item", "zone", "log_freq"]]


def _load_surprisal(root: Path, words: pd.DataFrame) -> pd.DataFrame:
    """GPT-3 token logprobs → per-zone surprisal (sum over subtokens of a zone).

    The probs file is tokenised at the GPT-3 BPE level with character offsets per
    story. We rebuild each story's text from the RT-level words joined by single spaces
    (which is how the corpus distributed it) and assign each BPE token to the zone whose
    character span contains the token's offset.
    """
    p = pd.read_csv(root / "probs" / "all_stories_gpt3.csv", keep_default_na=False)
    p["story"] = p["story"].astype(int) + 1  # probs file is 0-indexed, words.tsv is 1-indexed
    p["logprob"] = pd.to_numeric(p["logprob"], errors="coerce")
    out = []
    for item, grp in words.groupby("item"):
        spans = []
        cursor = 0
        for _, r in grp.iterrows():
            start = cursor
            end = start + len(r["word"])
            spans.append((start, end, r["zone"]))
            cursor = end + 1  # single space
        starts = np.array([s[0] for s in spans])
        zones = np.array([s[2] for s in spans])
        toks = p[p["story"] == item]
        # BPE tokens carry their leading space; the offset points at that space, so
        # shift to the first visible character before mapping to a zone.
        lead = toks["token"].str.len() - toks["token"].str.lstrip().str.len()
        first_char = toks["offset"].values + lead.values
        # token offset → zone index: last span whose start <= first visible char
        idx = np.searchsorted(starts, first_char, side="right") - 1
        idx = np.clip(idx, 0, len(zones) - 1)
        surp = pd.DataFrame({"zone": zones[idx], "surprisal": -toks["logprob"].values})
        surp = surp.groupby("zone", as_index=False)["surprisal"].sum(min_count=1)
        surp["item"] = item
        out.append(surp)
    return pd.concat(out, ignore_index=True)


def load(root: Path | str = DEFAULT_ROOT, rt_min: float = 100, rt_max: float = 3000,
         min_correct: int = 5) -> WordTable:
    """Load words with regressors and filtered per-subject RTs.

    Filters follow the corpus README: drop RTs outside [100, 3000] ms and subjects who
    answered fewer than `min_correct` of 6 comprehension questions on a story.
    """
    root = Path(root)
    words = _load_words(root)
    words = words.merge(_load_freqs(root), on=["item", "zone"], how="left")
    words = words.merge(_load_surprisal(root, words), on=["item", "zone"], how="left")
    rts = pd.read_csv(root / "naturalstories_RTS" / "processed_RTs.tsv", sep="\t",
                      usecols=["WorkerId", "correct", "item", "zone", "RT"])
    rts = rts[(rts["RT"] >= rt_min) & (rts["RT"] <= rt_max) & (rts["correct"] >= min_correct)]
    rts = rts.merge(words[["item", "zone"]], on=["item", "zone"], how="inner")
    return WordTable(words=words, rts=rts)


def mean_rt(table: WordTable, log: bool = True) -> pd.DataFrame:
    """Per-word mean (log) RT across subjects, with n subjects per word."""
    r = table.rts.copy()
    r["y"] = np.log(r["RT"]) if log else r["RT"]
    agg = r.groupby(["item", "zone"], as_index=False).agg(y=("y", "mean"), n=("y", "size"))
    return table.words.merge(agg, on=["item", "zone"], how="inner")
