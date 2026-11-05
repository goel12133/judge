"""Loader for the Lehne & Koelsch (2015) *Sandman* suspense ratings (named-feeling channel).

Source: PLoS ONE 10(5): e0124550, CC BY 4.0. 23 readers rated suspense (1–10) after
each of 65 segments of a shortened German version of E.T.A. Hoffmann's "Der Sandmann"
(6,859 words). The per-reader table and the segmented texts are bundled in the
SuspensePerception repo (Matlin et al. 2025, github.com/eilab-gt/SuspensePerception,
research/lehne2015/); the original PLOS supplements are the canonical source.

Text: German segments are the actual stimulus (65 paragraphs in lehne_de.md). The
English file is a machine translation with "N " line prefixes; the English is a convenience
for English-only models and is a pre-registered confound (DESIGN.md controls).
Contamination: Hoffmann (1816) is on Gutenberg in German and English; the shortened
edition is likely partially memorised. Label as such in every table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = (Path(__file__).resolve().parents[1] / "data" / "raw" / "SuspensePerception"
                / "research" / "lehne2015")


@dataclass
class Sandman:
    ratings: pd.DataFrame     # (23 readers, 65 segments), 1–10
    segments_de: list[str]    # 65 German segments (the stimulus)
    segments_en: list[str]    # 65 English segments (machine translation), aligned by number

    @property
    def long(self) -> pd.DataFrame:
        r = self.ratings.reset_index().melt(id_vars="reader", var_name="segment", value_name="y")
        r["segment"] = r["segment"].astype(int)
        return r

    def mean_rating(self) -> np.ndarray:
        return self.ratings.mean(0).values


def _paragraphs(path: Path) -> list[str]:
    t = path.read_text(encoding="utf-8")
    return [s.strip() for s in re.split(r"\n\s*\n", t) if s.strip()]


def load(root: Path | str = DEFAULT_ROOT) -> Sandman:
    root = Path(root)
    x = pd.read_excel(root / "lehne_human_results.xls", sheet_name="Sheet1", header=None)
    if x.shape != (23, 65):
        raise ValueError(f"unexpected rating table shape {x.shape}")
    x.index.name = "reader"
    x.columns = range(1, 66)
    de = _paragraphs(root / "lehne_de.md")
    if len(de) != 65:
        raise ValueError(f"expected 65 German segments, got {len(de)}")
    en_raw = _paragraphs(root / "lehne_de_to_en.md")
    en: dict[int, str] = {}
    for p in en_raw:
        m = re.match(r"^(\d+)\s+(.*)$", p, re.S)
        if m:
            en[int(m.group(1))] = m.group(2).strip()
        elif en:
            # continuation of the previous numbered segment
            k = max(en)
            en[k] = en[k] + " " + p
    en_list = [en.get(i, "") for i in range(1, 66)]
    return Sandman(ratings=x, segments_de=de, segments_en=en_list)
