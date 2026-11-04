# Where Does It Land? — Do LLMs Know What a Reader Feels

Design v0.2, 4 Nov 2026. Mirror of the shared Claude Doc; the doc is the editable copy, this file is the one sessions read.

## Question

Do language models represent, moment by moment, what a human reader feels and does while reading literature? Specifically: can a model's internal representations predict *where* in a passage an effect lands on readers, and does prompting the model fail where probing it succeeds?

## Why it is open

The suspense-replication paper (arXiv 2508.15794) showed prompted LLMs track the direction of suspense but miss turning points, and had only aggregate human ratings. Brain-encoding work (Toneva & Wehbe 2019; Schrimpf 2021; Caucheteux & King 2022; Goldstein 2022) maps LLM activations to reading responses to study language processing, not effect. The Alice fMRI emotion paper (arXiv 2606.07707) used an LLM to label emotion and the brain as the target. Nobody has used human reader response as ground truth and model internals as the thing under test, across response channels.

## Response channels (the dimensions)

A passage produces a vector of reader responses, each measured by a different public signal and meaning a different thing. They are kept as separate targets, never collapsed into one "effect".

| Channel | Signal | What it means | Source |
|---|---|---|---|
| Processing | Fixation duration, regressions, skips, reading time | Where the reader worked, re-read or skimmed | Natural Stories (SPR), ZuCo, Provo, GECO |
| Engagement | fMRI/MEG activity, inter-subject correlation | How strongly and how uniformly the passage grips | Harry Potter fMRI, Narratives, Alice |
| Named feeling | Per-segment ratings (tension, amusement, moved, unease) | What the reader reports feeling | Suspense studies; phase 2 |
| Expressed reaction | Comments, highlights at paragraph level | What mattered enough to say | Licensed platform data only |
| Outcome | Completion, return, re-read | Whether they kept going | Serial-fiction retention if licensed |

Expected result is a dissociation: model representations predict processing well (close to next-token surprise), engagement partly, named feeling poorly. Which channels a model predicts, at which layers, is the paper's core table; the surprise-controlled residual is the headline number per channel.

## Hypotheses

- H1: a linear probe on hidden states predicts per-segment reader response better than text-emotion baselines and better than the same model prompted.
- H2: the probe localises turning points (rise→fall) that prompting misses.
- H3: predictive power varies by layer and model size in a consistent way.
- H4: variance across readers is itself predictable from the text.
- H5 (belief dynamics): Ely–Frankel–Kamenica suspense and surprise computed from the model's own probabilities correlate with human suspense ratings; if not, the model's uncertainty is token-level, not story-level.

## Public data (phase 1, no recruitment)

Natural Stories (SPR, 180 subjects, 10 stories; **downloaded, aligned, loader + tests**); Harry Potter fMRI (Wehbe 2014); Narratives (Nastase 2021, ds002345); Alice (ds002322); ZuCo, Provo, GECO; the four suspense-rating studies via 2508.15794. Baselines: GoEmotions, NRC lexicons, story-arc sentiment. See `research/data_sources.md` for licences and URLs.

Caveat: brain and eye data measure processing load and arousal, not named feelings. Phase 1 answers "does the model know where the reader reacts"; phase 2 answers "with what".

## Method

1. Segment each text at word/sentence level; align responses (BOLD with HRF lag, fixations, ratings). Done for Natural Stories (`judge/naturalstories.py`).
2. Extract hidden states from 3–4 open models at several sizes at every layer.
3. Ridge probes with blocked CV across stories; report held-out residual r with bootstrap intervals and controls (`judge/probe.py`).
4. Same models prompted for per-segment affect; compare to probe.
5. Turning-point analysis: precision/recall on human-rated peaks.
6. Belief-dynamics measures (`judge/beliefs.py`) against suspense ratings.

## Intent-conditioned weighting

Channel importance is not set by hand. Intent is declared (genre label) or learned: a shared encoder, one head per channel, a small mixing layer conditioned on genre fit to the outcome channel. Learned weights are validated on held-out stories and claimed only for genres with outcome data.

## Controls (the section a reviewer checks first)

1. **Noise ceiling:** every probe r reported as a fraction of inter-subject reliability (`judge/ceiling.py`). Natural Stories: r = 0.861.
2. **Surprise confound:** probes fit on residuals after surprisal (with spillover), frequency, length, position (`judge/baselines.py`). Natural Stories: nuisance R² = 0.445 out of sample.
3. **Autocorrelation:** folds by story and subject; HRF lag chosen on a dev story.
4. **Contamination:** Harry Potter, Alice, Gutenberg texts are in pretraining. Results must replicate on verifiably unseen text or those sets are exploratory.
5. **Multiple comparisons:** pre-registered primary test (`PREREGISTRATION.md`); all others corrected and all reported.
6. **Probe capacity:** linear only, with shuffled-label and random-feature controls.
7. **Prompt fairness:** prompted baseline gets equal context and the best of several prompts chosen on a dev story.
8. **No circularity:** no model-generated labels as ground truth.
9. **Uncertainty:** bootstrap over subjects and stories; every number carries n and an interval.
10. **Reproducibility:** one command, pinned models, published segment boundaries and alignment code.

## Phase 2: dense self-report (small, cheap)

~30 readers, ~40 short passages (public-domain and model-generated), each reader marks felt response per sentence on a few axes with a keystroke while reading, plus re-reads and stops. Validates phase-1 probes against named feelings, measures inter-reader agreement per axis, tests H4. Consent, pay, withdrawal per hard rule 1. Only phase where all channels are measured on the same readers.

## Scope

One genre: narrative fiction. Transfer is paper two.

## Venue and timeline

COLM or EMNLP 2027 first; NeurIPS if H1–H3 hold with clean interpretability results; CogSci fallback. Nov: data + alignment + baselines. Dec: probes, prompted comparison, first result. Jan: turning points, cross-reader, phase 2 designed. Feb: phase 2. Mar–Apr: write-up, mentor reads two drafts. May–Jun: submit. arXiv preprint when phase 1 is clean.

## Compute

Sub-10B hidden-state extraction runs on free Colab or a CPU box (slow). 70B needs a rented GPU for hours. Founder decides.

## Open decisions (founder)

- Primary model family? (default: Llama-3.1-8B base)
- Spend on 70B extraction, or stay under 10B?
- Run phase 2, or publish phase 1 alone?
