# Reader-response paper — status and founder tasks

Repo: `judge` (local to the build session until pushed; see FOUNDER_TODO item 4). Updated 2026-11-05.

## Status

- Engine built and tested (13 tests): Natural Stories loader + alignment, split-half noise ceiling, nuisance residualiser (surprisal + spillover, freq, length, position), blocked ridge probe with shuffled-label and random-feature controls and story bootstrap, EFK suspense/surprise.
- First real numbers (processing channel, Natural Stories, 180 subjects, 10 stories, 10,256 words): noise ceiling r = 0.861 (95% 0.712–0.885); nuisance R² = 0.445 out of sample; approx residual ceiling r ≈ 0.54.
- Pipeline dry run with stand-in n-gram features: r_resid = 0.10 (CI 0.08–0.19), controls ≈ 0. Not an LLM result; proves the machinery.
- **Named-feeling channel now in hand:** Lehne & Koelsch 2015 *Sandman* suspense ratings (23 readers × 65 segments, CC BY, literary fiction, German stimulus + English translation) loaded and tested. Group ceiling r = 0.919 (95% 0.861–0.952); a single reader agrees with the rest at only r ≈ 0.53. Suspense is a reliable crowd signal and a noisy individual one — the H4 setup.
- Literature and data record written: `research/related_work.md` (~30 entries, verified tags) and `research/data_sources.md` (20+ datasets, licences, URLs, contamination table). Key facts: no paper found that regresses frozen-LLM hidden states on human continuous affect ratings or reading times over literary fiction at sentence/word resolution; closest are Zhang et al. CCN 2025 (event-level surprise, 2 stories), Matlin et al. 2025 (prompted), Goodfire 2026 (synthetic stories). EFK formulas verified from the paper. **No literary stimulus is verifiably unseen**; lowest-risk are Narratives/Moth transcripts and the Delatorre purpose-written story. A per-text memorisation test is added to the controls.
- Hidden-state extraction script for Colab and per-layer probe script ready; smoke-tested on fake activations.
- Sandbox cannot reach OpenNeuro/OSF/Hugging Face; fMRI, eye-tracking and model downloads need Colab or the VPS.

---

# FOUNDER_TODO — things only you can do, each prepared so it takes minutes

Updated 2026-11-05. Order matters: 1 and 2 gate everything else.

## 1. Make the three open decisions (2 min, phone)

Reply with three words, or edit DESIGN.md:
- Primary model: **Llama-3.1-8B base** (default) / Gemma-2-9B / Qwen2.5-7B
- 70B extraction: **no for now** (default) / yes, budget S$___
- Phase 2 self-report study: **decide in January** (default) / yes / no

## 2. Send five mentor emails (20 min)

Goal: one person who has published at ACL/EMNLP/COLM/CogSci and will read two drafts and endorse on arXiv (cs.CL). Shortlist (verify names and current affiliations before sending; the research agent's `research/related_work.md` has the papers):

1. First author of "Do Language Models Agree with Human Perceptions of Suspense in Stories?" (arXiv 2508.15794). Closest paper; they named the exact gap we fill.
2. Mariya Toneva (MPI-SWS) — LLM–brain alignment during reading; her group reads this kind of draft.
3. Leila Wehbe (CMU) — Harry Potter fMRI; the dataset owner.
4. An NLP faculty member at NUS or NTU (Min-Yen Kan at NUS is the obvious first ask; local, and NS-aware).
5. David Wilmot / Frank Keller (Edinburgh) — suspense modelling as uncertainty reduction (ACL 2020).

Draft (adapt per person; attach `DESIGN.md` as PDF):

> Subject: Short design doc — do LLM representations predict where a passage lands on readers?
>
> Dear Prof. ___,
>
> I'm a Singaporean student currently in national service, working independently on a study you may find close to your own work: whether a language model's internal representations predict, moment by moment, what a human reader does and feels while reading fiction, using public reading-time, eye-tracking, fMRI and per-segment suspense-rating data as ground truth, with linear probes, noise ceilings, surprisal-controlled residuals and a pre-registered primary test.
>
> I have the Natural Stories processing channel aligned and baselined (noise ceiling r = 0.86, nuisance R² = 0.45) and the probe pipeline tested with controls; model hidden states are next. The one-page design is attached.
>
> I'm not asking for supervision. I'm asking whether you'd be willing to read two drafts over the next six months and, if the work holds up, endorse an arXiv submission in cs.CL. If this is too far from your interests, a one-line pointer to someone closer would be just as valuable.
>
> Thank you for your time,
> Vikram Goel
> [one line of credentials: Syndrome2Vec preprint, Swarm Colosseum; link to GitHub]

## 3. Confirm the code licence (1 min)

README says Apache-2.0 placeholder. Say "Apache" or "MIT".

## 4. Make the repo public on GitHub when you're ready (5 min)

Currently local to the build sandbox only. Create `judge` under your account, then I push. Public is fine now: nothing in it is a result yet, and a dated public history helps priority. Your call per hard rule 6.

## 5. Later: GPU and data downloads

The build sandbox cannot reach OpenNeuro, OSF or Hugging Face. Hidden-state extraction and the fMRI/eye-tracking downloads need either the VPS (check its egress) or a Colab session you start. Colab free tier handles 8B models in 8-bit. I'll have a notebook ready that extracts all layers for Natural Stories and writes a single `.npz`; you run one cell.

## Not yet needed

Paying readers (phase 2, January at earliest). arXiv account/endorsement (needs a draft first, ~January).

---

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

---

# Pre-registration (DRAFT — becomes binding at the commit named below)

Status: **draft, not yet binding.** It becomes binding when the founder commits it unchanged with the message `preregister: v1` after the three open decisions in DESIGN.md are made and *before* any held-out story is scored by a probe. After that commit, the primary test is run once. Everything else is exploratory and labelled so.

## Primary hypothesis (H1, processing channel)

A linear ridge probe from the hidden states of **[MODEL, to be fixed: default Llama-3.1-8B, base]** at **the layer selected by inner cross-validation on training stories** predicts per-word residual log reading time on Natural Stories (self-paced reading, 180 subjects, 10 stories) better than both controls, after regressing out surprisal (with 2-word spillover), log frequency, length and position.

- Test statistic: residual correlation r_resid, leave-one-story-out, bootstrap over stories (200 resamples).
- Success: the 95% interval of r_resid excludes 0 and its lower bound exceeds the upper bound of both the shuffled-label and random-feature controls.
- Reported alongside, always: the noise ceiling (currently r = 0.861, 95% 0.712–0.885), r_resid as a fraction of the approximate residual ceiling (0.544), and the raw r.
- Contamination: Natural Stories texts were written for the corpus (2013) and are on GitHub; treated as **probably in pretraining**. H1 on Natural Stories is therefore a replication-grade result; the confirmatory version runs on a verifiably unseen text set (see DESIGN.md, Controls 4) once one is aligned.

## Secondary hypotheses (exploratory until a later pre-registration)

- H2 Engagement channel: fMRI inter-subject-correlation-weighted response (Narratives, Alice, Harry Potter), same pipeline, HRF lag chosen on a dev story.
- H3 Named feeling: per-segment human suspense ratings (four replicated studies), probe vs. prompted model vs. EFK belief-dynamics measures computed from model probabilities.
- H4 Layer and size trends across model families.
- H5 Reader variance predictable from text.

## Fixed in advance

- Probe: ridge, alphas {0.1, 1, 10, 100, 1000, 10000}, chosen by inner CV on training stories.
- Features: last-token hidden state at every layer, mean over the word's subtokens.
- Folds: by story. Bootstrap: by story.
- Controls: shuffled labels within story (seed 0), Gaussian random features (seed 0).
- Multiple comparisons: one primary test. All secondary tests Holm-corrected within family.

## What would falsify

If r_resid's interval includes 0, or does not separate from controls, the processing-channel claim is false on this corpus and is published as such.
