# Pre-registration (DRAFT — becomes binding at the commit named below)

Status: **draft, not yet binding.** It becomes binding when the founder commits it unchanged with the message `preregister: v1` *before* any held-out story is scored by a probe. After that commit, the primary test is run once. Everything else is exploratory and labelled so.

## Primary hypothesis (H1, processing channel)

A linear ridge probe from the hidden states of **Llama-3.1-8B (base)** at **the layer selected by inner cross-validation on training stories** predicts per-word residual log reading time on Natural Stories (self-paced reading, 180 subjects, 10 stories) better than both controls, after regressing out surprisal (with 2-word spillover), log frequency, length and position.

- Test statistic: residual correlation r_resid between held-out predictions and held-out residual reading times, leave-one-story-out (10 folds); 95% interval by bootstrap over stories (1,000 resamples of the held-out prediction/target pairs, probes not refit).
- Success: the 95% interval of r_resid excludes 0 and its lower bound exceeds the upper bound of the same bootstrap interval for both the shuffled-label and random-feature controls (computed at the median selected layer).
- Reported alongside, always: the noise ceiling (currently r = 0.861, 95% 0.712–0.885), r_resid as a fraction of the approximate residual ceiling (0.544), and the raw r.
- Contamination: Natural Stories texts were written for the corpus (2013) and are on GitHub; treated as **probably in pretraining**. H1 on Natural Stories is therefore a replication-grade result; the confirmatory version runs on a verifiably unseen text set (see DESIGN.md, Controls 4) once one is aligned.

## Secondary hypotheses (exploratory until a later pre-registration)

- H2 Engagement channel: fMRI inter-subject-correlation-weighted response (Narratives, Alice, Harry Potter), same pipeline, HRF lag chosen on a dev story.
- H3 Named feeling: per-segment human suspense ratings (four replicated studies), probe vs. prompted model vs. EFK belief-dynamics measures computed from model probabilities.
- H4 Layer and size trends across model families.
- H5 Reader variance predictable from text.

## Fixed in advance

- Probe: ridge, alphas {0.1, 1, 10, 100, 1000, 10000}, chosen by generalised CV on training stories (sklearn RidgeCV); features standardised on training stories.
- Layer selection: inside each outer fold, candidate layers {0, 4, 8, ..., 32} (every 4th layer, a compute budget for a 2-CPU box), scored by 3-fold grouped CV on the training stories; the best candidate is used for that fold. Controls and the bootstrap are computed at the median of the selected layers.
- Features: hidden state at every layer, mean over the word's subtokens (8-bit weights on a T4; Natural Stories items processed with full context).
- Nuisance regressors (fit on training stories only): the probed model's own surprisal with 2-word spillover, log frequency with 2-word spillover, word length with 2-word spillover, story position.
- Folds: by story. Bootstrap: by story over held-out predictions.
- Controls: shuffled labels within story (seed 0), Gaussian random features (seed 0), each with the same leave-one-story-out probe and the same story bootstrap.
- Command: `python scripts/run_probe_naturalstories.py --hs data/hs/hs_naturalstories_llama31_8b.npz --name llama31_8b` at the repository state of the binding commit; output `results/probe_llama31_8b.{json,md}`.
- Prior exploratory runs: Qwen2.5-7B was scored on the same held-out stories on 7 Nov 2026 before this commit (results/probe_qwen25_7b.md). That run informed the layer-candidate budget and the bootstrap method, not the hypothesis, the nuisance set or the success criterion. Llama-3.1-8B has not scored any Natural Stories held-out story.
- Multiple comparisons: one primary test. All secondary tests Holm-corrected within family.

## What would falsify

If r_resid's interval includes 0, or does not separate from controls, the processing-channel claim is false on this corpus and is published as such.
