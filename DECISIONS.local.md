# Decisions (one line each, with reason)

- 2026-11-04 — Paper pivoted from pairwise editor ratings (parked in `research/parked_directions.md`) to reader-response prediction with public data: a method result, not a dataset result; no recruitment needed for phase 1.
- 2026-11-04 — Response channels kept separate (processing, engagement, named feeling, expressed, outcome): each signal means something different; collapsing them would hide the expected dissociation.
- 2026-11-04 — Natural Stories is the first dataset: only one reachable from this sandbox (GitHub), and it ships GPT-3 surprisal and 180 subjects, enough for ceiling + nuisance baseline today.
- 2026-11-04 — Surprisal regressor: GPT-3 davinci logprobs shipped with the corpus, aligned by character offset to `all_stories.tok`. Replace with the probed model's own surprisal when hidden states are extracted, so the residual is model-consistent.
- 2026-11-04 — Residual ceiling reported as sqrt(ceiling² − nuisance R²) and labelled approximate; the exact residual split-half needs per-subject residualisation, queued.
- 2026-11-04 — Log RT, not raw RT, with the corpus README's filters (100–3000 ms, ≥5/6 comprehension).
- 2026-11-04 — Default primary model Llama-3.1-8B base (open, widely probed, fits free Colab); founder may override before pre-registration.
- 2026-11-04 — Belief-dynamics measures implemented generically (any belief process) so the same code scores humans and models; exact EFK normalisation flagged VERIFY in `judge/beliefs.py`.
- 2026-11-06 — Founder decisions: primary model Llama-3.1-8B base; 70B extraction yes (budget to be set at first rental); phase 2 runs at our own pace, January 2027 is the latest start. Code licence MIT.
- 2026-11-06 — Working doc moved to Google Docs with tabs (paper draft, design, pre-registration, status, mentors, resources); the repo markdown stays the source of truth for sessions, the doc is the founder's editing surface.
2026-11-06 — Colab extraction: LM head run in 256-token chunks (vocab-sized logits never on GPU); sequences over 4096 tokens use a sliding window (4096 context, 1024 step) because SDPA on a T4 materialises T×T attention. Natural Stories items are short, full context; Sandman (~9.5k tokens with Qwen tokenizer) is windowed — state in every Sandman table.
2026-11-06 — HF login made non-fatal: Qwen2.5-7B is ungated; a bad token only blocks Gemma/Llama.
2026-11-07 — Inner layer selection on Natural Stories uses every 4th layer and 3 inner splits (2 CPUs, ~30 s per ridge fit); the full 29-layer sweep is reported separately. Pre-registered run on Llama will use the same budget unless a GPU box is rented.
