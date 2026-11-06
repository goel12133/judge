# Parked directions (not the current paper)

Kept so they are not re-invented.

## Writing-judge directions (2026-11-04, before the pivot to reader response)

1. **Adversarial reward hardening.** Train a pairwise reward model on editor preferences; an attacker (best-of-N / small RL) finds passages the judge scores high and editors score low; retrain; report the gaming/recovery dynamic. Editors as a small per-round oracle. Possible paper two once a reader-response model exists to attack.
2. **Learn dimensions from edits.** Before/after editor edits; learn the decomposition; test against held-out preferences.
3. **Process over product.** Draft→read back→revise against the judge vs single pass at matched compute. Follow-on.

## Original six-dimension thesis

Register fit, Reception (context); Density, Specificity, Surprise, Effect (craft). The current paper is Effect, decomposed into response channels. The others remain the product thesis, not the paper.

## Rejected

- Cortical Labs / biological substrates as the "model": no language, cannot be shown text. Human brains are the ground truth here, not the model.
- Pairwise editor rating study as the main contribution: a dataset paper anyone can replicate with S$3k; no moat.
