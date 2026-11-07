# Sandman × qwen25_7b (exploratory)

- Human group ceiling r = 0.919; 65 segments, 23 readers.
- Surprisal baselines vs suspense: mean 0.012, sum 0.083
- Next-token entropy vs suspense: -0.020; entropy change vs human surprise proxy: -0.001
- Best probe: layer 9 (mean), r = 0.801; shuffled-label 0.257; random-feature -0.280; median over layers 0.749
- Permutation null (200 shuffles): mean -0.180, 95th pct 0.083, p = 0.005
- Nuisance (position, position², length, surprisal) alone: r = -0.369; position vs suspense r = 0.019
- Residual probe (hidden states → suspense after nuisance): r = 0.768
- Timescale: model surprisal best lag = 5 segments (positive = model leads), r = 0.113
- Reader variance (H4): r = 0.199 (shuffled -0.290)

Caveats: 65 points; contiguous-block CV; translated stimulus; likely pretraining contamination.
