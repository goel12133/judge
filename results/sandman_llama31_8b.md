# Sandman × llama31_8b (exploratory)

- Human group ceiling r = 0.919; 65 segments, 23 readers.
- Surprisal baselines vs suspense: mean 0.052, sum 0.098
- Next-token entropy vs suspense: -0.007; entropy change vs human surprise proxy: -0.019
- Best probe: layer 20 (mean), r = 0.810; shuffled-label -0.175; random-feature -0.235; median over layers 0.742
- Permutation null (200 shuffles): mean -0.157, 95th pct 0.123, p = 0.005
- Nuisance (position, position², length, surprisal) alone: r = -0.362; position vs suspense r = 0.019
- Residual probe (hidden states → suspense after nuisance): r = 0.739
- Timescale: model surprisal best lag = 5 segments (positive = model leads), r = 0.138
- Reader variance (H4): r = 0.055 (shuffled -0.313)

Caveats: 65 points; contiguous-block CV; translated stimulus; likely pretraining contamination.
