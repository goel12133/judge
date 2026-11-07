# Probe: qwen25_7b → Natural Stories log RT (residual after model surprisal + spillover, freq, length, position)

- Words 10256, stories 10; noise ceiling r = 0.861, approx residual ceiling 0.54 (see naturalstories_baseline.md)
- Inner-selected layer (pre-registered rule): held-out r_resid = 0.412; layers chosen per fold [4, 16, 16, 4, 4, 4, 4, 4, 16, 4]
- Bootstrap over stories at layer 4: median 0.412, 95% 0.356–0.465
- Controls at layer 4: shuffled labels 0.093, random features -0.004
- Best single layer (exploratory): 3, r_resid = 0.425
- Fraction of approx residual ceiling: 0.76
