# judge

Do language-model representations predict, moment by moment, what a human reader does and feels while reading fiction?

A research repo. Study design in `DESIGN.md`; binding commitments in `PREREGISTRATION.md`; literature and data record in `research/`; numbers in `results/` (each names the command that made it).

## Run

```
pip install -e .[dev]
pytest                                    # synthetic-data tests always run
git clone --depth 1 https://github.com/languageMIT/naturalstories data/raw/naturalstories
pytest                                    # real-data tests now run too
python scripts/run_naturalstories_baseline.py   # noise ceiling + nuisance baseline (~25 s)
```

## Current numbers

Processing channel, Natural Stories (self-paced reading, 180 subjects, 10 stories, 10,256 words):

- noise ceiling r = 0.861 (95% 0.712–0.885)
- surprisal + spillover + frequency + length + position explain R² = 0.445 out of sample
- approximate residual ceiling for a probe: r ≈ 0.54

No language-model hidden states have been probed yet; that needs a model download (not possible from the build sandbox).

## Modules

`judge/naturalstories.py` loader and alignment · `judge/ceiling.py` split-half ceilings · `judge/baselines.py` nuisance regressors and residualisation · `judge/probe.py` blocked ridge probes with controls and bootstrap · `judge/beliefs.py` Ely–Frankel–Kamenica suspense and surprise.

## Licence

Code: Apache-2.0 (placeholder; founder to confirm). Natural Stories is CC BY-NC-SA 4.0 and is not redistributed here.
