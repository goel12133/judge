"""judge — do LLM representations predict what a reader does and feels while reading?

Modules:
  naturalstories  loader for the Natural Stories self-paced-reading corpus (processing channel)
  ceiling         inter-subject noise ceilings
  baselines       nuisance regressors (surprisal, frequency, length, position) and residualisation
  probe           ridge probes with blocked (by-story) cross-validation and controls
  beliefs         Ely–Frankel–Kamenica suspense and surprise from a belief process
"""
__version__ = "0.0.1"
