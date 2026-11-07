"""Smoke test of the probe script on a fake hidden-state file, so the founder's Colab run
is the first time it meets real activations, not the first time it runs."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from judge import naturalstories as ns

pytestmark = pytest.mark.skipif(not ns.DEFAULT_ROOT.exists(), reason="corpus not downloaded")


def test_probe_script_runs_on_fake_hs(tmp_path):
    t = ns.load()
    w = t.words.sort_values(["item", "zone"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    n, d = len(w), 16
    hs = {"item": w["item"].values, "zone": w["zone"].values,
          "surprisal": w["surprisal"].fillna(3.0).values}
    for l in range(3):
        hs[f"layer_{l}"] = rng.standard_normal((n, d)).astype(np.float16)
    p = tmp_path / "hs.npz"
    np.savez_compressed(p, **hs)
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(repo / "scripts" / "run_probe_naturalstories.py"),
                        "--hs", str(p), "--name", "faketest", "--n_boot", "5"],
                       capture_output=True, text=True, cwd=repo)
    assert r.returncode == 0, r.stderr[-2000:]
    out = repo / "results" / "probe_faketest.json"
    assert out.exists()
    out.unlink(); (repo / "results" / "probe_faketest.md").unlink()


def test_memorisation_run_length():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "run_memorisation.py"
    spec = importlib.util.spec_from_file_location("mem", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    s = m.stats([3, 0.1, 0.2, 0.3, 2, 0.01, 0.02], "token")
    assert s["longest_run_lt_0.5"] == 3 and abs(s["frac_lt_0.05"] - 2 / 7) < 1e-9
