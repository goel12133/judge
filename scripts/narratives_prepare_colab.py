"""Prepare Narratives (OpenNeuro ds002345, CC0) for the engagement channel — run on Colab.

    python scripts/narratives_prepare_colab.py --out narr --stories pieman bronx black forgot \
        notthefallintact merlin sherlock tunnel --max_subjects 30

For each story: download each subject's cleaned MNI BOLD (afni-nosmooth, desc-clean; falls
back to fMRIPrep desc-preproc with confound regression), reduce to Schaefer-400 parcel
means, z-score per parcel within subject, and save

    narr_{story}.npz : bold (n_subj, n_TR, 400) float16; subjects; TR; onset_s; words table
                       (word, start_s, end_s from the Gentle alignment, in scan time); text

plus narr_meta.json. Nothing voxel-level leaves Colab; the per-story file is a few MB.
Atlas: Schaefer 2018 400 parcels / 7 networks, 2 mm, MNI152NLin6Asym (nilearn). The BOLD
is in MNI152NLin2009cAsym: the two MNI templates differ by ~1 mm, which parcel averaging
over 2 mm voxels tolerates (state in the paper; resample if a reviewer asks).
Raw data stay on Colab; derived parcel series are CC0 like the source.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

S3 = "https://s3.amazonaws.com/fcp-indi/data/Projects/narratives"   # OpenNeuro bucket holds raw BIDS only; derivatives + Gentle alignments are on the fcp-indi mirror


def http_ok(url: str) -> bool:
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(dest), url])
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="narr")
    ap.add_argument("--stories", nargs="+", default=["pieman", "bronx", "black", "forgot",
                                                     "notthefallintact", "merlin", "sherlock", "tunnel"])
    ap.add_argument("--max_subjects", type=int, default=30)
    ap.add_argument("--n_parcels", type=int, default=400)
    args = ap.parse_args()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nilearn", "nibabel"], check=True)
    import numpy as np
    import pandas as pd
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker

    out = Path(args.out); out.mkdir(exist_ok=True)
    raw = out / "raw"
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=args.n_parcels, yeo_networks=7, resolution_mm=2)
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]

    # metadata from the Narratives code repo (task -> subjects, event onsets)
    code = out / "narratives_code"
    if not code.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/snastase/narratives", str(code)], check=True)
    task_meta = json.load(open(code / "code" / "task_meta.json"))
    event_meta = json.load(open(code / "code" / "event_meta.json"))
    scan_exclude = json.load(open(code / "code" / "scan_exclude.json"))

    meta = {}
    for story in args.stories:
        subjects = sorted(task_meta[story])
        excl = set(scan_exclude.get(story, {}).keys())
        subjects = [s for s in subjects if s not in excl][: args.max_subjects]
        onset = list(event_meta[story].values())[0]["onset"]
        # transcript / alignment
        gentle_dir = raw / "stimuli" / "gentle" / story
        for fn in ("align.csv", "align.json", "transcript.txt"):
            fetch(f"{S3}/stimuli/gentle/{story}/{fn}", gentle_dir / fn)
        text = (gentle_dir / "transcript.txt").read_text() if (gentle_dir / "transcript.txt").exists() else ""
        words = None
        if (gentle_dir / "align.csv").exists():
            words = pd.read_csv(gentle_dir / "align.csv", header=None, names=["word", "aligned", "start", "end"])
        elif (gentle_dir / "align.json").exists():
            g = json.load(open(gentle_dir / "align.json"))
            words = pd.DataFrame([{"word": w["word"], "aligned": w.get("alignedWord"), "start": w.get("start"), "end": w.get("end")}
                                  for w in g["words"]])
        if words is None:
            print(f"[{story}] no Gentle alignment at {S3}/stimuli/gentle/{story}/ — check layout"); continue
        words = words.dropna(subset=["start"]).reset_index(drop=True)
        words["start_s"] = words["start"].astype(float) + onset
        words["end_s"] = words["end"].astype(float) + onset

        series, kept, TR = [], [], None
        masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize="zscore_sample", detrend=False, t_r=None)
        for sub in subjects:
            base = f"{sub}_task-{story}"
            cands = [
                (f"{S3}/derivatives/afni-nosmooth/{sub}/func/{base}_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz", "clean"),
                (f"{S3}/derivatives/afni-nosmooth/{sub}/func/{base}_run-1_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz", "clean"),
                (f"{S3}/derivatives/fmriprep/{sub}/func/{base}_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz", "preproc"),
                (f"{S3}/derivatives/fmriprep/{sub}/func/{base}_run-1_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz", "preproc"),
            ]
            got = None
            for url, kind in cands:
                dest = raw / sub / Path(url).name
                if fetch(url, dest):
                    got = (dest, kind, url); break
            if not got:
                print(f"[{story}] {sub}: no BOLD found"); continue
            dest, kind, url = got
            import nibabel as nib
            img = nib.load(str(dest))
            TR = float(img.header.get_zooms()[3])
            confounds = None
            if kind == "preproc":   # minimal cleaning: motion + CSF/WM + cosine drifts from fMRIPrep confounds
                cf_url = url.replace("_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz", "_desc-confounds_regressors.tsv")
                cf = raw / sub / Path(cf_url).name
                if fetch(cf_url, cf):
                    df = pd.read_csv(cf, sep="\t")
                    cols = [c for c in df.columns if c.startswith(("trans_", "rot_", "cosine")) or c in ("csf", "white_matter")]
                    confounds = df[cols].fillna(0).values
            ts = masker.fit_transform(img, confounds=confounds)        # (n_TR, n_parcels)
            series.append(ts.astype(np.float16)); kept.append(sub)
            dest.unlink(missing_ok=True)                              # keep Colab disk small
            print(f"[{story}] {sub}: {kind} {ts.shape} TR={TR}", flush=True)
        if not series:
            continue
        n_tr = min(s.shape[0] for s in series)
        bold = np.stack([s[:n_tr] for s in series])
        np.savez_compressed(out / f"narr_{story}.npz", bold=bold, subjects=np.array(kept), TR=TR, onset_s=onset,
                            word=words["word"].astype(str).values, start_s=words["start_s"].values,
                            end_s=words["end_s"].values, text=text, labels=np.array(labels))
        meta[story] = {"n_subjects": len(kept), "n_TR": int(n_tr), "TR": TR, "onset_s": onset, "n_words": int(len(words))}
        print(f"[{story}] wrote narr_{story}.npz {bold.shape}", flush=True)
    json.dump(meta, open(out / "narr_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
