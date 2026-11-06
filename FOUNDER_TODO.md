# FOUNDER_TODO — things only you can do, each prepared so it takes minutes

Updated 2026-11-05. Order matters: 1 and 2 gate everything else.

## 1. Decisions — DONE 6 Nov (Llama-3.1-8B; 70B yes; phase 2 at our pace, Jan latest)

## 2. Send five mentor emails (20 min)

Goal: one person who has published at ACL/EMNLP/COLM/CogSci and will read two drafts and endorse on arXiv (cs.CL). Shortlist (verify names and current affiliations before sending; the research agent's `research/related_work.md` has the papers):

1. First author of "Do Language Models Agree with Human Perceptions of Suspense in Stories?" (arXiv 2508.15794). Closest paper; they named the exact gap we fill.
2. Mariya Toneva (MPI-SWS) — LLM–brain alignment during reading; her group reads this kind of draft.
3. Leila Wehbe (CMU) — Harry Potter fMRI; the dataset owner.
4. An NLP faculty member at NUS or NTU (Min-Yen Kan at NUS is the obvious first ask; local, and NS-aware).
5. David Wilmot / Frank Keller (Edinburgh) — suspense modelling as uncertainty reduction (ACL 2020).

Draft (adapt per person; attach `DESIGN.md` as PDF):

> Subject: Short design doc — do LLM representations predict where a passage lands on readers?
>
> Dear Prof. ___,
>
> I'm a Singaporean student currently in national service, working independently on a study you may find close to your own work: whether a language model's internal representations predict, moment by moment, what a human reader does and feels while reading fiction, using public reading-time, eye-tracking, fMRI and per-segment suspense-rating data as ground truth, with linear probes, noise ceilings, surprisal-controlled residuals and a pre-registered primary test.
>
> I have the Natural Stories processing channel aligned and baselined (noise ceiling r = 0.86, nuisance R² = 0.45) and the probe pipeline tested with controls; model hidden states are next. The one-page design is attached.
>
> I'm not asking for supervision. I'm asking whether you'd be willing to read two drafts over the next six months and, if the work holds up, endorse an arXiv submission in cs.CL. If this is too far from your interests, a one-line pointer to someone closer would be just as valuable.
>
> Thank you for your time,
> Vikram Goel
> [one line of credentials: Syndrome2Vec preprint, Swarm Colosseum; link to GitHub]

## 3. Licence — DONE: MIT

## 4. GitHub — repo created by founder 6 Nov; Claude to push (needs owner/name). Public is fine now: nothing in it is a result yet, and a dated public history helps priority. Your call per hard rule 6.

## 5. Later: GPU and data downloads

The build sandbox cannot reach OpenNeuro, OSF or Hugging Face. Hidden-state extraction and the fMRI/eye-tracking downloads need either the VPS (check its egress) or a Colab session you start. Colab free tier handles 8B models in 8-bit. I'll have a notebook ready that extracts all layers for Natural Stories and writes a single `.npz`; you run one cell.

## Not yet needed

Paying readers (phase 2, January at earliest). arXiv account/endorsement (needs a draft first, ~January).
