# Data sources

Public datasets for: *Do LLM internal representations predict, moment by moment, what a human reader feels/does while reading narrative fiction?*

Status key: **[V]** = official page/paper fetched this session; **[S]** = seen only in search listing; **UNVERIFIED** = not confirmed. "Login?" refers to the download, not to viewing. "Contamination" = is the stimulus text likely in LLM pretraining corpora (CLAUDE.md rule 5: label in every table).

Only URLs fetched or seen in search results are listed.

---

## 1. Brain data during narrative reading / listening

### 1.1 Wehbe et al. 2014 — Harry Potter fMRI (reading) [V]
- **Contents.** 8 subjects; Chapter 9 of *Harry Potter and the Sorcerer's Stone* (~5,176 words) presented word-by-word (RSVP, 0.5 s/word) in 3T fMRI, TR = 2 s, ~45 min, 4 runs. Raw-ish (unsmoothed, undetrended, unmasked) volumes plus word-timing arrays.
- **Genre.** Children's fantasy novel (Rowling 1997).
- **Licence.** Not stated on the page; "available for use" with citation request (Wehbe et al. 2014, PLoS ONE 9(11): e112575). Treat as research-use-only.
- **Download.** http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/ (cited in Toneva & Wehbe 2019). Format: per-subject NumPy/MATLAB arrays (page has a "Description of the format" section; exact file list not visible in fetch — confirm on download). Size: UNVERIFIED (order of a few GB).
- **Login?** No.
- **Text redistributable?** No — copyrighted; the dataset includes word labels/timings; chapter text must be sourced by the user. **Contamination: HIGH** (HP is certainly in pretraining; Oh & Schuler-style memorisation risk).

### 1.2 Wehbe et al. 2014 — Harry Potter MEG [V]
- **Contents.** Same chapter, RSVP 0.5 s/word, Elekta 306-sensor MEG, 1 kHz; 3 subjects in Toneva & Wehbe 2019 (subject count not stated on KiltHub page).
- **Download.** KiltHub record "RSVP reading of book chapter in MEG": https://kilthub.cmu.edu/articles/dataset/RSVP_reading_of_book_chapter_in_MEG/20465898 — only time and label arrays are public; full MEG requires a Google form (https://forms.gle/9pjRk6B7aw79w2xs6) and a data-use agreement. **Founder task** (contacting/sign-up) if we want it.
- **Licence.** Not stated. **Login?** Form + agreement. **Contamination: HIGH.**

### 1.3 Narratives (Nastase et al. 2021) — OpenNeuro ds002345 [V]
- **Contents.** 345 subjects, 891 functional scans, 27 spoken stories (3–56 min, ~4.6 h audio), BIDS, with age/gender/condition/comprehension metadata; audio stimuli included. Collected 2011–2018 (Hasson lab). Story names not listed on fetched pages (only "Pie Man" seen); per-story lists are in the Scientific Data paper (PMC8479122 — blocked by captcha this session) and the OpenNeuro README.
- **Genre.** Mostly spoken personal narratives / live storytelling (The Moth-style), some radio fiction; listening, not reading.
- **Licence.** **CC0** (public domain) for data; audio stimuli "for non-commercial scholarly research, principally feature extraction, under fair use."
- **Download.** OpenNeuro https://openneuro.org/datasets/ds002345/ (doi 10.18112/openneuro.ds002345.v1.1.4); DataLad `datalad install ///labs/hasson/narratives`; AWS S3 `s3://fcp-indi/data/Projects/narratives/` (anonymous, HTTP only); scripts https://github.com/snastase/narratives/ (GPL-3). Format: BIDS NIfTI + derivatives (fMRIPrep). Size: >350k volumes; "6.4 days of data" — expect ~1 TB raw; subsets per story are far smaller.
- **Login?** No (OpenNeuro/S3 anonymous).
- **Text redistributable?** Transcripts/alignments: existence of Gentle word-level alignments in the release is documented in the paper (per memory of the descriptor) but was not confirmed from fetched pages — **verify in ds002345 `stimuli/` on download.** **Contamination: LOW–MEDIUM** (spoken, mostly unpublished stories; some, e.g. Pie Man, have transcripts online).

### 1.4 Alice in Wonderland fMRI — OpenNeuro ds002322 (Bhattasali, Brennan, Luh, Franzluebbers & Hale, LREC 2020) [V]
- **Contents.** 29 subjects (v1.0.3) listening to Chapter 1 of *Alice's Adventures in Wonderland* (~12.4 min audiobook, ~2,100 words); raw + preprocessed fMRI, code and stimuli.
- **Download.** https://openneuro.org/datasets/ds002322/versions/1.0.3 ; lab page https://sites.lsa.umich.edu/cnllab/?p=735 ; format BIDS. Size UNVERIFIED (small; tens of GB).
- **Licence.** Not stated on lab page; OpenNeuro datasets are CC0 by platform policy (UNVERIFIED for this accession — check dataset_description.json).
- **Login?** No. **Text redistributable?** Yes — public domain (Carroll 1865). **Contamination: HIGH** (Gutenberg classic). Used by arXiv 2606.07707.

### 1.5 Alice in Wonderland EEG (Brennan & Hale 2019, PLoS ONE) [V]
- **Contents.** 49 participants (42 usable, 33 in some releases), 61-channel EEG at 500 Hz, same 12.4-min Chapter 1 audio in 12 segments; word-by-word CSV with n-gram/RNN/CFG surprisal (2,130 rows); comprehension quiz. Format: MATLAB/FieldTrip `.mat` (v1), BrainVision (2023 update); MNE `.fif` re-release via Brodbeck et al. at UMD DRUM (doi 10.13016/pulf-lndn).
- **Download.** Deep Blue https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t (doi 10.7302/Z29C6VNH); 56 files, **4.04 GB**. Mirror documentation: https://deepstrf.readthedocs.io/en/latest/_source/md/README_Alice_EEG.html
- **Licence.** Licence field blank on Deep Blue — UNVERIFIED; treat as research use with citation. **Login?** No. **Contamination: HIGH.**

### 1.6 Le Petit Prince multilingual fMRI — OpenNeuro ds003643 (Li et al. 2022, Scientific Data 9: 530) [V paper; accession from task spec, not seen on page]
- **Contents.** 112 subjects: 49 English, 35 Chinese, 28 French, each listening to the full audiobook of *The Little Prince* in their native language (~94–99 min, 9 runs of ~10 min; multi-echo BOLD). Rich annotations: word-boundary TextGrids, prosody, lemmas/frequency/POS, GloVe & BERT embeddings, constituency/dependency parses, coreference/NER (EN/CN), 36 quiz questions.
- **Download.** OpenNeuro (accession ds003643 per task brief; paper text did not print the accession — confirm). Paper: https://www.nature.com/articles/s41597-022-01625-7 ; lab page https://sites.lsa.umich.edu/cnllab/?p=1182. Format BIDS. Size UNVERIFIED (large; multi-echo).
- **Licence.** Not stated in paper text; OpenNeuro default CC0 (UNVERIFIED). **Login?** No.
- **Genre.** Children's novella (Saint-Exupéry 1943). **Text redistributable?** English translation (Wilkinson) is copyrighted; French original is public domain in some jurisdictions only. **Contamination: HIGH** (well-known, many translations online).

### 1.7 LeBel et al. 2023 — Moth stories fMRI, OpenNeuro ds003020 [V preprint; accession from task spec]
- **Contents.** 8 subjects (UTS01–08), ~370 min (6.2 h) of listening each: 26 unique 10–15-min stories from *The Moth* plus a 10-min test story (*wheretheressmoke*) repeated in each of 5 sessions. NIfTI BOLD, 44.1 kHz WAV stimuli, preprocessed HDF5, hand-corrected cortical surfaces, hand-corrected **TextGrids with word- and phoneme-level timings and full transcripts**.
- **Download.** OpenNeuro ds003020 (accession per brief; preprint https://www.biorxiv.org/content/10.1101/2022.09.22.509104.full.pdf says "OpenNeuro" without number — confirm); code/features https://github.com/HuthLab/deep-fMRI-dataset. Paper: *Scientific Data* 10: 555 (2023), https://www.nature.com/articles/s41597-023-02437-z (fetch failed; PMC10447563 in listing). Size UNVERIFIED (hundreds of GB raw).
- **Licence.** Not stated in preprint; OpenNeuro default CC0 (UNVERIFIED). **Login?** No.
- **Genre.** First-person true-life spoken stories. **Text redistributable?** Transcripts shipped in the dataset; Moth audio is copyrighted but distributed for research. **Contamination: LOW–MEDIUM** (Moth episodes exist online as audio; transcripts less likely in text corpora). Used by Zhang et al. CCN 2025 (*How to Draw*).

---

## 2. Reading behaviour (eye-tracking / self-paced reading / EEG)

### 2.1 Natural Stories (Futrell et al. 2018/2021) [V — already downloaded locally]
- **Contents.** 10 English stories (~10k words) edited to include rare syntactic constructions; self-paced reading times from MTurk readers (`naturalstories_RTS`), parses, frequencies, audio recordings, `words.tsv` tokenisation. May 2025 commit fixed a one-position SPR misalignment — **check which version is on disk.**
- **Licence.** **CC BY-NC-SA 4.0** confirmed from https://github.com/languageMIT/naturalstories/blob/master/LICENSE.txt. No commercial use; cite; share-alike.
- **Download.** https://github.com/languageMIT/naturalstories (git clone). Format: TSV/XLSX. Size: tens of MB. **Login?** No.
- **Genre.** Adapted folk/fairy tales and anecdotes ("Boar", "Aqua", "MatchstickSeller" etc. — names from memory, UNVERIFIED). **Text redistributable?** Yes under CC BY-NC-SA. **Contamination: MEDIUM** — repo is on GitHub since 2017 and the base stories are classics, but the edited versions are specific; still plausibly in Common Crawl/GitHub-derived pretraining.

### 2.2 Provo (Luke & Christianson 2018) [V]
- **Contents.** 84 readers, EyeLink 1000 Plus @1000 Hz, 55 passages (2,689 tokens) from online news, popular-science magazines and public-domain fiction; cloze predictability norms from 470 participants (orthographic, morphosyntactic, semantic).
- **Download.** https://osf.io/sjefs (two CSVs: predictability norms, eye-tracking data); also via `pymovements` (`pm.Dataset('Provo')`). Size: tens of MB. **Login?** No.
- **Licence.** Not stated in paper; OSF project licence UNVERIFIED (check project page). **Genre.** Mixed; fiction subset small. **Text redistributable?** Passages ship in the CSVs. **Contamination: HIGH** (online news/public-domain sources).

### 2.3 ZuCo 1.0 (Hollenstein et al. 2018, Sci Data 5: 180291) [V]
- **Contents.** 12 subjects; 1,107 sentences (400 SST movie-review sentences with sentiment labels; ~700 Wikipedia sentences about people), 21,629 words, 154,173 fixations; 128-ch EEG + EyeLink 1000 Plus, both 500 Hz; 3 tasks (sentiment reading, normal Wikipedia reading, relation-search reading).
- **Download.** https://doi.org/10.17605/OSF.IO/Q3ZWS . Format: MATLAB `.mat` per subject/task (plus preprocessed). Size: ~tens of GB raw (UNVERIFIED). **Login?** No. **Licence: CC BY 4.0.**
- **Genre.** Non-narrative isolated sentences. **Contamination: HIGH** (SST, Wikipedia).

### 2.4 ZuCo 2.0 (Hollenstein et al. 2020, LREC) [V]
- **Contents.** 18 subjects; 739 Wikipedia sentences (349 normal reading, 390 task-specific relation annotation); simultaneous EEG + eye-tracking.
- **Download.** https://osf.io/2urht/ . Licence not stated on ACL page (OSF project page to confirm). **Login?** No. **Genre.** Non-narrative. **Contamination: HIGH.**

### 2.5 MECO — Multilingual Eye-movement Corpus (Siegelman, Schroeder, Kuperman et al.; BRM 2022 wave 1, Sci Data 2025 wave 2) [V case-study page]
- **Contents.** Wave 2: 654 participants, 13 languages, 16 labs; 12 encyclopedic Wikipedia-style passages (5 translated, 7 topic-matched). Wave 1 (BRM 2022): 13 languages, similar design; also an L2-English component.
- **Download.** Project site https://meco-read.com/ (OSF links from there — not fetched); wave-2 paper doi 10.1038/s41597-025-05453-3. Licence UNVERIFIED. **Login?** No (OSF).
- **Genre.** Expository, non-narrative. **Contamination: HIGH** (Wikipedia-style). Low priority for this paper.

### 2.6 GECO — Ghent Eye-tracking Corpus (Cop, Dirix, Drieghe & Duyck 2017, BRM 49: 602–615) [V]
- **Contents.** A **complete novel**: Agatha Christie, *The Mysterious Affair at Styles* (1920; 54,364 English words, 5,031 sentences; Dutch *De zaak Styles* 59,716 words). 14 English monolinguals + 19 Dutch–English bilinguals (bilinguals read half in each language). Word-level measures: first fixation, single fixation, gaze duration, total reading time, go-past, skipping; ~4 h of reading per participant.
- **Download.** http://expsy.ugent.be/downloads/geco (stated "freely available"; page not fetched this session — URL from the paper PDF). Format: Excel/CSV per participant. Size: ~hundreds of MB. **Login?** UNVERIFIED (likely none).
- **Licence.** Not named; citation of paper + corpus required.
- **Genre.** Detective fiction — the **only public eye-tracking corpus of a whole novel**, hence the best behavioural dataset for narrative suspense. **Text redistributable?** Public domain (Gutenberg). **Contamination: HIGH** (Gutenberg #863).

### 2.7 Dundee corpus (Kennedy & Pynte 2005) [S]
- **Contents.** 10 English readers, 20 *The Independent* editorials (~51k words); 10 French readers, 20 *Le Monde* articles. Eye-tracking.
- **Availability.** Not openly downloadable; historically obtained by request from Alan Kennedy (https://dundee.ac.uk/people/alan-kennedy) — UNVERIFIED current status; licence restrictive. **Founder task** if needed.
- **Genre.** Newspaper editorials, non-narrative. **Contamination: HIGH.** Low priority.

---

## 3. Human affect / suspense ratings on narrative text

### 3.1 Lehne & Koelsch 2015 — *The Sandman* per-segment suspense ratings [V — PUBLIC]
- **Contents.** 23 participants × 65 segments, 10-point suspense ratings (S2 Dataset, XLS); participant data (S1); per-segment control measures — action, imageability, arousal, valence, sentence length (S3); **the segmented German text** (S1 Text, PDF). fMRI not public.
- **Download (CC BY 4.0, PLoS ONE):**
  - S1 Dataset: https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371%2Fjournal.pone.0124550.s001
  - S2 Dataset (ratings): https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371%2Fjournal.pone.0124550.s002
  - S3 Dataset (controls): https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371%2Fjournal.pone.0124550.s003
  - S1 Text (segmented text): https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371%2Fjournal.pone.0124550.s008
- **Login?** No. Size: < 5 MB. **Text redistributable?** Hoffmann (1816) is public domain; the shortened version is in the CC BY supplement. **Contamination: HIGH for the full German text (Gutenberg-DE); MEDIUM for the shortened 6,859-word version.** Matlin et al. 2025 used an English translation — if we do the same, our translation step is a confound to pre-register. Best-in-class target: real per-reader, per-segment suspense on literary fiction.

### 3.2 Delatorre et al. 2018 — 12-passage story [V paper; data partial]
- **Contents.** 68 participants, 12 passages, 9-point SAM suspense after each passage, 2×2×2 conditions. Supplementary material contains the **story text(s)** (journalistic and novel style, good/bad ending). Raw per-participant ratings: **not found** — no data-availability statement or OSF link on the article page. Aggregate per-segment means are in the paper's figures/tables and were used by Matlin et al. 2025 (their repo https://github.com/eilab-gt/SuspensePerception has `configs/` for gerrig/lehne; whether it bundles rating tables is UNVERIFIED — clone and inspect `research/`, `experiments_archive/`).
- **Licence.** Frontiers CC BY 4.0 (article and supplement). **Genre.** Purpose-written realist drama. **Contamination: LOW** (unpublished story in a 2018 supplement PDF). **Founder task:** email Delatorre/Gervás for raw data if needed.

### 3.3 Gerrig & Bernardo 1994; Brewer & Ohtsuka 1988 [S]
- Aggregate ratings only, in the original papers (Poetics; Brewer's CSR tech report https://www.ideals.illinois.edu/items/17726). Few data points (per-excerpt / 5 per story). Bond excerpts copyrighted; Brewer stories (Birthday, Flying, Lottery, Old Phoebe, Ylla) include published fiction (e.g. Bradbury's *Ylla*, Jackson's *The Lottery*) — **Contamination: HIGH**, text not redistributable. Use only as replication sanity checks.

### 3.4 Wilmot & Keller 2020 — WritingPrompts sentence-level suspense annotations [V paper; release UNVERIFIED]
- **Contents.** 200 stories (100 dev, 100 test; 25–75 sentences) from WritingPrompts (Fan et al. 2018, Reddit r/WritingPrompts), 5 MTurk "master" raters each, per-sentence relative change in suspense on 5 levels, converted to cumulative z-scored curves; α 0.52/0.57.
- **Download.** Code: https://github.com/dwlmt/Story-Untangling/tree/acl-2020-dec-submission — annotation CSVs not confirmed to be in the repo. **Action:** clone and grep; otherwise **founder task** to email D. Wilmot / F. Keller.
- **Licence.** UNVERIFIED. WritingPrompts text is Reddit user content (scraped; no explicit licence) — CLAUDE.md rule 1 caution on living authors' work. **Contamination: HIGH** (WritingPrompts is a standard pretraining/fine-tuning corpus).

### 3.5 EmoBank (Buechel & Hahn 2017) [V]
- **Contents.** ~10k sentences (MASC genres incl. **fiction**, letters, news, blogs; SemEval-2007 headlines), VAD ratings from **writer** and **reader** perspective; per-annotator ratings (2022); subset with Ekman labels; stratified split.
- **Download.** https://github.com/JULIELab/EmoBank (CSV). Size < 5 MB. **Licence: CC BY-SA 4.0.** **Login?** No. **Contamination: HIGH** (MASC public). Non-narrative-continuous; use for probe training only.

### 3.6 GoEmotions (Demszky et al. 2020) [V]
- 58k Reddit comments, 27 emotions + neutral, 3 raters; Apache-2.0; https://huggingface.co/datasets/google-research-datasets/go_emotions . Non-narrative; probe-training only. **Contamination: HIGH.**

### 3.7 Story-level sentence emotion corpora
- **Alm (2008) children's stories + Christ et al. 2024 re-annotation** [V]: 169 Grimm/Andersen/Potter stories, 14,884 sentences, discrete feeler/mood labels mapped to V/A via NRC-VAD; https://github.com/lc0197/emotional_trajectories_stories . Licence UNVERIFIED (check repo). Labels are *text* emotion by annotators, not reader self-report. **Contamination: HIGH** (Gutenberg).
- **childTale-A (Lüdtke & Herrmann 2023)** [V]: 80 Grimm tales (German), 2 annotators, sentence-level valence/arousal + 6 basic emotions, valence arcs; Zenodo https://zenodo.org/record/7737329 , **CC BY 4.0**, 2.1 MB. German; **Contamination: HIGH.**
- **DENS (Liu, Osama & de Andrade 2019)** [V arXiv only]: Gutenberg + Wattpad passages with MTurk multi-class emotion labels; no download link found — UNVERIFIED availability; Wattpad text raises living-author/licence issues.
- **Mori et al. 2019 shared-character stories** [V paper]: 100 five-sentence stories with sentence-level V/A from character and **reader** viewpoints (33 raters); release "in the future" — UNVERIFIED/likely unavailable.
- **Not found:** any public dataset with *continuous* (slider) reader affect on written literary fiction at sentence resolution. Zhang et al. CCN 2025 collected slider surprise ratings (30 participants × 2 spoken stories) — not released as far as fetched pages show; **founder task:** email Rosenberg lab.

### 3.8 Lexicons (baselines)
- NRC EmoLex, NRC-VAD, NRC-EIL: https://saifmohammad.com/WebPages/lexicons.html ; terms https://saifmohammad.com/WebPages/AccessResource.htm — free for **non-commercial research**, no redistribution, commercial licence via NRC shop. Download is a zip from the site (no login).
- labMT / hedonometer (Reagan et al. 2016 appendices): https://www.uvm.edu/storylab/share/papers/reagan2016b/ .

---

## 4. Contamination summary

| Dataset | Stimulus text | Likely in pretraining? |
|---|---|---|
| HP fMRI / MEG | Harry Potter ch. 9 | **Yes** (verbatim memorisation documented in the literature) |
| Alice fMRI / EEG | Alice ch. 1 | **Yes** (Gutenberg) |
| Le Petit Prince | full novella (EN/CN/FR) | **Yes** |
| GECO | Christie, *Styles* | **Yes** (Gutenberg) |
| Provo, ZuCo, MECO, Dundee | news/Wikipedia/SST | **Yes** |
| Natural Stories | edited folk tales | Probably (GitHub since 2017) |
| Sandman (Lehne) | shortened Hoffmann | Full text yes; shortened edition partially |
| Wilmot WritingPrompts | Reddit stories | **Yes** |
| Narratives (ds002345) | spoken, mostly unpublished | Mostly **no** (audio only; some transcripts online) |
| LeBel Moth (ds003020) | Moth stories | Audio public; transcripts unlikely in text corpora → **low** |
| Delatorre 2018 | purpose-written story (PDF supplement) | **Low** |
| childTale-A / Alm | Grimm/Andersen | **Yes** |

**Verifiably-unseen text for headline claims:** none of the above is guaranteed unseen. Options: (a) Delatorre story + any new story we commission (founder: paid readers); (b) Narratives/Moth transcripts as lowest-risk existing text; (c) check memorisation directly (prefix-completion test) and report per text.

---

## 5. Open items / founder tasks surfaced here
1. Clone `dwlmt/Story-Untangling` and `HCAI-Lab-GT/SuspensePerception`; inspect for bundled human rating tables (no approval needed beyond network).
2. Confirm OpenNeuro accessions ds003643 and ds003020 and their `dataset_description.json` licence fields on download.
3. Founder: request HP MEG (CMU form), Delatorre raw ratings, Wilmot annotations if absent, Rosenberg-lab slider data, Dundee (if ever needed).
4. Verify GECO download page (http://expsy.ugent.be/downloads/geco) is live and whether registration is required.
