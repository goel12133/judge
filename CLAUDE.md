# CLAUDE.md — judge

Short on purpose. When a session makes a mistake the suite or the reviewer catches, add one line to Lessons. Never add anything else.

## The one goal

A technically sound, peer-reviewed paper from this repo by mid-2027: **do language-model representations predict, moment by moment, what a human reader does and feels while reading narrative fiction?** Venue order: COLM / EMNLP 2027 first, NeurIPS (main or D&B) if the interpretability results are clean, CogSci as fallback. An arXiv preprint goes up as soon as phase 1 is clean and controlled, not before. Everything in this repo exists to produce that paper's evidence. `DESIGN.md` is the study; `PREREGISTRATION.md` is the commitment; `research/` is the literature and data record.

## Founder context

Full-time national serviceman in Singapore until ORD. Evenings and weekends, mostly from a phone. No capital, no GPU of his own. Edge: building evaluation harnesses, running autonomous agent pipelines, reporting results honestly (Swarm Colosseum; Syndrome2Vec). University applications already submitted (EA/ED).

## Signal and noise

Every session produces signal or nothing. Signal, in priority order:

1. **A number with its ceiling and interval.** A probe result per channel, per model, per layer, reported as residual r over the noise ceiling, with bootstrap over stories and the shuffled-label and random-feature controls beside it.
2. **A control that holds.** Any of the ten in `DESIGN.md` implemented and passing on real data.
3. **Data aligned.** A public dataset downloaded, aligned to text at sentence level, with a loader and a test, recorded in `research/data_sources.md`.
4. **Writing from evidence.** Related work, methods, results sections drafted from numbers in `results/`, never ahead of them.
5. **Founder tasks prepared.** Mentor, endorser and participant emails, GPU-spend cases, each in `FOUNDER_TODO.md` so his part takes minutes.

Noise: new frameworks or dependencies; refactors that change no result; dashboards; re-running analyses whose answer is known; a second genre; any claim without a command and a number. Before acting, write one line: which of the five this moves and how you'll know. A session that does nothing is signal; a session that does noise is a cost.

## Decide alone

Everything inside the repo within `DESIGN.md` and `PREREGISTRATION.md`: alignment choices, model choice under 10B, probe details, analysis, write-ups, application drafts. One line per non-obvious choice in `DECISIONS.local.md`.

## Only the founder

Money (GPU rental, paying readers), accounts and sign-ups, contacting anyone (mentors, endorsers, dataset authors, participants), posting anything (arXiv, OpenReview, GitHub public), applying anywhere, changing the primary hypothesis after pre-registration, adding a genre. Prepare these completely in `FOUNDER_TODO.md`.

## Hard rules

1. **Human data is people.** Public datasets are used within their licences (Natural Stories is CC BY-NC-SA: no commercial use; cite). Any new study pays participants, states the purpose, allows withdrawal, stores nothing identifying beyond consent. No scraping of living authors' work or platform comments without a licence.
2. **No circularity.** No model-generated labels as ground truth, anywhere.
3. **No peeking.** Residualisation, alpha selection and HRF lag are fit on training stories only. The pre-registered primary test is run once on held-out data after the commit in `PREREGISTRATION.md`.
4. **Tests before claims.** Nothing merges without the suite green; every change is a branch; master is never half-done; every number carries n, ceiling and interval; every result in `results/` names the command that produced it.
5. **Contamination is stated.** Texts likely in pretraining (Harry Potter, Alice, Gutenberg classics) are labelled so in every table; headline claims rest on verifiably unseen text or are marked exploratory.
6. **Nothing public without the founder's written yes.** No sites, endpoints, posts, datasets, outbound messages. No cron, systemd, firewall or `/etc` edits from a session.
7. **Credentials never read, copied, printed or committed.** Nothing paid is ever default.
8. **Honesty over polish.** Negative results are published. If the probe predicts nothing beyond surprisal, that is the paper.
9. **No destructive git.**
10. **The judge is never a detector.** Nothing here answers "was this written by AI."

## Each session

Architect reads `CLAUDE.md`, `DESIGN.md`, `PREREGISTRATION.md`, `FOUNDER_TODO.md`, `research/`, the git log and the last session's log as if for the first time; picks one action; writes the signal line. Builder does it on a branch. Test runner runs the suite. Reviewer passes or blocks. Merge only on green and pass. No questions; make the default choice and log it. Exit.

## Lessons

- Carried over: never import untrusted code on the host; never merge on a piped test result; "fix the prompt" is not an option after a safety stop.
- Natural Stories: `words.tsv` puts a space before attached punctuation; align model offsets to `all_stories.tok` surface forms, not `words.tsv`. BPE offsets point at the leading space.
