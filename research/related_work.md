# Related work

Paper: *Do LLM internal representations predict, moment by moment, what a human reader feels/does while reading narrative fiction?*

Verification key: **[V]** = landing page / PDF fetched directly in this session; **[S]** = existence confirmed from search-result listing only (bibliographic details from memory, not re-read); **UNVERIFIED** = could not confirm.

---

## A. Theory of suspense and surprise

### Ely, Frankel & Kamenica (2015). "Suspense and Surprise." *Journal of Political Economy* 123(1): 215–260. [V]
- PDF fetched: https://www.wiwi.uni-bonn.de/kraehmer/Lehre/TopicsSS14/ElyEtAl-SurpriseSuspense.pdf ; RePEc record https://econpapers.repec.org/RePEc:ucp:jpolec:doi:10.1086/677350 (doi 10.1086/677350).
- **Summary.** An information-design model in which an agent holds beliefs μ_t over a state ω and derives utility from the *dynamics* of those beliefs rather than from the state itself. Suspense is utility from anticipated belief change (variance of next period's belief); surprise is utility from realised belief change. The authors characterise the information policies (belief martingales) that maximise each and apply them to sports, mysteries, and political primaries; suspense-optimal policies keep per-period suspense constant with increasingly large, increasingly unlikely "plot twists," while surprise-optimal policies require commitment and small per-period belief moves.
- **Exact formulas** (μ_t^ω = belief in state ω at period t; μ̃_{t+1} = random next-period belief; u increasing, strictly concave, u(0)=0):
  - Suspense utility: U_susp = Σ_{t=0}^{T−1} u( E_t[ Σ_ω (μ̃_{t+1}^ω − μ_t^ω)² ] )
  - Surprise utility: U_surp = Σ_{t=1}^{T} u( Σ_ω (μ_t^ω − μ_{t−1}^ω)² )
  - Baseline case u(x)=√x: **suspense_t = standard deviation of next-period beliefs**; **surprise_t = Euclidean distance ‖μ_t − μ_{t−1}‖**.
  - Expectation variant: if the agent cares only about E_μ[ω] for bounded ω, replace the belief vector by its expectation inside the squared differences.
- **Relation.** This is the formal backbone: we operationalise μ_t with an LLM (either the next-token/next-sentence distribution or, following Wilmot & Keller, the hidden-state representation) and test whether Ely-style suspense/surprise computed from LLM states tracks human ratings and reading behaviour. Gap: Ely et al. give no procedure for estimating beliefs from text.

### Gerrig & Bernardo (1994). "Readers as problem-solvers in the experience of suspense." *Poetics* 22(6): 459–472. [S]
- Record: https://researchconnect.stonybrook.edu/en/publications/readers-as-problem-solvers-in-the-experience-of-suspense/
- **Summary.** Proposes that readers experience suspense in proportion to how few plausible escape routes they can generate for a protagonist. Manipulated James Bond excerpts (adding/removing an escape option) and found suspense ratings (1–7) fall when readers can think of more solutions. Replicated across Yale and University of the Philippines samples.
- **Relation.** Supplies a small human-rated stimulus set (reused by Matlin et al. 2025) and a mechanistic hypothesis — suspense ≈ constrained solution space — that maps onto low entropy over *bad* outcomes rather than mere uncertainty. Gap: handful of excerpts, aggregate ratings only.

### Brewer & Ohtsuka (1988). "Story structure, characterization, just world organization, and reader affect in American and Hungarian short stories." *Poetics* 17: 395–415. [S]
- Preprint/CSR technical report: https://www.ideals.illinois.edu/items/17726 (UIUC Center for the Study of Reading).
- **Summary.** Within Brewer & Lichtenstein's structural-affect theory, readers rated suspense (and surprise, curiosity, liking) at five plot points in American and Hungarian short stories. Suspense rose with structural features (initiating event, delayed outcome) and ratings were comparable across cultures; story liking tracked affect.
- **Relation.** Early evidence that per-segment suspense ratings are reliable and cross-cultural; stories (Birthday, Flying, Lottery, Old Phoebe, Ylla) reused by Matlin et al. 2025. Gap: only five rating points per story.

### Lehne, Engel, Rohrmeier, Menninghaus, Jacobs & Koelsch (2015). "Reading a suspenseful literary text activates brain areas related to social cognition and predictive inference." *PLoS ONE* 10(5): e0124550. [V]
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0124550 ; PMC4422438.
- **Summary.** 23 German readers read a shortened (6,859-word) version of E.T.A. Hoffmann's *Der Sandmann* in the scanner, split into 65 segments (~105 words each); after each segment they rated suspense on a 10-point scale. Suspense ratings parametrically modulated activity in medial frontal cortex, IFG, premotor and posterior temporal/temporo-parietal regions, i.e. social-cognition and predictive-inference networks, not amygdala.
- **Relation.** The only fMRI-plus-continuous-suspense-rating literary dataset; per-participant per-segment ratings, control variables and segmented text are published as supporting files (S1–S3 Dataset, S1 Text; CC BY 4.0). fMRI data not public. We can regress LLM-derived suspense on the 65 segment ratings directly.

### Delatorre, León, Salguero, Palomo-Duarte & Gervás (2018). "Confronting a Paradox: A New Perspective of the Impact of Uncertainty in Suspense." *Frontiers in Psychology* 9: 1392. [V]
- https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01392
- **Summary.** 68 students read a purpose-written 12-passage story (child awaiting a liver transplant; passages mapped to Freytag's pyramid) twice, rating suspense after each passage on a 9-point SAM scale; conditions crossed ending-known vs chronological, good vs bad ending, journalistic vs novelistic style. Suspense survived knowing the ending (the "paradox of suspense"), supporting accounts where emotional involvement, not outcome uncertainty alone, drives suspense.
- **Relation.** Gives a clean, ordered 12-point human suspense trajectory and a test case where an uncertainty-only (Ely) model should *fail* if told the ending; a good discriminating stimulus. Raw per-participant data not located (only story text in supplement) — see data_sources.md.

---

## B. Computational models of suspense/surprise in text

### Wilmot & Keller (2020). "Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation." *ACL 2020*, pp. 1763–1788. arXiv:2004.14905. [V]
- https://arxiv.org/abs/2004.14905 ; https://aclanthology.org/2020.acl-main.161 ; code https://github.com/dwlmt/Story-Untangling/tree/acl-2020-dec-submission
- **Summary.** Trains a hierarchical sentence-level LM (GPT-2/transformer-XL-style) on WritingPrompts and defines four measures: Hale-surprise −log P(e_t), Ely-surprise (e_t − e_{t−1})², Hale uncertainty reduction H_{t−1} − H_t, and Ely uncertainty reduction Σ_i P(e^i_{t+1})(e_t − e^i_{t+1})² over sampled next-sentence embeddings. Collected sentence-by-sentence relative suspense judgements (5 MTurk raters per story, 200 WritingPrompts stories of 25–75 sentences, Krippendorff α ≈ 0.52–0.57), converted to absolute curves by cumulative sum. Ely-style uncertainty reduction over *representations* correlated best with humans (near inter-annotator level) and also located turning points in movie synopses.
- **Relation.** Closest prior work: it already instantiates Ely with neural representations and has human sentence-level labels. Gaps we fill: (i) uses a bespoke small model, not frozen modern LLM hidden states; (ii) only verbal ratings, no physiological/reading-time or brain data; (iii) annotation data release not confirmed (code repo only).

### Wilmot & Keller (2021). "Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories." *EMNLP 2021* (venue from memory). arXiv:2109.03754. [V arXiv]
- https://arxiv.org/abs/2109.03754 (also related: Wilmot & Keller 2021, "A Temporal Variational Model for Story Generation," arXiv:2109.06807; and Wilmot's 2022 PhD thesis "Great Expectations: Unsupervised Inference of Suspense, Surprise and Salience in Storytelling," arXiv:2206.09708.)
- **Summary.** Extends the 2020 framework to whole novels (e.g. Shmoop-summarised classics) by adding external memory and knowledge (plot summaries, Wikipedia) to the LM so that Barthes-cardinal-function salience and surprise can be inferred unsupervised over long contexts. Reports that surprise-based salience, improved by memory, aligns with human-annotated salient events.
- **Relation.** Shows the uncertainty-reduction idea scales to novel length; our frozen-LLM approach side-steps the memory engineering by using long-context models. Gap: no moment-by-moment reader affect ground truth.

### Matlin, Zhang, Loza, Popescu, Isbell, Chakraborty & Riedl (2025). "Do Language Models Agree with Human Perceptions of Suspense in Stories?" arXiv:2508.15794. [V]
- https://arxiv.org/abs/2508.15794 ; code https://github.com/eilab-gt/SuspensePerception (redirects to HCAI-Lab-GT/SuspensePerception)
- **Summary.** Replicates four human suspense studies (Gerrig & Bernardo 1994; Brewer & Ohtsuka 1988; Delatorre et al. 2018; Lehne & Koelsch 2015) by prompting ~10 open LMs (Llama 2/3, Gemma 2, DeepSeek-V3, Qwen2, Mistral/Mixtral, WizardLM) plus closed models for suspense ratings. LMs can tell suspenseful from non-suspenseful text and get the coarse rise/fall right, but mis-estimate magnitude, miss inflection points, and are insensitive to adversarial permutations (typos, ciphers, word swaps) that would disrupt human suspense.
- **Relation.** Establishes that *prompted* LM judgements only weakly track human suspense dynamics — motivating our hypothesis that *internal* states (probabilities/hidden representations) may track them better than verbal reports. Also enumerates exactly which legacy human datasets are usable at aggregate level.

### Zhang, Park, Summe, Gollapudi, Leong & Rosenberg (2025). "Surprising narrative events elicit convergent responses in humans and large language models." *CCN 2025* abstract. [V]
- https://2025.ccneuro.org/abstract_pdf/Zhang_2025_Surprising_narrative_events_elicit_convergent_responses.pdf
- **Summary.** Defines LLM surprise as cosine dissimilarity between Llama-3.1-8B layer-20 embeddings of *predicted* vs *actual* upcoming text (N=1–30 words), on two spoken stories (*Paranoia*, Finn et al. 2018; *How to Draw*, LeBel et al. 2023) segmented into events. LLM surprise at N≈20 correlated with continuous slider self-reports of surprise (r = 0.38, 0.69) and with an fMRI edge-fluctuation surprise model.
- **Relation.** The most direct precedent for "LLM internal representation vs. moment-by-moment reader affect," but event-level (36–61 events), surprise only, and conference-abstract scale. We extend to suspense/valence/arousal, finer temporal resolution, multiple readout types (hidden states, probabilities), and behaviour (reading times).

---

## C. LLM representations and the reading brain

### Wehbe, Murphy, Talukdar, Fyshe, Ramdas & Mitchell (2014). "Simultaneously Uncovering the Patterns of Brain Regions Involved in Different Story Reading Subprocesses." *PLoS ONE* 9(11): e112575. [V via PMC listing + Toneva & Wehbe description]
- PMC4245107; data: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/
- **Summary.** 8 (9 in paper, 8 usable) participants read Chapter 9 of *Harry Potter and the Sorcerer's Stone* word-by-word (0.5 s/word, TR = 2 s, ~45 min) in fMRI. A single encoding model with hundreds of hand-built features (semantics, syntax, characters, motion, emotion, visual word form) simultaneously mapped distinct story-reading subprocesses to distinct cortical regions.
- **Relation.** Provides the canonical narrative-fiction *reading* (not listening) fMRI dataset, with word-level timing; it is also the dataset on which most "LLM-to-brain alignment" work was done. Gap: no affect ratings, only 1 chapter, HP certainly in LLM pretraining.

### Wehbe, Vaswani, Knight & Mitchell (2014). "Aligning context-based statistical models of language with brain activity during reading." *EMNLP 2014*, pp. 233–243. [V]
- https://aclanthology.org/D14-1030/
- **Summary.** Same HP chapter, MEG (306 sensors, 1 kHz) in 3+ subjects; RNN-LM hidden states and next-word probabilities predicted MEG activity word-by-word, with context representations aligned to earlier/later sensors than the word-probability signal.
- **Relation.** First demonstration that neural-LM internal states predict brain activity during fiction reading; our question is whether such states also predict the *reader's affect*. MEG data not openly downloadable (form-based access; see data_sources.md).

### Toneva & Wehbe (2019). "Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)." *NeurIPS 2019*. arXiv:1905.11833. [V]
- https://arxiv.org/abs/1905.11833 ; https://papers.nips.cc/paper_files/paper/2019/file/749a8e6c231831ef7756db230b4359c8-Paper.pdf
- **Summary.** Fit ridge encoding models from layer-wise representations of ELMo, USE, BERT and Transformer-XL (context lengths 1–40 words) to the HP fMRI and MEG data. Middle layers and longer contexts predicted brain activity best; Transformer-XL kept improving with context; replacing BERT's shallow-layer attention with uniform attention improved brain prediction *and* syntactic probing accuracy.
- **Relation.** Methodological template (layer × context-length encoding sweep on HP) that we adapt, with affect rather than BOLD as the target. Gap: nothing about what the reader feels.

### Schrimpf, Blank, Tuckute, Kauf, Hosseini, Kanwisher, Tenenbaum & Fedorenko (2021). "The neural architecture of language: Integrative modeling converges on predictive processing." *PNAS* 118(45): e2105646118. [S]
- https://cbmm.mit.edu/publications/neural-architecture-language-integrative-modeling-converges-predictive-processing ; https://hdl.handle.net/1721.1/138214
- **Summary.** Benchmarks 43 language models against fMRI/ECoG of sentence/passage reading and self-paced reading times; GPT-2-XL-class models predict nearly 100% of explainable variance in the language network, and brain-predictivity correlates with next-word prediction performance but not with other NLP tasks.
- **Relation.** Establishes next-word prediction as the linking hypothesis between LLMs and the language network; we ask whether the same representations also carry the *affective* consequences of prediction (suspense/surprise). Gap: non-narrative stimuli, no affect.

### Caucheteux & King (2022). "Brains and algorithms partially converge in natural language processing." *Communications Biology* 5: 134. [S]
- DOAJ record https://doaj.org/article/6f8ac33012674bcab4e30674755ccd23
- **Summary.** Compared 102 transformer architectures' activations to MEG/fMRI of 100+ participants reading sentences; brain mapping peaks in middle layers and improves with language-modelling accuracy, and the brain's representations partially but not fully converge with those of LMs.
- **Relation.** Confirms middle-layer / prediction-driven alignment across many models — guides which layers we probe first. Gap: sentence lists, not narrative affect.

### Goldstein, Zada, Buchnik, … & Hasson (2022). "Shared computational principles for language processing in humans and deep language models." *Nature Neuroscience* 25: 369–380. [S]
- **Summary.** ECoG from epilepsy patients listening to a 30-min podcast story; the brain predicts upcoming words before onset, shows surprise/confidence-modulated post-onset responses, and GPT-2 contextual embeddings linearly map to neural activity, consistent with shared autoregressive prediction, pre-onset prediction and prediction-error signalling.
- **Relation.** Provides the neural evidence that *surprise* (prediction error) is explicitly represented during narrative comprehension; our behavioural/affective target asks whether that same quantity is what readers *feel* as surprise.

### Nastase, Liu, Hillman, … & Hasson (2021). "The 'Narratives' fMRI dataset for evaluating models of naturalistic language comprehension." *Scientific Data* 8: 250. [V]
- https://snastase.github.io/datasets/ds002345 ; OpenNeuro ds002345
- **Summary.** Data descriptor: 345 participants, 891 functional scans, 27 spoken stories (3–56 min, ~4.6 h audio) collected 2011–2018, BIDS-formatted, with stimulus audio, word-level alignments and behavioural comprehension data; released CC0.
- **Relation.** Largest naturalistic story fMRI resource for testing LLM-state → brain mappings; several stories are obscure (lower contamination risk). Gap: listening, not reading; no affect ratings bundled.

### "Decoding Naturalistic Emotion Dynamics from the Brain: An LLM-Enhanced Regression Framework." Zhang, Liu, Kvadsheim, Aasvær, Ye, Bonyadi, Ziaei & Gulla (2026). arXiv:2606.07707. [V]
- https://arxiv.org/abs/2606.07707
- **Summary.** Uses an LLM to annotate continuous multi-dimensional sentiment trajectories for the *Alice in Wonderland* audio narrative, then regresses these from fMRI dynamic functional connectivity (multi-target regression with graph-theoretical XAI). DFC snapshots outperform static ROI amplitudes; emotion-specific network configurations emerge.
- **Relation.** Shows the "LLM as affect annotator → brain decoding" pipeline on a public narrative dataset; but ground truth is LLM-generated, not human. Our design uses *human* affect/behaviour as target and LLM internals as predictor — the reverse direction, which avoids circularity.

---

## D. Reading-time / eye-tracking and surprisal

### Shain, Meister, Pimentel, Cotterell & Levy (2024). "Large-scale evidence for logarithmic effects of word predictability on reading time." *PNAS* 121(10): e2307876121. [V]
- PMC10927576; https://par.nsf.gov/servlets/purl/10559592
- **Summary.** Across six reading-time datasets (self-paced reading and eye-tracking, incl. Natural Stories, Dundee, Provo, Brown) and multiple LMs, reading time is a linear function of surprisal (−log p), supporting a logarithmic effect of predictability with no evidence of super-linear growth.
- **Relation.** Fixes the baseline behavioural link (surprisal → RT) that any affect-based predictor must add variance *beyond*. Our question is whether higher-level LLM-state quantities (belief change, uncertainty reduction) predict reading behaviour and affect over and above token surprisal.

### Oh & Schuler (2023). "Why Does Surprisal From Larger Transformer-Based Language Models Provide a Poorer Fit to Human Reading Times?" *TACL* 11: 336–350. [V]
- https://aclanthology.org/2023.tacl-1.20
- **Summary.** Surprisal from larger GPT-2/OPT/GPT-Neo models fits reading times *worse*; the degradation is driven by the largest models "over-predicting" rare/named-entity words that humans still find hard, i.e. models memorise more than humans do.
- **Relation.** Warns that bigger is not better for human-alignment and that memorisation of training text (e.g. Harry Potter) can distort predictability; motivates testing contamination and using obscure texts.

### Futrell, Gibson, Tily, Blank, Vishnevetsky, Piantadosi & Fedorenko (2021). "The Natural Stories Corpus: a reading-time corpus of English texts containing rare syntactic constructions." *Language Resources and Evaluation* 55: 63–77. (Earlier LREC 2018 version L18-1012.) [V repo; venue from search listing]
- https://github.com/languageMIT/naturalstories ; MIT DSpace https://dspace.mit.edu/handle/1721.1/137277
- **Summary.** Ten naturalistic English stories (~10k words) edited to contain rare syntactic constructions, with self-paced reading times from hundreds of MTurk participants, parses, frequencies and audio. Released CC BY-NC-SA 4.0; a May 2025 fix corrected a one-position misalignment in SPR data.
- **Relation.** Narrative-ish texts with word-level RTs and an fMRI companion in the literature; good for the behavioural arm. Gap: stories are adapted/edited and short; no affect ratings.

### Luke & Christianson (2018). "The Provo Corpus: A large eye-tracking corpus with predictability norms." *Behavior Research Methods* 50: 826–833. [V]
- https://osf.io/sjefs
- **Summary.** 84 readers' eye movements on 55 short passages (2,689 tokens; news, popular science, public-domain fiction) with cloze norms from 470 participants, including morphosyntactic and semantic predictability measures.
- **Relation.** Standard eye-tracking benchmark; fiction subset small. Useful as a sanity check for surprisal baselines, less so for narrative affect.

### Hollenstein, Rotsztejn, Troendle, Pedroni, Langer & Zhang (2018). "ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading." *Scientific Data* 5: 180291. [V]
- https://doi.org/10.17605/OSF.IO/Q3ZWS
- **Summary.** 12 participants, 1,107 sentences (SST movie reviews + Wikipedia), 128-ch EEG + EyeLink at 500 Hz, three reading tasks; CC BY 4.0.
### Hollenstein, Troendle, Zhang & Langer (2020). "ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation." *LREC 2020*. [V]
- https://aclanthology.org/2020.lrec-1.18 ; https://osf.io/2urht/
- **Summary.** 18 participants, 739 Wikipedia sentences, normal vs task-specific reading, simultaneous EEG + eye-tracking.
- **Relation (both).** Only public simultaneous EEG+ET reading data; sentiment-labelled SST sentences in 1.0 give a crude valence target. Gap: isolated sentences, no narrative continuity, so moment-by-moment narrative affect cannot be tested here.

---

## E. Emotion in text: lexicons, corpora, arcs

### Reagan, Mitchell, Kiley, Danforth & Dodds (2016). "The emotional arcs of stories are dominated by six basic shapes." *EPJ Data Science* 5: 31. [V]
- https://www.uvm.edu/storylab/share/papers/reagan2016b/ ; https://doaj.org/article/ed9cacc0f7284941a5907ab0fe5cbb0a
- **Summary.** Sliding-window hedonometer (labMT lexicon) valence over 1,327 Project Gutenberg stories, decomposed by SVD/clustering/SOM into six arc shapes (rags-to-riches, tragedy, man-in-a-hole, Icarus, Cinderella, Oedipus); some shapes are associated with higher download counts.
- **Relation.** The reference point for "emotional arc" as a text-derived trajectory; but arcs are lexicon-based and never validated against readers' felt emotion. We test whether LLM-state arcs predict *human* moment-by-moment affect.

### Mohammad & Turney (2013). "Crowdsourcing a Word–Emotion Association Lexicon." *Computational Intelligence* 29(3): 436–465 (NRC EmoLex); Mohammad (2018). "Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words." *ACL 2018* (NRC-VAD). [V lexicon pages; citations from memory]
- https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm ; https://www.saifmohammad.com/WebPages/nrc-vad.html ; terms https://saifmohammad.com/WebPages/AccessResource.htm
- **Summary.** Crowd-sourced word-level associations with 8 Plutchik emotions + 2 sentiments (~14k words) and best–worst-scaled VAD ratings for ~20k words. Free for non-commercial research, commercial licence via NRC; no redistribution.
- **Relation.** Lexicon baselines for valence/arousal trajectories against which LLM-state predictors are compared (and used by Christ et al. 2024 to map discrete story labels to VAD).

### Demszky, Movshovitz-Attias, Ko, Cowen, Nemade & Ravi (2020). "GoEmotions: A Dataset of Fine-Grained Emotions." *ACL 2020*. arXiv:2005.00547. [V]
- https://huggingface.co/datasets/google-research-datasets/go_emotions
- **Summary.** 58k Reddit comments labelled by 3 raters with 27 emotions + neutral; Apache-2.0. Used to train emotion classifiers / probes.
- **Relation.** Source of supervised emotion probes on LLM hidden states; but labels are writer-expressed emotion in social media, not reader-felt emotion in fiction.

### Buechel & Hahn (2017). "EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis." *EACL 2017*. [V repo]
- https://github.com/JULIELab/EmoBank (CC BY-SA 4.0)
- **Summary.** ~10k sentences (MASC genres incl. fiction, letters, news; SemEval-2007 headlines) annotated with VAD from both *writer* and *reader* perspective; per-annotator ratings released 2022.
- **Relation.** The only sizeable corpus with explicit *reader-perspective* affect ratings, including fiction sentences — a natural probe-training set. Gap: isolated sentences.

### Christ, Amiriparian, Milling, Aslan & Schuller (2024). "Modeling Emotional Trajectories in Written Stories Utilizing Transformers and Weakly-Supervised Learning." arXiv:2406.02251 (ACL 2024 Findings, venue from memory). [V arXiv]
- https://github.com/lc0197/emotional_trajectories_stories
- **Summary.** Re-annotates Alm's (2008) 169 children's stories (Grimm/Andersen/Potter, ~14.9k sentences) and maps discrete per-sentence labels to continuous V/A via NRC-VAD, then trains transformer regressors with weak supervision to predict sentence-level emotional trajectories.
- **Relation.** Provides sentence-level emotional-trajectory ground truth on narrative text (annotator-labelled text emotion, not reader self-report), publicly released.

---

## F. 2025–2026 work probing LLM internals for emotion/affect

### Sofroniew, Kauvar, Saunders, … Olah & Lindsey (2026). "Emotion Concepts and their Function in a Large Language Model." arXiv:2604.07729. (Anthropic) [V]
- https://arxiv.org/abs/2604.07729 ; https://www.anthropic.com/research/emotion-concepts-function
- **Summary.** In Claude Sonnet 4.5, identifies internal representations of emotion concepts that generalise across contexts, activate in proportion to the relevance of each emotion to the current context and to predicting upcoming text, and causally influence outputs (preferences, and rates of misaligned behaviours such as sycophancy or blackmail). Framed as "functional emotions" with no claim of subjective experience.
- **Relation.** Directly supports the premise that LLMs carry context-tracking emotion representations tied to next-text prediction; the open question we address is whether those representations align with *what a human reader feels* at the same point in a story.

### Reichman, Avsian & Heck (2025/2026). "Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models." arXiv:2510.22042. [V]
- https://arxiv.org/abs/2510.22042
- **Summary.** Finds a low-dimensional, depth-stable "universal emotional subspace" in LLM hidden states that generalises across eight emotion datasets in five languages, with strong linear-probe performance and a learned steering intervention that changes perceived emotion while preserving semantics.
- **Relation.** Provides probing and geometric tools; evaluates against text-emotion labels, not readers.

### Zhang & Zhong (2025). "Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion." arXiv:2510.04064. [V]
- https://arxiv.org/abs/2510.04064
- **Summary.** Linear probes on Qwen3/LLaMA hidden states over a 400k-utterance seven-emotion Reddit set: emotion is sharply linearly decodable, peaks in middle layers, strengthens with scale, and a system-prompted emotional tone persists for hundreds of tokens.
- **Relation.** Layer/scale guidance for probe placement; again social-media text, no human reader signal.

### Bigelow, Sarfati, Wurgaft, Lewis, McGrath, Merullo, Geiger & Lubana (2026). "The shape of stories inside neural networks." Goodfire research post. [V page; not a peer-reviewed paper]
- https://www.goodfire.ai/research/the-shape-of-stories-inside-neural-networks
- **Summary.** After each sentence of simple stories, an LLM rates six emotions 0–10; the ratings form a 2-D space resembling valence–arousal, and last-token activations trace a curved emotional trajectory manifold that moves sentence-by-sentence with the story. Compared against human valence/arousal judgements of the same sentences.
- **Relation.** The nearest existing demonstration that LLM activations trace an emotional arc through a story; limited to simple synthetic stories, and human comparison is text-judgement, not moment-by-moment reader affect.

### UCL/NTT (2025–2026). "Can large language models predict human emotions?" — ACII 2025 study (title/arXiv id not found). [S; details partial]
- https://www.ucl.ac.uk/uclic/news/2026/apr/can-large-language-models-predict-human-emotions
- **Summary.** GPT-4o estimated VAD intensity from participants' written descriptions of artworks; valence aligned strongly with human readers, arousal moderately, dominance weakly.
- **Relation.** Prompted (not internal-state) estimation; non-narrative. Marginal.

### Searched but not found (as of 2026-11-05)
- No paper was located that regresses frozen-LLM hidden states against human *continuous* suspense/valence/arousal ratings or reading times on literary narrative at sentence/word resolution. Closest: Zhang et al. CCN 2025 (surprise, event-level, 2 stories); Matlin et al. 2025 (prompted ratings); Goodfire 2026 (synthetic stories). This is the gap the planned paper fills.
