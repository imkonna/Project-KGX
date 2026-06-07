# Poster Session — Prep Pack
### Graph & Geometric Deep Learning for Drug Toxicity Prediction
*MSc Artificial Intelligence · Aristotle University of Thessaloniki · Supervisor: Prof. Grigorios Tsoumakas*
*Team: Konstantina Paschou · Georgios Kesoglidis · Theocharis Angelidis*

---

## How to use this pack

Read it once end-to-end, then drill the two things examiners actually test at a poster: (1) can you state the **point of the project in one breath**, and (2) can you **defend a choice** when pushed. Everything below is built around those two skills. The audience is CS/AI, so expect mostly *technology* questions — representation, architecture, evaluation, leakage — with a few problem-statement and biology probes. We weight the material accordingly.

**Contents**
1. The pitch, in three lengths (30 s / 2 min / 5 min)
2. The narrative spine — Why → What → How → Results → So what
3. Methods deep-dive (the technical backbone)
4. Results & how to interpret them
5. The compute story (GitHub Actions) — present this as engineering, not apology
6. Limitations & threats to validity
7. Numbers to memorize (cheat sheet)
8. FAQ — by category, with crisp answers
9. Division of labor & delivery tips
10. Glossary

---

## 1. The pitch, in three lengths

**30-second (elevator).**
"Molecules are graphs — atoms are nodes, bonds are edges. We asked whether models that read that structure directly beat classical fingerprint baselines at predicting drug toxicity, across five tasks of different size and balance. The headline: there's no universal winner. Graph networks win on the small, hard, imbalanced datasets; classical fingerprints win on the large, balanced ones. Along the way we found a leakage trap in a popular regression benchmark, and showed 3D geometry buys you uncertainty estimates rather than accuracy."

**2-minute.** Add: the four method families (classical fingerprints + trees; a frozen chemical language model; 2D GNNs; a 3D geometric net), the evaluation protocol (scaffold splits, three seeds — the realistic generalization test), and the three deep dives: (a) under scaffold splits every GNN's LD50 R² collapses below zero, exposing the cited random-split R²≈0.49 as memorization; (b) SchNet's 3D conformers never beat the 2D bond-aware GINE, but the spread of predictions across conformers flags unreliable molecules for free; (c) forcing one shared backbone across the 12 Tox21 endpoints *hurts* — the biology is too diverse to share.

**5-minute.** Walk the poster left-to-right: problem → pipeline → main result → full benchmark table → the three deep dives → conclusions. Land every claim on a number from the cheat sheet (§7).

---

## 2. The narrative spine

**Why.** Most drug candidates fail, and a large share fail on safety. Predicting toxicity *before* synthesis saves money and time. Molecules have a natural graph structure, and modern GNNs can learn on it directly — so the timely scientific question is whether that structural learning actually pays off versus the fingerprint+tree pipelines the field has used for years.

**What.** A controlled, like-for-like benchmark: 8 methods × 5 toxicity datasets (+ the 12-endpoint Tox21 panel), one shared scaffold splitter, three seeds, AUROC (R² for the LD50 regression). The contribution is not a new model — it's a *rigorous, leakage-aware comparison* plus three findings the standard leaderboards miss.

**How.** Four representation families so the comparison is fair (see §3). Identical splits and metrics across methods. Class-weighted training for the imbalanced tasks. Everything run as an idempotent, resumable pipeline on CI (see §5).

**Results.** The winner depends on the task; a leakage finding on LD50; 3D → uncertainty not accuracy; multi-task transfer hurts; frozen language-model features don't transfer. (Numbers in §4 and §7.)

**So what.** Method choice should follow dataset size, class balance, and chemistry — not a reflex toward the newest architecture. And *how you split your data* changes the conclusion more than *which model you pick*.

---

## 3. Methods deep-dive (the technical backbone)

This is the section most questions come from. Know it cold.

### 3.1 Inputs & representations

- **SMILES** — a text string encoding a molecule (atoms, bonds, rings, branches). The raw input for every method; we parse it with RDKit.
- **Morgan / ECFP4 fingerprint** — a fixed 1024-bit vector. Each atom's circular neighborhood up to **radius 2** (hence ECFP*4* = diameter 4) is hashed to bits. It is a *bag of substructures*: great at flagging "does this fragment exist," blind to global arrangement. Input to RF and XGBoost.
- **Molecular graph** — nodes = atoms with a **9-dim feature vector** (atomic number, degree, formal charge, aromaticity, #H, in-ring, hybridization, valence, radical electrons); edges = bonds. For the bond-aware model we add **4 edge features** (bond type 1/1.5/2/3, aromatic, in-ring, stereo). Input to the GNNs.
- **3D conformers** — actual 3D coordinates. We generate them with RDKit **ETKDG** + UFF optimization (10 per molecule), and feed atomic numbers + positions to SchNet.
- **Tokenized SMILES** — input to ChemBERTa.

### 3.2 The eight methods

**Classical (baselines).**
- **Random Forest** and **XGBoost** on Morgan fingerprints. Strong, fast, hard-to-beat baselines. 500 trees; XGBoost depth 6, lr 0.1.

**Chemical language model.**
- **ChemBERTa** — a RoBERTa pre-trained on ~77M SMILES via masked-language-modeling (`DeepChem/ChemBERTa-77M-MLM`). We **freeze** the backbone and train a small MLP head on the `[CLS]` embedding. (Frozen = no GPU-heavy fine-tuning; also the cleanest test of whether the *pre-trained features* transfer.)

**2D GNNs** (message-passing on the graph). All share a skeleton: 2–3 conv layers → global pooling → linear head.
- **GCN** (Kipf & Welling) — symmetric-normalized **mean** aggregation of neighbors; spectral motivation.
- **GIN** (Xu et al.) — **sum** aggregation + an MLP. Provably as discriminative as the **Weisfeiler–Lehman** isomorphism test — the most *expressive* standard message-passing GNN. Sum-pooling preserves multiset information that mean/max throw away.
- **GAT** (Veličković et al.) — learns **attention weights** over neighbors (4 heads).
- **GraphSAGE** (Hamilton et al.) — sample-and-aggregate; designed to be **inductive** (generalize to unseen nodes/graphs).
- **GINE** — GIN extended to consume **edge (bond) features** via `GINEConv`. Our "hardened" GNN; the GIN→GINE comparison is our edge-feature ablation.

**3D geometric net.**
- **SchNet** (Schütt et al.) — **continuous-filter convolutions** over interatomic *distances*. Rotation/translation invariant; built to model physical/quantum interactions from geometry. Our test of whether 3D shape adds predictive signal.

### 3.3 Training details (have these ready)

- Loss: **BCEWithLogitsLoss** with **`pos_weight = n_neg/n_pos`** for class imbalance; **MSE** for LD50 regression.
- Optimizer **Adam**, lr `1e-3`, weight decay; **ReduceLROnPlateau**; **early stopping** on validation loss (patience 15–20).
- Hidden dim 64–128; GIN/GINE use **global add-pool**, GCN/GAT/SAGE use **global mean-pool**.
- Everything is small on purpose — see the compute story (§5).

### 3.4 Evaluation protocol (the part that makes or breaks credibility)

- **Bemis–Murcko scaffold split.** Reduce each molecule to its ring-system-plus-linker scaffold; allocate whole scaffold groups to train/val/test so the three sets are **structurally disjoint**. This simulates real screening, where you predict on *new chemistry*, not analogs of what you've seen. 70/10/20.
- **3 seeds**, report **mean ± std**. The **†** on the poster marks std > 0.03 (i.e., "treat this number with caution").
- **AUROC** for classification (threshold-free, robust to imbalance); **R²** for LD50 regression. Negative R² = worse than predicting the mean.

---

## 4. Results & how to interpret them

**(1) No universal winner — it depends on the task.**
GNNs win where data is small/imbalanced and the toxic signal is subtle and holistic: **DILI +0.043** (GIN 0.870 vs RF 0.827) and **ClinTox +0.020** (GIN 0.820 vs XGB 0.800). Classical fingerprints win where data is large and balanced and toxicity is driven by specific **reactive substructures** a fingerprint already encodes: **AMES −0.012** and **hERG −0.018** (RF on top). *Interpretation, stated honestly:* the AMES/hERG margins are small and partly within seed noise — the robust claim is "no method dominates across regimes," not "GNNs are 0.012 worse on AMES."

**(2) Frozen ChemBERTa finishes last on every classification task** despite 77M-molecule pre-training. MLM features optimized for SMILES reconstruction don't reshape into toxicity-discriminative features without fine-tuning, and they transfer poorly to unseen scaffolds. Lesson: pre-training is not a free lunch under distribution shift.

**(3) LD50 is a leakage cautionary tale.** Under **random** splits GIN looks strong (R² ≈ 0.49). Under **scaffold** splits **every GNN collapses below zero** (GIN −0.52, GINE −0.76; even XGB −0.05; RF barely positive at 0.030). The random-split score was measuring **scaffold memorization** — near-duplicate analogs split across train and test — not generalization. This is the single most "CS-examiner-friendly" result: it's about experimental hygiene.

**(4) 3D buys uncertainty, not accuracy.** SchNet (3D) **never beats** GINE (2D + bonds): ClinTox 0.711 vs 0.810 (gap ≈ 0.10), AMES 0.756 vs 0.773 (gap 0.017 — shrinks as data grows). But the **standard deviation of SchNet's predictions across the 10 conformers** correlates with its errors, so it works as a free, post-hoc **uncertainty/abstention signal**. Novelty hook: this conformer-variance-as-uncertainty angle isn't documented on these TDC toxicity benchmarks.

**(5) Multi-task transfer hurts.** One shared GIN backbone across all 12 Tox21 endpoints vs 12 single-task models → **macro-AUROC −0.024**. Only a couple of biologically related endpoints (e.g., NR-AR-LBD, SR-ATAD5) benefit. The 12 assays span distinct nuclear-receptor and stress-response pathways; forcing one representation creates **negative transfer**.

---

## 5. The compute story (GitHub Actions) — frame it as engineering

This is a strength, not an excuse. Examiners in a CS department *like* this.

**The problem.** 8 methods × 5 datasets × 3 seeds, plus 12 Tox21 endpoints and SchNet's per-molecule conformer generation — hundreds of training runs. No GPU cluster, and we didn't want to lock up personal laptops for days.

**The solution.** We turned the experiment runner into an **idempotent, resumable pipeline** and executed it on **GitHub Actions CI runners** — effectively a free, ephemeral compute pool.
- Each `(dataset, method, seed)` is an atomic **unit of work**; its result is checkpointed to disk (JSON) the moment it finishes.
- Re-running the runner **skips any completed unit** (it checks for the result file), so a job that hits the runner time limit can simply be restarted and **resumes where it left off**.
- A partial results CSV is written after every run, so nothing is lost mid-sweep.
- The environment is pinned, so runs are **reproducible** and results are **versioned in the repo**.

**The honest trade-off (and why it shaped the models).** CI runners are CPU-bound and time-limited. That's *why* the GNNs are small (hidden 64–128, 2–3 layers), *why* ChemBERTa is frozen rather than fine-tuned, and *why* SchNet trains on one canonical conformer. State this proactively — it pre-empts the "your models are tiny" question and turns a constraint into a deliberate, defensible design.

**Be ready to specify** (fill in your exact setup): whether you used a build **matrix** to parallelize, whether results were persisted via **Actions artifacts** or **committed back** to the repo, runner type, and per-job timeout. Have one sentence ready for each.

---

## 6. Limitations & threats to validity

Say these *before* you're asked — it signals maturity.

- **Compute-bound model scale.** Small GNNs, frozen (not fine-tuned) ChemBERTa, single-conformer SchNet training. Bigger models *might* shift individual numbers — but the *task-dependence* and *leakage* findings are about protocol, not capacity, so they're robust.
- **No per-model hyperparameter search.** Shared, sensible defaults for fairness; we did not tune each architecture to its ceiling.
- **Five datasets + Tox21.** Conclusions are about toxicity endpoints of this kind, not all of cheminformatics.
- **Single fingerprint / single 3D method.** Morgan-only for classical; SchNet-only for 3D (no DimeNet/EGNN). 
- **AUROC under heavy imbalance** (ClinTox 8%) can look optimistic; PR-AUC would be a useful complement.
- **Conformer uncertainty** is shown correlational on ClinTox; broader validation (AMES, calibration) is future work.

**Future work:** fine-tune ChemBERTa; multi-conformer training/augmentation for SchNet; compare to MC-dropout uncertainty; more 3D architectures; PR-AUC and calibration reporting.

---

## 7. Numbers to memorize (cheat sheet)

**Datasets**

| Dataset | Task | n | % positive | Biology in one line |
|---|---|---|---|---|
| DILI | binary | 475 | ~45% | Drug-induced liver injury |
| ClinTox | binary | 1,484 | ~8% | Failed clinical trials for toxicity |
| AMES | binary | 7,255 | ~50% | Bacterial mutagenicity (Ames test) |
| hERG | binary | 13,445 | ~40% | hERG cardiac K⁺-channel blockade |
| LD50 | regression | 7,385 | — | Acute oral lethal dose (log scale) |
| Tox21 | 12 binary | ~7–8k | varies | NR + SR toxicity pathways |

**Full benchmark — AUROC (R² for LD50); bold = best per column**

| Method | AMES | ClinTox | DILI | hERG | LD50 (R²) |
|---|---|---|---|---|---|
| RF | **0.792** | 0.733 | 0.827 | **0.798** | **0.030** |
| XGBoost | 0.757 | 0.800 | 0.815 | 0.794 | −0.049 |
| ChemBERTa | 0.708 | 0.625 | 0.701 | 0.688 | — |
| GCN | 0.718 | 0.691 | 0.845 | 0.692 | −0.073 |
| GIN | 0.780 | **0.820** | **0.870** | 0.778 | −0.523 |
| GAT | 0.749 | 0.745 | 0.836 | 0.703 | −0.020 |
| GraphSAGE | 0.774 | 0.815 | 0.840 | 0.716 | −0.009 |
| GINE | 0.758 | 0.810 | 0.860 | 0.780 | −0.755 |

**Headline deltas (best GNN − best classical):** DILI **+0.043** · ClinTox **+0.020** · AMES **−0.012** · hERG **−0.018** · LD50 **−0.050**.

**3D vs 2D:** ClinTox SchNet 0.711 vs GINE 0.810 (gap ≈ 0.10) · AMES SchNet 0.756 vs GINE 0.773 (gap 0.017).

**LD50 leakage:** random-split GIN R² ≈ 0.49 → scaffold-split every GNN negative (GIN −0.52, GINE −0.76).

**Multi-task Tox21:** macro-AUROC **−0.024** (shared GIN vs 12 single-task).

---

## 8. FAQ — by category

### A. Problem statement & framing
**Q. What exactly are you predicting?** A per-molecule label: for four datasets a binary toxicity flag (mutagenic / cardiotoxic / liver-injuring / clinically toxic), and for LD50 a continuous lethal-dose value. Input is just the molecule (a SMILES string).

**Q. Why is this an ML problem worth doing?** Wet-lab tox testing is slow and expensive; a model that ranks candidates by risk before synthesis is high-value triage. And it's a clean testbed for "does structure-aware deep learning beat classical features?"

**Q. Why these five datasets?** They span the axes that matter: size (475 → 13k), balance (8% → 50%), and task type (classification + one regression). That's what lets us say *when* a method wins, not just *whether*.

### B. Representation
**Q. What's the difference between a fingerprint and a graph here?** A Morgan fingerprint is a fixed bag-of-substructures vector — it asks "is fragment X present?" A graph keeps atoms, bonds, and their wiring, and lets the model *learn* which structural patterns matter. Fingerprints are strong when local fragments are decisive; graphs help when the signal is more global.

**Q. Why ECFP4 / radius 2?** It's the field-standard circular fingerprint; radius 2 captures each atom's 2-bond neighborhood, a good balance of specificity and generality. 1024 bits keeps it compact.

**Q. What node/edge features did you use?** 9 atom features (atomic number, degree, charge, aromaticity, #H, ring membership, hybridization, valence, radicals) and, for GINE, 4 bond features (type, aromatic, in-ring, stereo).

### C. Architectures (expect the most here)
**Q. Why so many GNN variants?** They aggregate neighbor information differently — GCN normalized mean, GIN sum+MLP, GAT attention, SAGE sample-aggregate. Comparing them isolates *what kind* of inductive bias helps per task.

**Q. Why is GIN "the most expressive"?** Xu et al. proved sum-aggregation + MLP makes GIN as discriminative as the Weisfeiler–Lehman graph-isomorphism test — the theoretical ceiling for standard message-passing. Mean/max pooling lose multiset information that sum preserves. That it tops our imbalanced tasks is consistent with the theory.

**Q. GIN vs GINE — what did bond features buy?** GINE injects edge (bond) features into the update. It edges out plain GIN on DILI and hERG, and matches it elsewhere — modest but real gains on the structure-sensitive tasks. (This is our ablation.)

**Q. What is message passing / global pooling, in one sentence each?** Message passing = each node repeatedly updates its vector from its neighbors' vectors (after k layers a node "sees" its k-hop neighborhood). Global pooling = collapse all node vectors into one graph vector (we sum for GIN/GINE, mean otherwise) before the classifier.

**Q. How is SchNet different from the 2D GNNs?** It convolves over **continuous interatomic distances** in 3D rather than discrete bonds, using continuous-filter convolutions, and is invariant to rotation/translation. It needs 3D conformers as input.

**Q. Why freeze ChemBERTa instead of fine-tuning?** Two reasons: fine-tuning a 77M-param transformer is GPU-heavy and outside our CI compute budget; and freezing is the clean test of whether the *pre-trained representation itself* transfers. It doesn't, here — a useful negative result. Fine-tuning is explicit future work.

### D. Evaluation, metrics, statistics
**Q. Why scaffold splits instead of random?** Random splits leak: analogs of test molecules sit in training, so a model can memorize rather than generalize. Scaffold splits force structurally novel test molecules — the realistic deployment setting. Our LD50 result is the proof: random-split R²≈0.49 vanished to negative under scaffold splits.

**Q. Why AUROC?** It's threshold-independent and robust to class imbalance — it measures the probability a random positive is ranked above a random negative. For LD50 (continuous) we use R².

**Q. What does negative R² mean?** The model predicts worse than just outputting the dataset mean. For the GNNs on scaffold-split LD50, it means they failed to generalize at all.

**Q. Are deltas like +0.02 statistically meaningful?** We report mean ± std over 3 seeds and flag std > 0.03 with †. Small deltas (AMES −0.012) are near noise — which is exactly why our claim is "depends on the regime," not "X is better by 0.012." Honest framing wins points.

**Q. Three seeds is few — why not more?** Compute budget on CI. Three is enough to expose the std and the LD50 collapse, which are large effects; we'd add seeds with more compute.

### E. Results interpretation
**Q. Why would GNNs win on the *small* dataset (DILI)?** Counter-intuitive, granted. Our reading: DILI/ClinTox toxicity is driven by subtler, more global structure that fingerprints fragment away, and class-weighting let the GNNs exploit it; AMES/hERG are dominated by specific reactive substructures that a fingerprint encodes directly, so trees already capture the signal and extra data favors them. We present this as interpretation of an empirical pattern, with the noise caveat.

**Q. Doesn't the frozen-ChemBERTa result just mean you used it wrong?** It means *frozen* features don't transfer — which is the claim we make, not "language models are useless." Fine-tuned ChemBERTa could do better; that's future work. The negative result is still informative.

**Q. If 3D doesn't help accuracy, why include SchNet?** Because the honest answer to "does geometry help?" is part of the contribution — and because the conformer-variance uncertainty signal is a genuinely useful by-product.

### F. Engineering / reproducibility
**Q. How did you run all this without a cluster?** Idempotent, resumable runner on GitHub Actions CI; per-run checkpoints; restart-to-resume; pinned environment. (See §5.) 

**Q. Doesn't CI being CPU-only bias the comparison?** It caps absolute performance for everyone equally, so the *relative* comparison stays fair. It does mean we don't report each model's ceiling — a stated limitation.

### G. Biology (lighter, but be ready)
**Q. What do the assays actually measure?**
- **AMES** — the Ames test: do bacteria mutate when exposed? A mutagenicity / genotoxic-carcinogenicity proxy.
- **hERG** — does the molecule block the hERG cardiac potassium channel? Blockade can prolong the QT interval and cause arrhythmia, a classic cause of drug withdrawal.
- **DILI** — drug-induced liver injury; another leading withdrawal cause.
- **ClinTox** — did the drug fail clinical trials for toxicity (vs FDA-approved)?
- **LD50** — the dose lethal to 50% of animals; acute systemic toxicity.
- **Tox21** — 12 high-throughput assays across nuclear-receptor (NR-*) and stress-response (SR-*) pathways.

**Q. What's a "reactive substructure" / structural alert?** A fragment associated with toxicity — e.g., aromatic nitro groups, epoxides, Michael acceptors for mutagenicity. Fingerprints flag these directly, which is part of why classical methods do well on AMES/hERG.

**Q. Do you need to be chemists to do this?** No — we treat molecules as data (graphs / fingerprints / strings). The biology tells us *why* certain features matter and how to read results, but the modeling is general ML.

### H. Likely "gotcha" questions — short defenses
- *"Your models are too small to conclude anything."* → Scale is capped by our CI compute, equally for all methods; the **protocol** findings (leakage, task-dependence, negative transfer) don't depend on capacity. Stated as a limitation.
- *"You're just rediscovering that scaffold splits are hard."* → We quantify it on a *specific, widely cited* benchmark (LD50 R²≈0.49 → negative) and show it flips a published-looking result. That's the value.
- *"Isn't +0.043 on DILI just luck on n=475?"* → Possibly inflated by small-n variance; that's why we report std and lean on the *pattern across datasets*, not any single delta.
- *"Why not a GPU on Colab?"* → Sessions are interruptible and not reproducible/versioned; our CI pipeline is restartable, pinned, and tracked in the repo.

---

## 9. Division of labor & delivery tips

**Suggested ownership (adapt freely):**
- **Person A — Problem & data.** Owns the why, the datasets, scaffold splits, biology questions. First contact for visitors.
- **Person B — Models & training.** Owns the architecture deep-dive (GCN/GIN/GAT/SAGE/GINE/SchNet/ChemBERTa), training details, the GIN→GINE ablation.
- **Person C — Results, evaluation & engineering.** Owns the benchmark table, the leakage finding, 3D/uncertainty, multi-task, and the GitHub Actions story.

Everyone should be able to give the **30-second pitch** and field the top FAQ in §8C/D.

**At the board:**
- Open with the one-line thesis, then ask "want the 1-minute or the deep version?" — let the visitor set depth.
- Always end an answer on a **number** from the cheat sheet.
- If you don't know, say "we didn't test that — here's what we'd expect and why." Examiners reward honest reasoning over bluffing.
- Keep one finger on the figure you're discussing; walk left-to-right.
- Practice the **leakage story** until it's smooth — it's your most memorable, most CS-flavored result.

---

## 10. Glossary (fast definitions)

- **SMILES** — text encoding of a molecule.
- **Morgan / ECFP4** — 1024-bit circular substructure fingerprint, radius 2.
- **Scaffold (Bemis–Murcko)** — a molecule's ring systems + linkers, side chains stripped.
- **Scaffold split** — train/test split with disjoint scaffolds; the realistic generalization test.
- **Message passing** — nodes iteratively update from neighbors; k layers ⇒ k-hop receptive field.
- **Global pooling** — aggregate node vectors into one graph vector (sum / mean).
- **GCN / GIN / GAT / GraphSAGE** — mean / sum+MLP / attention / sample-aggregate GNNs.
- **GINE** — GIN that also consumes bond (edge) features.
- **WL test** — Weisfeiler–Lehman graph isomorphism test; GIN matches its power.
- **SchNet** — 3D GNN using continuous-filter convolutions over interatomic distances.
- **ChemBERTa** — RoBERTa pre-trained on SMILES; we use it frozen + a small head.
- **Conformer / ETKDG** — a 3D shape of a molecule / RDKit's method to generate one.
- **AUROC** — threshold-free ranking metric; P(random positive ranked above random negative).
- **R²** — fraction of variance explained; negative ⇒ worse than the mean.
- **pos_weight** — class-imbalance weight (n_neg/n_pos) in the loss.
- **Negative transfer** — multi-task sharing that *hurts* vs separate models.
- **Idempotent / resumable runner** — re-running skips finished work; safe to interrupt.
