# Graph Neural Networks for Drug Toxicity Prediction
## A Comparative Study of GCN, GIN, GAT, and GraphSAGE

**Course:** Advanced Topics in Machine Learning — AUTH, MSc Artificial Intelligence
**Project:** KGX — Toxicity Prediction
**Period:** April 2026

---

## Abstract

We build a Graph Neural Network pipeline for predicting drug toxicity from molecular
structure, evaluating four architectures — GCN, GIN, GAT, and GraphSAGE — on
**17 toxicity prediction tasks** drawn from Therapeutics Data Commons (TDC). The tasks
span binary classification of drug-induced toxicity (AMES, hERG, ClinTox, DILI),
regression of acute lethal dose (LD50), and the 12 individual Tox21 nuclear-receptor /
stress-response endpoints.

Our initial pipeline used a small per-atom feature vector (4 properties), a compact
hidden dimension, and an unweighted binary cross-entropy loss. Training revealed three
characteristic failure modes — F1 scores collapsing to zero on imbalanced datasets,
weak regression performance on LD50, and an apparent capacity bottleneck on the harder
Tox21 endpoints — all pointing back to limited information per node and to the loss
function being agnostic to class balance. We then upgraded the molecular encoding to
**9 atom-level descriptors**, scaled the hidden dimension by 4×, and replaced the loss
with a class-weighted variant whose weights are derived per-dataset from the training
split. These changes improved Tox21 macro-AUROC by ~3 points (best model: GIN at 0.781),
roughly tripled mean F1 for the architectures that had previously collapsed, and lifted
LD50 R² from 0.43 to 0.49.

Across both phases, **GIN consistently outperforms** the other architectures, winning
the majority of individual datasets and dominating the Tox21 macro benchmark. We report
one regression worth flagging honestly: GraphSAGE underperforms after the upgrades on
three of the most imbalanced classification datasets, suggesting its
concatenate-and-mean aggregation is more sensitive to loss reweighting than the others.

---

## 1. Problem & Motivation

Drug toxicity is the single largest reason candidate molecules fail in clinical trials
or are withdrawn from the market. The ability to predict toxicity from molecular
structure alone — before synthesis, before assays, before animal models — has direct
economic and ethical value. Molecules are naturally represented as **graphs**: atoms
are nodes, bonds are edges. Graph Neural Networks (GNNs) operate directly on this
representation, in contrast to fingerprint-based methods which compress structural
information into a fixed binary vector at the input.

This project asks: **for the toxicity-prediction task, which GNN architecture works
best, and what data-side issues limit performance?**

---

## 2. Datasets

All datasets are loaded via the [Therapeutics Data Commons](https://tdcommons.ai/) Python SDK.
TDC provides each dataset as a pandas `DataFrame` with columns `Drug` (SMILES string),
`Drug_ID`, and `Y` (label or scalar target), along with deterministic train/valid/test
splits.

### 2.1 Toxicity datasets (5)

| Dataset | Task | Size | Positive ratio | Description |
|---|---|---|---|---|
| AMES | Binary classification | 7,255 | ~50% | DNA mutagenicity (cancer risk) |
| hERG_Karim | Binary classification | 13,445 | ~40% | Cardiac arrhythmia risk |
| ClinTox | Binary classification | 1,484 | ~8% | Clinical-trial toxicity failure |
| DILI | Binary classification | 475 | ~45% | Drug-induced liver injury |
| LD50_Zhu | Regression | 7,385 | — | Acute lethal dose, log scale |

### 2.2 Tox21 — 12 individual binary endpoints

A single ~7,800-molecule dataset screened against 12 different biological pathways. We
train a separate model per endpoint (one binary head each) rather than a single
multi-output model.

| Endpoint | Description | Pos. ratio |
|---|---|---|
| NR-AR, NR-AR-LBD | Androgen receptor | 4–5% |
| NR-AhR | Aryl hydrocarbon receptor | 13% |
| NR-Aromatase | Aromatase enzyme | 5% |
| NR-ER, NR-ER-LBD | Estrogen receptor | 4–13% |
| NR-PPAR-gamma | Peroxisome receptor | 4% |
| SR-ARE | Antioxidant response | 14% |
| SR-ATAD5 | DNA damage response | 4% |
| SR-HSE | Heat shock response | 5% |
| SR-MMP | Mitochondrial function | 14% |
| SR-p53 | p53 tumour suppressor | 6% |

### 2.3 Excluded: DAVIS / KIBA

Drug-target interaction (DTI) datasets are loaded for completeness but excluded from
the main evaluation. They require encoding both the drug and the target protein, which
is a fundamentally different pipeline (joint drug-graph + protein-sequence model).
This is identified as future work.

---

## 3. How the Data is Used

This section describes how the raw data — SMILES strings paired with labels — is
transformed into the tensor inputs that the GNN consumes. Understanding this pipeline
is critical because every subsequent design decision (model choice, loss function,
upgrade strategy) is constrained by what information actually reaches the model.

### 3.1 From SMILES strings to graph tensors

A SMILES string (Simplified Molecular Input Line Entry System) is a one-line text
representation of a molecule. For example, aspirin is `CC(=O)OC1=CC=CC=C1C(=O)O`.
This is a *human-readable* format; it is not directly usable by a neural network.

We convert each SMILES through the following pipeline:

```
SMILES string                                 ("CC(=O)Oc1ccccc1C(=O)O")
        │
        ▼  RDKit parser
RDKit Mol object                             (atom and bond objects with chemistry metadata)
        │
        ▼  per-atom feature extraction
Node feature matrix X ∈ ℝ^(n_atoms × F)       (F descriptors per atom)
        │
        ▼  per-bond enumeration (both directions for undirected graph)
Edge index E ∈ ℤ^(2 × 2·n_bonds)              (sender, receiver atom indices)
        │
        ▼  PyTorch Geometric `Data` wrapper
Data(x=X, edge_index=E, y=label)              (one graph object per molecule)
```

The resulting `Data` object is what the GNN sees — a **node feature matrix** plus an
**edge index**. There is no atom ordering assumption, no fixed size, and no global
descriptor: every property the model can learn from must be either in `X` (per atom)
or implicit in the bond connectivity.

### 3.2 Atom encoding (initial design)

Each atom is summarised by a 4-dimensional feature vector containing the most basic
chemical properties: which element it is, how many bonds it makes, whether it carries
a formal charge, and whether it is part of an aromatic ring.

| Index | Feature | Type | Example values |
|---|---|---|---|
| 0 | Atomic number | int | C=6, N=7, O=8, S=16 |
| 1 | Degree | int | 1–4 (number of bonded neighbours) |
| 2 | Formal charge | int | −1, 0, +1 |
| 3 | Aromaticity | bool | 0 or 1 |

These four properties are deliberately minimal — they describe what a chemist would
read off the structural formula at a glance, with no reference to ring membership,
hydrogens, or 3D geometry. The intent was to give the GNN room to *learn* higher-order
structure from connectivity rather than hand it pre-computed descriptors.

### 3.3 Bond encoding

Bonds are represented purely as **edges** in the graph — sender-receiver pairs of atom
indices — with no additional features attached. Each chemical bond produces *two*
entries in the edge index (one per direction) so the graph is treated as undirected.

```python
for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    edge_index += [[i, j], [j, i]]
```

Bond *type* (single / double / aromatic), stereochemistry, and ring membership at the
edge level are not encoded. The GNN therefore knows *which atoms are connected* but
not *how strongly*.

### 3.4 Batching variable-sized molecules

A core practical question with GNNs is how to put molecules of different sizes (5
atoms vs 50 atoms) into the same training batch. PyTorch Geometric handles this by
**concatenating graphs into a single big disconnected graph** per batch, and supplying
a `batch` tensor that records which graph each node originally came from:

```
Batch of 3 graphs: [molecule_A (12 atoms), molecule_B (8 atoms), molecule_C (20 atoms)]

x:     stacked 40-row node feature matrix
batch: [0,0,...,0,  1,1,...,1,  2,2,...,2]   (12 zeros, 8 ones, 20 twos)
```

`batch` is what allows the **global pooling** step at the end of the model to
recombine per-atom embeddings into one vector per molecule — by averaging or summing
all nodes that share the same batch index.

### 3.5 What the GNN actually sees

For a single forward pass with batch size B and total atoms N across all molecules in
the batch:

| Input | Shape | What it carries |
|---|---|---|
| `x` | (N, 4) | Per-atom features |
| `edge_index` | (2, 2·E) | Sender and receiver atom indices for each bond direction |
| `batch` | (N,) | Graph-membership index per atom |
| `y` | (B,) | Target label or value, one per molecule |

The GNN propagates information along edges (each atom collects messages from its
bonded neighbours), repeats this over multiple layers (so atoms further out start
contributing), pools the resulting node embeddings into one vector per graph, and
passes that vector through a linear head to produce the final prediction.

This pipeline determines the **information ceiling** for the model. If a property
needed to predict toxicity is not in `x`, not in `edge_index`, and not derivable from
the graph topology, the model cannot learn it — no matter how deep or wide the GNN is.

### 3.6 Class-imbalance preview

Before any modelling, looking at the positive ratios in §2 makes one issue obvious:
many of these targets are heavily skewed. Tox21_SR-ATAD5 has only ~4% positive
samples; ClinTox has ~8%. A naive classifier predicting "non-toxic" for everything
would already be 96%-accurate on Tox21_SR-ATAD5 — but useless. This shapes how we
choose evaluation metrics (§4.3) and motivates the loss-function upgrade in §8.

---

## 4. Models

### 4.1 GNN Architectures

All four models share the same skeleton:

```
graph (x, edge_index, batch)
        │
        ▼  Conv layer 1  →  ReLU/ELU
        │
        ▼  Conv layer 2  →  ReLU/ELU
        │
        ▼  global pooling  (mean or sum over each graph)
        │
        ▼  Linear  →  1 output
        │
        ▼  prediction (logit for classification, scalar for regression)
```

The only differences between the four are the conv layer's aggregation rule:

| Model | Aggregation | Pooling | Theoretical property |
|---|---|---|---|
| **GCN** | Degree-normalised mean | Mean | Symmetric Laplacian smoothing |
| **GIN** | Learnable sum (ε-parameterised) with MLP | Sum | As powerful as 1-WL graph isomorphism test |
| **GAT** | Attention-weighted mean (4 heads → 1) | Mean | Learns which neighbours matter |
| **GraphSAGE** | Concatenate self + mean(neighbours) | Mean | Inductive; generalises to unseen graphs |

References: Kipf & Welling 2017 (GCN); Xu et al. 2019 (GIN); Veličković et al. 2018 (GAT); Hamilton et al. 2017 (GraphSAGE).

### 4.2 Training Procedure

- **Optimiser:** Adam with weight decay 5×10⁻⁴
- **LR scheduler:** ReduceLROnPlateau (factor 0.5, patience 10 epochs)
- **Loss (classification):** Binary cross-entropy with logits
- **Loss (regression):** MSE
- **Early stopping:** patience 18–25 epochs on validation loss
- **Batch size:** 16 for small datasets (DILI, ClinTox), 32 otherwise
- **`drop_last=True`** on the training DataLoader — prevents size-1 batches that
  crash GIN's BatchNorm layer
- **Reproducibility:** seed 42 across `random`, `numpy`, `torch`, and CUDA

### 4.3 Evaluation Metrics

| Metric | When | Why |
|---|---|---|
| **ROC-AUC** | Primary metric for classification | Threshold-independent; correctly handles imbalance |
| **F1 (positive class)** | Secondary | Reveals whether the model actually predicts the minority class |
| Accuracy | Reported but not used for comparison | Misleading under imbalance — predicting all-negative gives 95% accuracy on Tox21 |
| **R²** | LD50 regression | Fraction of variance explained |
| RMSE | LD50 regression | Same units as the target |

> **Methodological note for the report.** ROC-AUC is the right primary metric on
> imbalanced toxicity data. A model that always predicts "non-toxic" achieves 95%
> accuracy on Tox21_SR-ATAD5 (4% positive) but ROC-AUC = 0.5 — correctly identifying
> it as useless. This metric choice becomes critical in §6 where we observe that some
> model–dataset pairs have high ROC-AUC but F1 = 0, and again in §8 where we use F1
> as the diagnostic signal that the upgrades worked.

---

## 5. Initial Configuration

The first experiment trained all 17 datasets × 4 architectures = 68 models with the
following configuration:

| Parameter | Value |
|---|---|
| Atom features | 4 (atomic number, degree, formal charge, aromaticity) |
| Hidden dimension | 16 |
| Number of conv layers | 2 |
| Learning rate | 0.01 |
| Loss (binary) | `BCEWithLogitsLoss`, unweighted |
| Decision threshold | Fixed at 0.5 |
| Total compute | ~4–5 hours on AMD Radeon RX 7800 XT (16 GB) via PyTorch ROCm |

Results, weights, and TensorBoard logs were saved in versioned directories so the
initial run remains reproducible alongside later experiments.

---

## 6. Initial Results

### 6.1 GIN dominates from the start

Across 16 classification datasets and the LD50 regression task, **GIN wins 13 of 16**
classification datasets by ROC-AUC and is the best regressor on LD50. Its
sum-aggregation with learnable ε is provably more expressive than GCN's normalised
mean, and the extra MLP inside each GINConv lets it capture interactions between atom
features that the simpler architectures cannot.

| Model | Tox21 macro-AUROC | Win count (16 datasets) | LD50 R² |
|---|---|---|---|
| GCN | 0.7049 | 1 | 0.260 |
| GIN | **0.7531** | **13** | **0.430** |
| GATGraph | 0.7212 | 1 | 0.206 |
| GraphSAGEModel | 0.7343 | 1 | 0.236 |

### 6.2 LD50 regression is weak across all models

The best R² is 0.430 (GIN), meaning **57% of the variance in acute toxicity remains
unexplained**. The other three models score below 0.27. This was the first signal that
the model lacked information rather than capacity to fit a noisy target — LD50 depends
on lipophilicity, polarity, and metabolism, all of which require descriptors beyond
the 4 atom features in use.

### 6.3 The class-imbalance failure mode (F1 collapse)

The most striking finding is what happened on imbalanced classification datasets. At
the default 0.5 decision threshold, GCN and GAT collapse to predicting only the
majority class on most Tox21 endpoints and on hERG_Karim, producing ROC-AUC ≈ 0.5 and
F1 = 0 simultaneously.

Mean F1 across the 16 classification datasets:

| Model | Mean F1 |
|---|---|
| GCN | 0.108 |
| GIN | 0.307 |
| GATGraph | 0.122 |
| GraphSAGEModel | 0.187 |

The starkest individual case is **hERG_Karim**: same dataset, same features.

| Model | ROC-AUC | F1 |
|---|---|---|
| GCN | 0.500 | 0.000 |
| GAT | 0.500 | 0.000 |
| GIN | **0.832** | **0.756** |
| GraphSAGE | 0.737 | 0.663 |

GIN survives where GCN and GAT collapse because its BatchNorm-stabilised MLP keeps
gradients flowing on skewed data. The other models converge to "always predict the
majority class" and never escape.

A second telling case is **ClinTox** (8% positive rate):

| Model | ROC-AUC | F1 |
|---|---|---|
| GCN | 0.598 | 0.000 |
| GIN | **0.922** | 0.000 |
| GATGraph | 0.602 | 0.000 |
| GraphSAGE | 0.804 | 0.000 |

GIN ranks toxic vs non-toxic molecules near-perfectly (AUROC 0.922) yet never crosses
the 0.5 sigmoid threshold to commit to a positive prediction. The model knows the
answer; the training procedure never asked it to act on that knowledge.

### 6.4 Threshold calibration as a quick fix

A natural first response to F1 = 0 is to **lower the decision threshold** to match
the dataset's positive-class ratio, with no retraining required. Setting threshold =
positive ratio (≈ 0.08 for ClinTox) recovers nonzero F1 on most affected datasets.

| ClinTox | F1 @ 0.5 | F1 @ calibrated threshold |
|---|---|---|
| GCN | 0.000 | 0.150 |
| GIN | 0.000 | 0.340 |
| GAT | 0.000 | 0.000 |
| GraphSAGE | 0.000 | 0.270 |

This confirms the initial models *do* learn meaningful signal — they were just being
asked to commit at a threshold that does not match the data's class prior. But
threshold calibration is a post-hoc patch: it changes the report, not the model. A
proper fix should make the training itself aware of the imbalance.

---

## 7. Diagnosis: where the initial pipeline fell short

Stepping back from the per-dataset numbers, three concrete limitations emerge:

**Limitation 1 — atoms are described too sparsely.** The 4-feature encoding tells the
GNN what *element* each atom is and *how many* bonds it makes, but says nothing about
hydrogen-bonding capacity, ring membership at the atom level, or 3D geometry. These
properties are first-class citizens in medicinal chemistry, and their absence is the
most plausible explanation for the low LD50 R².

**Limitation 2 — model capacity is a bottleneck.** With hidden dimension 16, each conv
layer maps 16 features to 16 features. The MLPs inside GIN have only 16 → 16 inner
projections. On the harder Tox21 endpoints (SR-ATAD5, NR-Aromatase) ROC-AUC stays in
the 0.65–0.72 band — competitive but visibly bounded. Bigger models could push this.

**Limitation 3 — the loss does not see the imbalance.** Plain
`BCEWithLogitsLoss` weights every sample equally. On a dataset with 5% positive rate,
predicting "non-toxic" for everything gives near-minimum loss; the model has no
incentive to find positives. Threshold calibration patches this at evaluation time
but the trained representations themselves are still biased toward the majority class.

A fourth, smaller observation: with the planned increase in capacity, learning rate
0.01 risks overshooting on a more complex loss landscape.

---

## 8. Upgrades

We address the three diagnosed limitations directly:

### 8.1 Richer atom features (4 → 9)

The headline upgrade. Each atom now carries five additional descriptors picked to
capture exactly the chemistry the initial encoding was missing.

| # | Feature | Why it matters |
|---|---|---|
| 1 | Atomic number | (kept) Element identity |
| 2 | Degree | (kept) Bond count |
| 3 | Formal charge | (kept) Ionisation state |
| 4 | Aromaticity | (kept) Aromatic ring membership |
| 5 | **Total Hs attached** | Hydrogen bonding — donor/acceptor capacity |
| 6 | **In ring** | Ring membership at the atom level (key for toxicity) |
| 7 | **Hybridisation** | SP / SP2 / SP3 — encodes local 3D geometry |
| 8 | **Total valence** | Includes contribution from double / triple bonds |
| 9 | **Radical electrons** | Reactive species detection |

The change is a single-cell modification in `smiles_to_graph`. The downstream
`in_channels` argument is read off the data automatically, so no model code needs
editing — the GNNs simply receive 9-dim atom vectors instead of 4-dim.

### 8.2 Larger hidden dimension (16 → 64)

The hidden dimension is bumped 4× — a modest absolute increase (≈60K extra parameters
per model) that directly addresses Limitation 2. With 9 atom features in and 64 hidden
units, each conv layer now has substantially more room to learn non-trivial atom
combinations.

### 8.3 Class-weighted loss

For each classification dataset, we compute a per-dataset positive class weight from
the training split:

```python
n_pos = (train_labels == 1).sum()
n_neg = (train_labels == 0).sum()
pos_weight = n_neg / n_pos          # ratio of negatives to positives
criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
```

For Tox21_SR-ATAD5 (~4% positive) this gives `pos_weight ≈ 24`, telling the loss
function that a missed positive is 24 times more costly than a false alarm. The
gradient signal during training is now strongly biased toward finding positives —
addressing the imbalance *during training* rather than patching at evaluation time.

### 8.4 Lower learning rate (0.01 → 0.001)

With a bigger model and a class-aware loss landscape, lr 0.01 risks overshooting.
0.001 gives smoother convergence; the existing ReduceLROnPlateau scheduler then
adjusts further if validation loss stalls.

### 8.5 What did NOT change

To keep the comparison clean and isolate the effect of the upgrades:
- Same four architectures
- Same number of conv layers (2)
- Same optimiser (Adam, weight_decay 5e-4)
- Same LR scheduler
- Same early-stopping patience
- Same datasets, same train/valid/test splits
- Same random seed (42)

---

## 9. Results After Upgrades

### 9.1 Tox21 macro-AUROC

The standard summary metric for the Tox21 benchmark. All architectures except
GraphSAGE improve.

| Model | Before | After | Δ |
|---|---|---|---|
| GCN | 0.7049 | 0.7315 | **+0.027** |
| GIN | 0.7531 | **0.7810** | **+0.028** |
| GATGraph | 0.7212 | 0.7476 | **+0.026** |
| GraphSAGEModel | 0.7343 | 0.7096 | **−0.025** |

### 9.2 LD50 regression — the largest single improvement

R² roughly *doubles* for GAT, and improves substantially for all four. This is where
the richer atom features pay off most: LD50 depends on subtle physico-chemical
properties (lipophilicity, polarity) that need geometric and hybridisation
information to estimate.

| Model | Before R² | After R² | Δ R² | Before RMSE | After RMSE |
|---|---|---|---|---|---|
| GCN | 0.260 | 0.335 | +0.075 | 0.813 | 0.771 |
| GIN | 0.431 | **0.491** | +0.060 | 0.713 | 0.674 |
| GATGraph | 0.206 | **0.396** | **+0.190** | 0.842 | 0.735 |
| GraphSAGEModel | 0.236 | 0.415 | +0.179 | 0.826 | 0.723 |

### 9.3 The class-imbalance fix

Mean F1 (at the default 0.5 threshold) roughly **doubles or triples** for the
architectures that had previously collapsed:

| Model | F1 before | F1 after | Gain |
|---|---|---|---|
| GCN | 0.108 | 0.279 | **+0.171** |
| GIN | 0.307 | 0.388 | +0.081 |
| GATGraph | 0.122 | 0.314 | **+0.192** |
| GraphSAGEModel | 0.187 | 0.314 | +0.127 |

The GAT gain (+0.19 F1) and the GCN gain (+0.17 F1) are direct evidence that those
architectures *can* predict the minority class — they just needed the loss function
to ask for it. **hERG_Karim** GCN goes from F1 = 0.000 to F1 = 0.279; the previously
collapsed model now produces real predictions.

A confirmatory observation: in this configuration, threshold calibration *hurts* mean
F1 by 0.06–0.10. That is the expected behaviour — the model is now calibrated through
the loss function, so a second post-hoc shift over-corrects. Before the upgrades, the
two corrections worked together; afterward they double-shift the decision boundary.

### 9.4 Best model per dataset

| Dataset | Before: best / AUROC | After: best / AUROC |
|---|---|---|
| AMES | GIN / 0.814 | GIN / **0.851** |
| ClinTox | GIN / **0.922** | GIN / 0.812 |
| DILI | GIN / **0.899** | GAT / 0.877 |
| hERG_Karim | GIN / 0.832 | GIN / **0.849** |
| Tox21_NR-AR | GCN / 0.767 | GCN / **0.785** |
| Tox21_NR-AR-LBD | GIN / 0.823 | GAT / **0.824** |
| Tox21_NR-AhR | GIN / 0.831 | GIN / **0.877** |
| Tox21_NR-Aromatase | GIN / 0.725 | GIN / **0.830** |
| Tox21_NR-ER | GIN / 0.679 | GIN / **0.707** |
| Tox21_NR-ER-LBD | GIN / **0.807** | GIN / 0.772 |
| Tox21_NR-PPAR-γ | GIN / 0.786 | GAT / **0.791** |
| Tox21_SR-ARE | GIN / 0.732 | GIN / **0.739** |
| Tox21_SR-ATAD5 | GIN / 0.695 | GIN / **0.801** |
| Tox21_SR-HSE | SAGE / 0.682 | GAT / **0.737** |
| Tox21_SR-MMP | GIN / 0.818 | GIN / **0.881** |
| Tox21_SR-p53 | GAT / 0.812 | GIN / 0.811 |

The post-upgrade pipeline wins on **11 of 16** datasets. Notable jumps: SR-ATAD5
(+0.106), NR-Aromatase (+0.105), SR-MMP (+0.063), NR-AhR (+0.046).

### 9.5 Win counts

| Model | Before | After |
|---|---|---|
| GIN | **13** | **11** |
| GATGraph | 1 | 4 |
| GCN | 1 | 1 |
| GraphSAGEModel | 1 | 0 |

GAT becomes more competitive — its attention mechanism has more signal to work with
once the atom features are richer. GIN remains the dominant choice.

### 9.6 GraphSAGE regression — honest reporting

Three datasets get *worse* after the upgrades, all GraphSAGE, all heavily imbalanced:

| Dataset | Before AUROC | After AUROC | Δ |
|---|---|---|---|
| ClinTox | 0.804 | 0.534 | **−0.270** |
| Tox21_NR-ER-LBD | 0.728 | 0.538 | **−0.190** |
| Tox21_NR-Aromatase | 0.668 | 0.498 | **−0.170** |

These are all 4–8% positive datasets. GraphSAGE's concatenate-then-mean aggregation
appears more sensitive to the aggressive `pos_weight` re-weighting than the others —
it likely becomes over-eager to predict positives early in training, and the gradient
signal pushes it away from useful representations. A milder pos_weight cap, or a
switch to focal loss, would likely recover this. We did not explore the fix in the
present study; it is the first item in the next-steps roadmap.

---

## 10. Discussion

### 10.1 Why GIN works best

GIN's `aggregate = (1+ε) · self + Σ neighbours` followed by an MLP is provably as
powerful as the Weisfeiler-Lehman (WL) graph isomorphism test, while GCN's
degree-normalisation maps structurally distinct graphs to the same embedding. For
molecular data — where small structural differences (a single ring vs. open chain, a
methyl group vs. ethyl) determine toxicity — this expressiveness gap matters.

GIN also includes BatchNorm in its message-passing MLP, which we observed makes it
markedly more stable on imbalanced datasets in the initial setup (it never fully
collapsed to majority-class prediction even with the unweighted loss).

### 10.2 The class-imbalance theme

Class imbalance is the dominant data-side issue in this study. Three independent
observations converge:

1. ClinTox initial run: ROC-AUC 0.922, F1 0.000. The model perfectly ranks toxic vs
   non-toxic but never crosses the threshold to commit to a positive.
2. hERG_Karim initial run: GCN and GAT collapse to AUROC = 0.5 / F1 = 0; GIN handles
   the same data with AUROC = 0.832, F1 = 0.756.
3. The negative trend line in the imbalance vs F1 scatter: as positive-class ratio
   approaches zero, even the best F1 across all four models drops to zero.

The class-weighted loss (§8.3) addresses this directly in the loss function: F1 nearly
*triples* for GCN and GAT after the upgrade. The post-upgrade Tox21 macro-AUROC
improvement (+0.027) is substantial but smaller than the F1 improvement, which is
expected — ROC-AUC is threshold-independent and was already mostly fine; F1 was the
metric that exposed the problem and shows the largest gain.

### 10.3 Why the LD50 ceiling persists

Even after upgrades, the best LD50 R² is 0.491 — meaning **51% of the variance in
acute toxicity remains unexplained**. Three structural reasons:

1. **No bond features.** We use `edge_index` only; bond type, stereochemistry, and
   ring membership at the edge level are not encoded. LD50 depends sensitively on
   these.
2. **No 3D geometry.** Atom hybridisation hints at it, but conformational ensembles
   are not represented.
3. **No metabolism awareness.** LD50 reflects how the body processes the drug, not
   just intrinsic chemistry.

Bond features are the cheapest of the three to add (see §11.4); 3D and metabolism
require fundamentally different inputs.

---

## 11. Limitations

1. **No bond features.** As noted in §10.3 — single largest information gap.
2. **Fixed model capacity per dataset.** Tiny datasets (DILI: 475 molecules) and
   large datasets (hERG_Karim: 13K) use the same hidden dimension. A capacity
   schedule would be defensible.
3. **Random splits, not scaffold splits.** TDC's default `get_split()` is random.
   Scaffold-based splitting gives a more honest measure of generalisation to truly
   novel molecules; standard for publication-quality molecular benchmarks.
4. **No hyperparameter search.** All training uses fixed lr, hidden dim, layer count.
5. **Single seed per experiment.** Each phase was run once; no confidence intervals
   over multiple seeds.

---

## 12. Conclusion

We presented a systematic comparison of four GNN architectures on a 17-dataset
toxicity benchmark. Two methodological observations stand out:

1. **GIN is the strongest baseline architecture for molecular toxicity prediction**,
   winning the Tox21 macro-AUROC and the majority of individual datasets in both the
   initial setup and the upgraded one.
2. **Class-weighted loss substantially outperforms post-hoc threshold calibration** as
   a fix for the F1-collapse problem on imbalanced data. The class weight acts during
   training and reshapes the learned representations themselves — calibration acts at
   evaluation time and only adjusts the cut-off.

The upgrades — adding 5 atom features, scaling hidden dimension from 16 to 64, and
adding a per-dataset class weight — improved Tox21 macro-AUROC by ~0.03, LD50 R² by
0.06 (GIN), and roughly tripled mean F1 for the architectures that had collapsed.

---

## 13. Deeper Research Directions

The findings above raise a number of questions that go beyond engineering tweaks.
This section frames each as a research-level inquiry — grounded in something we
observed — and outlines how we would investigate it. These are open-ended efforts
(months, not days), distinct from the engineering roadmap in §14.

### 13.1 Why does GIN dominate, and how universal is this?

**Observation.** GIN won 13 of 16 classification datasets initially, 11 of 16 after
upgrades, and led LD50 regression by a wide margin in both phases.

**The deeper question.** Is GIN's dominance a property of *molecular toxicity
prediction specifically*, of *molecular property prediction generally*, or of
*small-graph classification in any domain*? Tox21, AMES, and ClinTox are all built
on similar distributions of organic small molecules — the result might not transfer
to biomolecules, polymers, or material science graphs.

**How we would investigate.**
- Run the same comparison on graph datasets from other domains: OGB social-network
  benchmarks, OGB protein-function prediction, MoleculeNet's quantum-mechanical
  endpoints (QM9), polymer property prediction.
- Vary molecular-size distribution: are large drug-like molecules (MW > 500) vs.
  small fragments different stories?
- Compare GIN against more recent architectures (Graph Transformers, GraphGPS,
  higher-order GNNs like 3-WL, GINE+) — is GIN already saturating, or is it just
  ahead of its competitors at this scale?

**Why it matters.** A field-wide claim about which architecture to pick depends on
this transferability. The strongest version of our finding would be: "for molecular
property prediction up to ~50 atoms, GIN is the right baseline."

---

### 13.2 The mechanism of class-imbalance collapse

**Observation.** GCN and GAT collapse to AUROC = 0.5, F1 = 0 on hERG_Karim — even
though that dataset is 40% positive — while GIN handles the same data with
AUROC = 0.832. Same features, same optimiser, same seed.

**The deeper question.** What property of GIN makes it imbalance-resistant where the
other two fail? Three plausible mechanisms:
1. The BatchNorm layer inside each GINConv MLP keeps per-batch statistics flowing,
   preventing one-class dominance during training.
2. Sum-aggregation preserves more signal than degree-normalised mean — the rare
   positive class doesn't get averaged into oblivion.
3. The two-layer MLP inside GINConv has more parameters in the message function
   than GCN's single linear projection, giving more representational room before
   pooling.

**How we would investigate.**
- *Ablation:* train GIN without BatchNorm. If it collapses, BatchNorm is the cause.
- *Aggregation swap:* make GCN use sum instead of mean. Does this fix collapse?
- *Architecture interpolation:* build "GCN + BatchNorm" and "GIN minus the MLP".
  Compare collapse rates on hERG_Karim and ClinTox.
- *Latent-space geometry:* visualise pre-pooling node embeddings of positive vs.
  negative molecules (UMAP / t-SNE). Does GCN actually mix the classes in latent
  space, or merely fail to find a good linear separator?

**Why it matters.** This finding generalises. If BatchNorm is the cure, every GNN
should adopt it. If sum-aggregation is the cure, the field's preference for
mean-aggregation in GCN/GAT/SAGE is mis-aligned with imbalanced learning — a
substantive critique.

---

### 13.3 The information ceiling of 2D molecular graphs

**Observation.** Adding 5 atom features lifted Tox21 macro-AUROC by ~0.027 and LD50
R² by 0.06 — meaningful but bounded. After upgrades, LD50 R² is still only 0.49.

**The deeper question.** What is the theoretical maximum performance achievable from
a 2D molecular graph alone? Toxicity ultimately depends on:
- 3D conformation (which our graphs cannot represent at all)
- Electrostatic potential surfaces
- Dynamic behaviour (vibrational modes, rotamer populations)
- Metabolism (a function of the human body, not the molecule)

How much of toxicity is "knowable" from a 2D graph, vs. genuinely requiring 3D or
biological information?

**How we would investigate.**
- *Upper bound experiment:* add 3D features (RDKit `AllChem.EmbedMolecule`,
  conformer-averaged). Does R² jump?
- *Add electronic descriptors:* incorporate precomputed properties (logP, TPSA, MW,
  rotatable bonds). What's the gain?
- *Compare to fingerprint baselines:* Random Forest on Morgan fingerprints vs. our
  GNN. If the RF hits a similar ceiling, the data itself caps performance, not the
  model.
- *Estimate irreducible noise:* LD50 is a wet-lab measurement with 10–20% error
  intrinsically. Some of that R² ceiling is just experimental noise.

**Why it matters.** This frames future architectural work. If 2D graphs cap at
R² = 0.55, building bigger GNNs is wasted effort — we would need fundamentally
different inputs (3D, multi-modal, biological).

---

### 13.4 Endpoint-level difficulty in Tox21

**Observation.** ROC-AUC across the 12 Tox21 endpoints ranges from 0.71 (NR-ER) to
0.88 (SR-MMP) for the same models on the same molecules.

**The deeper question.** Why are some toxicity endpoints inherently easier to
predict from structure than others? Hypotheses:
- *Mechanism specificity:* nuclear-receptor endpoints (NR-AR, NR-ER) require
  specific receptor binding — predictable from local geometry.
- *Pathway integration:* stress-response endpoints (SR-ARE, SR-HSE) involve cellular
  responses with many possible upstream causes — harder to localise structurally.
- *Label noise:* some assays may simply be noisier.

**How we would investigate.**
- *Cross-endpoint correlation:* for each pair of endpoints, correlate per-molecule
  rankings produced by the same model. High correlation suggests shared mechanism;
  low correlation suggests independent biology.
- *Substructure attribution:* use GNNExplainer to identify which molecular
  fragments drive predictions for each endpoint. Are easy endpoints driven by
  sharper, more localised substructures?
- *Compare to literature:* Tox21 has been studied for ten years. Is our difficulty
  ranking consistent with reported state-of-the-art?

**Why it matters.** Understanding what makes an endpoint hard guides feature
engineering: easy endpoints need sharp local descriptors, hard endpoints need
global / pathway-level features (or different inputs entirely).

---

### 13.5 Interpretability — which substructures drive toxicity?

**Observation.** Our pipeline produces a single scalar prediction per molecule. We
have no insight into *why* the model thinks a given molecule is toxic.

**The deeper question.** Can we identify the substructures (functional groups, ring
systems) that drive each prediction, and do they match medicinal-chemistry intuition?

**How we would investigate.**
- *GNNExplainer* (Ying et al. 2019): identifies the minimal subgraph that preserves
  prediction. Apply to our trained GIN on each Tox21 endpoint; inspect top
  identified subgraphs.
- *Attention rollout* (for GAT): use the learned attention weights to assign
  per-edge importance scores; aggregate over a layer to get atom-level saliency.
- *Comparison to known toxicophores:* medicinal chemistry has catalogued canonical
  toxic substructures (nitroaromatics, alkyl halides, Michael acceptors). Does the
  GNN learn to flag these, or does it find different patterns?
- *Counterfactual perturbation:* modify one atom at a time in a predicted-toxic
  molecule. Which substitutions reduce predicted toxicity most? Cross-check with
  known SAR (structure-activity relationships).

**Why it matters.** This is the gap between "the model works" and "the model is
trustworthy". Drug-discovery pipelines cannot adopt a black-box predictor; they need
the chemist to see *why* a molecule was flagged. Interpretability is also a path to
discovering new chemistry: if the GNN learns toxicophores we did not know about,
that is a research output in itself.

---

### 13.6 Transfer learning across toxicity tasks

**Observation.** We trained 17 separate models on 17 separate datasets. Each task
saw only its own ~7,000 molecules. The total chemistry universe seen by any one
model is small.

**The deeper question.** How much of what GIN learns is generic "molecular
representation" versus task-specific? If we pretrain on one toxicity task and
fine-tune on another, do we beat from-scratch training?

**How we would investigate.**
- *Pretrain–finetune protocol:* pretrain GIN on AMES (largest balanced binary
  task), freeze conv layers, fine-tune only the linear head on each Tox21 endpoint.
  Compare to from-scratch.
- *Self-supervised pretraining:* mask atoms in unlabelled SMILES from PubChem and
  predict the masked element — the GNN equivalent of MolBERT / ChemBERTa. Then
  fine-tune on toxicity.
- *Multi-task baseline:* train one model on all 17 datasets simultaneously with
  17 output heads. Does shared representation help or hurt?

**Why it matters.** Real-world drug-discovery toxicity data is sparse — typical
project datasets have 200–500 molecules, not 7,000. Transfer learning is how the
field handles this in practice; we have not yet quantified its value on our setup.

---

## 14. Engineering Roadmap

The following work is organised by **expected impact / effort ratio**. Items are
roughly ordered as we'd recommend tackling them. Each entry includes the
implementation outline, expected gain, risks, and an effort estimate.

### Priority 1 — Quick wins (high impact / low effort)

#### 14.1 Cap or replace `pos_weight` to fix the GraphSAGE regression

**Problem.** The current setup uses `pos_weight = n_neg / n_pos`. For Tox21_SR-ATAD5
(4% positive) this gives `pos_weight ≈ 24` — aggressive enough to push GraphSAGE
into pathological behaviour on three imbalanced datasets (ClinTox, NR-ER-LBD,
NR-Aromatase: −0.17 to −0.27 ROC-AUC).

**What to do.** Either:
- Cap the weight: `pos_weight = min(n_neg / n_pos, K)` with K ∈ {5, 10, 15}.
- Switch to **focal loss**: `(1 − p_t)^γ · BCE` with γ ∈ {1, 2}. Focal loss down-weights
  *easy* examples instead of *negative* examples — mathematically more elegant for
  this problem.

**Code change.** ~10 lines in `train_one_model`. The focal loss implementation:
```python
def focal_bce(logits, targets, gamma=2.0):
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return ((1 - p_t) ** gamma * bce).mean()
```

**Expected gain.** Recover the 0.17–0.27 ROC-AUC drop on the affected datasets. May
also slightly help GIN.

**Effort.** Half a day including a fresh training run.

---

#### 14.2 Scaffold-based train/test splits

**Problem.** TDC's default `get_split()` uses random splits, which over-estimate
generalisation: structurally similar molecules end up in both train and test, so the
model effectively "memorises" the chemical scaffold. Real-world drug discovery needs
generalisation to *novel* scaffolds.

**What to do.** TDC supports it natively:
```python
data.get_split(method='scaffold', seed=42, frac=[0.7, 0.1, 0.2])
```

**Expected outcome.** All ROC-AUC and F1 numbers will drop — likely by 0.05–0.15 —
but the resulting numbers are far more honest. This is the standard split for
publication-quality molecular benchmarks (MoleculeNet, OGB).

**Code change.** Two-line change in the dataset configs cell. Re-runs the pipeline as
a new version.

**Effort.** Half a day including overnight retrain.

---

#### 14.3 Multiple seeds + confidence intervals

**Problem.** Each version is run once. We cannot tell whether the +0.028 Tox21
macro-AUROC gain after upgrades is robust or noise.

**What to do.** Run with seeds {42, 7, 123, 2024, 99}. Report mean ± std.

**Code change.** Wrap the outer training loop in a seeds loop and aggregate by
`(dataset, model, seed)`.

**Expected outcome.** Confidence intervals of typical width ±0.01–0.03 ROC-AUC.
Allows defensible claims like "the upgrades significantly improve over the baseline
(p < 0.05, paired t-test on per-dataset improvements)".

**Effort.** Trivial code change; 5× the compute (≈ 20–30 GPU-hours total).

---

### Priority 2 — Architectural improvements (medium impact / medium effort)

#### 14.4 Bond features as edge attributes

**Problem.** Currently `edge_index` only encodes bond *existence* — not type
(single/double/triple/aromatic), stereochemistry, ring membership at the edge level,
or conjugation. The 49% R² ceiling on LD50 strongly suggests this missing information
matters.

**What to do.**
1. Extend `smiles_to_graph` to compute bond features:
   ```python
   bond_features.append([
       bond.GetBondTypeAsDouble(),            # 1.0, 1.5 (aromatic), 2.0, 3.0
       int(bond.GetIsConjugated()),
       int(bond.IsInRing()),
       int(bond.GetStereo() != Chem.BondStereo.STEREONONE),
   ])
   data.edge_attr = torch.tensor(bond_features, dtype=torch.float)
   ```
2. Modify the four model `forward()` methods to use `edge_attr`:
   - **GAT** supports it natively: `GATConv(..., edge_dim=4)`.
   - **GIN** does not. Use `GINEConv` (the GIN-edge variant in PyG).
   - **GCN** and **GraphSAGE** do not. Either swap to `NNConv` / `GENConv`, or
     concatenate edge features into source-node features at preprocessing time.

**Expected gain.** LD50 R² up to 0.55–0.60. Tox21 macro-AUROC +0.01–0.02. Matches
published baselines on molecular benchmarks.

**Effort.** 1–2 days plus training.

---

#### 14.5 Multi-task learning for Tox21

**Problem.** Currently we train 12 separate models for the 12 Tox21 endpoints, each
on ~7,800 molecules with ~5% positive rate. The endpoints are biologically correlated
(NR-AR and NR-AR-LBD test the same receptor at different binding sites).

**What to do.** Train **one** GNN with 12 output heads, using the *same* molecules.
Tox21 stores all 12 labels per molecule (with NaN for "not assayed"). Use a masked
multi-output BCE loss:
```python
loss = (mask * F.binary_cross_entropy_with_logits(logits, targets, reduction='none')).sum() / mask.sum()
```

**Expected gain.** +0.02–0.05 macro-AUROC and faster training (4× fewer model
parameters to tune). Standard MoleculeNet practice.

**Effort.** 1 day. Requires changing the data loader to keep all 12 labels and a
small change to the model output dimension.

---

#### 14.6 Add a 3rd GNN layer + residual connections

**Problem.** With 2 layers and a typical molecular diameter of 8–15 atoms, the model
sees only a 2-hop neighbourhood. A 3rd layer would bring this to 3-hop, helping with
ring chemistry and substituent effects.

**What to do.** Add a 3rd `GCNConv` / `GINConv` etc. and add residual connections
(`x = x + conv(x)`) to prevent over-smoothing — a known issue when stacking GNNs.

**Expected gain.** Modest — perhaps +0.01–0.02 ROC-AUC. Higher gain if combined with
larger hidden dim.

**Effort.** A few hours. Risk of over-smoothing — must include residuals or layer norm.

---

#### 14.7 Hyperparameter search with Optuna

**Problem.** Every choice (lr, hidden dim, layer count, weight decay, dropout) is
fixed across all 17 datasets, despite huge variation in dataset size and difficulty.

**What to do.** Use [Optuna](https://optuna.org/) to search per-dataset:
```python
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: train_with_hp(trial), n_trials=50)
```
Search space:
- `hidden_dim` ∈ {32, 64, 128, 256}
- `lr` ∈ log-uniform [1e-4, 1e-2]
- `n_layers` ∈ {2, 3, 4}
- `dropout` ∈ [0, 0.5]
- `weight_decay` ∈ log-uniform [1e-5, 1e-3]
- `batch_size` ∈ {16, 32, 64}

Optimise on the validation ROC-AUC for each (dataset × model) pair.

**Expected gain.** +0.02–0.05 ROC-AUC across the board. The biggest gains will be
on small datasets (DILI, ClinTox) where the optimal capacity differs from large ones.

**Effort.** 2–3 days setup. Compute scales with `n_trials` × 17 × 4 models — about
40–60 GPU-hours for 50 trials each.

---

### Priority 3 — Major scope expansion (high impact / high effort)

#### 14.8 DAVIS / KIBA Drug-Target Interaction pipeline

**Problem.** This was a stated stretch goal. Currently the molecules' graph
representation is the only input — but predicting *which* protein the molecule binds
to (and how strongly) requires also encoding the protein sequence.

**What to do.** Build a joint drug-graph + protein-sequence model:
1. **Drug encoder:** Reuse the existing GIN backbone (it dominated both phases).
2. **Protein encoder:** Either
   - **Simple baseline:** 1D CNN over the amino-acid sequence with one-hot encoding.
   - **State of the art:** Use a pretrained protein language model embedding —
     [ESM-2](https://github.com/facebookresearch/esm) (Meta) gives a 1280-dim vector
     per protein. This is the field-standard approach.
3. **Joint model:** Concatenate drug embedding + protein embedding → MLP → scalar
   binding affinity (regression).
4. **Loss:** MSE on Kd (DAVIS) or KIBA score.

**Expected outcome.** Working DTI predictor with R² in the 0.65–0.75 range on DAVIS
(reported by recent papers using GIN + ESM-2). Significantly extends the project's
scope.

**Effort.** 1 week minimum. Requires substantial new pipeline code, potentially
downloading ESM-2 weights (~3 GB).

**Risk.** ESM-2 inference on long protein sequences is memory-heavy on a 16 GB GPU;
batch size will need careful tuning.

---

### Priority 4 — Polishing and reporting

#### 14.9 Cross-validation instead of single-split

**What to do.** 5-fold CV on each dataset; report mean ± std across folds.
TDC supports this with `get_split(method='cold_split', column_name='Drug_ID', frac=...)`
or manual `KFold` over the combined train+valid set.

**Expected outcome.** Tighter confidence on every reported number.

**Effort.** Low; 5× compute.

---

#### 14.10 Calibration analysis (expected calibration error)

**What to do.** For each model, compute the Expected Calibration Error (ECE) on
test-set predictions to quantify *how trustworthy* the predicted probabilities are.
A reliable model should have ECE < 0.05.

**Why it matters.** A drug-discovery downstream pipeline doesn't just need the
*ranking* of molecules — it needs to know "I am 90% confident this is toxic". A
high-AUROC, badly-calibrated model is dangerous.

**Effort.** Half a day. Use `sklearn.calibration.calibration_curve`.

---

### Recommended order

For a typical follow-up sprint:

1. **Day 1:** 14.1 (capped pos_weight) → run training overnight
2. **Day 2:** 14.2 (scaffold splits) → run training overnight
3. **Day 3:** 14.3 (multi-seed) on the best version → analyse results
4. **Week 2:** 14.4 (bond features) — likely the largest single quality boost
5. **Week 2–3:** 14.5 (multi-task Tox21)
6. **Week 3:** 14.7 (Optuna search) on the final architecture
7. **Stretch:** 14.8 (DTI pipeline) — separate project-scale effort

Following items 1–7 in order should bring the project to publication-quality
benchmarks for molecular property prediction (Tox21 macro-AUROC > 0.82,
LD50 R² > 0.55), comparable to recent papers in the field.

---

## Appendix A — Reproducibility

All code, model weights, CSVs, and plots are in the project repository:

```
Project-KGX/
├── notebooks/data_exploration.ipynb     # main notebook (sections 1–7)
├── docs/
│   ├── experiment_progression.md        # change log
│   └── technical_report.md              # this file
├── data/                                # cached TDC datasets
├── models/v1/, models/v2/               # saved .pt weights
├── results/v1/, results/v2/             # CSVs and TensorBoard logs
└── plots/, plots/v2/                    # generated figures
```

Each phase of the experiment is stored in its own versioned directory tree so the
full progression is reproducible. Random seed is fixed at 42; runs are reproducible
to within floating-point determinism on the same hardware (AMD ROCm).

## Appendix B — Software stack

- Python 3.12, PyTorch (ROCm 7.2), PyTorch Geometric
- RDKit for SMILES parsing and atom feature extraction
- Therapeutics Data Commons (TDC) for dataset loading
- scikit-learn for evaluation metrics
- pandas, matplotlib for analysis and plotting
- TensorBoard for live training-curve monitoring
