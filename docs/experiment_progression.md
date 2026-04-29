# Experiment Progression & Design Decisions

## Overview

This document tracks every design decision, bug fix, and improvement made during the
development of the GNN-based drug toxicity prediction pipeline.
Results are versioned — each version's weights and CSVs are stored separately so the
full progression can be reproduced and compared.

---

## Datasets

| Dataset | Task | Size | Positive ratio | Notes |
|---|---|---|---|---|
| AMES | Binary classification | 7,255 | ~50% | DNA mutagenicity |
| hERG_Karim | Binary classification | 13,445 | ~40% | Cardiac arrhythmia risk |
| ClinTox | Binary classification | 1,484 | ~8% | Clinical trial toxicity failure |
| DILI | Binary classification | 475 | ~45% | Drug-induced liver injury |
| LD50_Zhu | Regression | 7,385 | — | Acute toxicity (log dose) |
| Tox21 (×12) | Binary classification | ~7,831 each | 3–15% | 12 nuclear receptor / stress-response endpoints |

**Note:** DAVIS and KIBA (drug-target interaction) are loaded but excluded from training —
they require a protein encoder and a joint drug+protein model, which is a separate pipeline.

---

## Model Architectures

All four models share the same template:
- 2 GNN convolutional layers
- Global pooling → linear head (1 output)
- No skip connections or layer normalisation (baseline design)

| Model | Aggregation | Key property |
|---|---|---|
| GCN | Degree-normalised mean | Simple, fast, well-understood |
| GIN | Learnable sum (ε-parameterised) | Theoretically as powerful as 1-WL test |
| GAT | Attention-weighted mean | Learns which neighbours matter |
| GraphSAGE | Concatenate + mean | Inductive; generalises to unseen nodes |

**Reference papers:**
- GCN: Kipf & Welling, ICLR 2017
- GIN: Xu et al., ICLR 2019
- GAT: Veličković et al., ICLR 2018
- GraphSAGE: Hamilton et al., NeurIPS 2017

---

## Version 1 — Baseline

**Saved to:** `models/v1/`, `results/v1/`

### Configuration

| Parameter | Value |
|---|---|
| Atom features | 4: atomic_num, degree, formal_charge, is_aromatic |
| Hidden dim | 16 |
| Layers | 2 |
| Optimizer | Adam, lr=0.01, weight_decay=5e-4 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Loss | BCEWithLogitsLoss (unweighted) / MSELoss |
| Threshold | Fixed 0.5 |
| Early stopping | patience=18–25 (per dataset) |
| Batch size | 16 (small datasets) / 32 (large) |
| Datasets | 17: 5 toxicity + 12 Tox21 endpoints |
| Models | 4: GCN, GIN, GAT, GraphSAGE |
| Total runs | 68 |

### Key Results

| Model | Tox21 macro-AUROC | Classification wins (16 datasets) | LD50 R² |
|---|---|---|---|
| GCN | 0.705 | 1 | 0.260 |
| GIN | **0.753** | **13** | **0.430** |
| GATGraph | 0.721 | 1 | 0.206 |
| GraphSAGEModel | 0.734 | 1 | 0.236 |

### Findings

1. **GIN dominates across all metrics.** Its sum-based aggregation with learnable ε is more
   expressive than GCN's normalised mean, which dilutes information from high-degree nodes.

2. **Class imbalance causes F1 collapse.** At threshold 0.5, GCN and GAT produce F1=0 on
   hERG_Karim, ClinTox, and most Tox21 endpoints. They converge to predicting only the
   majority class. GIN is more robust due to BatchNorm stabilising gradients on skewed data.

3. **LD50 regression is underfit.** Best R²=0.43 (GIN). With only 4 atom features and
   hidden_dim=16, the model lacks capacity to capture the 3D geometric and electronic
   properties that drive acute toxicity.

4. **ClinTox: AUROC 0.922 but F1=0.** The model correctly ranks molecules by toxicity
   risk (high AUROC) but never crosses the 0.5 sigmoid threshold to predict a positive.
   This is a threshold calibration problem, not a model capacity problem.

### Post-hoc Fix: Threshold Calibration

Without retraining, setting **threshold = positive-class ratio** (instead of 0.5)
recovers F1 on all datasets that had collapsed to zero.

Example gains (GIN):
- ClinTox: F1 0.000 → 0.340
- Tox21_NR-AhR: F1 0.187 → 0.450
- Tox21_SR-MMP: F1 0.316 → 0.480

This confirms the models *do* learn meaningful signal — they just needed a calibrated
decision boundary to match the class prior.

---

## Version 2 — Improved Features, Capacity & Loss

**Saved to:** `models/v2/`, `results/v2/`

### Changes from v1

#### 1. Richer Atom Features (4 → 9)

| Feature | New? | Rationale |
|---|---|---|
| atomic_num | — | Element identity |
| degree | — | Bond count |
| formal_charge | — | Ionisation state |
| is_aromatic | — | Ring aromaticity |
| **total_num_Hs** | ✓ | Hydrogen bonding capacity |
| **is_in_ring** | ✓ | Ring membership (critical for toxicity) |
| **hybridization** | ✓ | Local 3D geometry (SP2=planar, SP3=tetrahedral) |
| **total_valence** | ✓ | Total bonding capacity including double bonds |
| **radical_electrons** | ✓ | Reactive species detection |

The previous 4-feature encoding was largely blind to ring membership, hydrogen bonding,
and molecular geometry — all of which strongly correlate with toxicity.

#### 2. Hidden Dimension: 16 → 64

4× increase in model capacity. With hidden=16 the models were severely underfitting,
particularly on the regression task and on the harder Tox21 endpoints (SR-*).
Hidden=64 adds ~60K parameters per model — still lightweight but meaningfully more expressive.

#### 3. Class-Weighted Loss

```
pos_weight = n_negative / n_positive   (computed per dataset from training split)
```

Passed to `BCEWithLogitsLoss(pos_weight=pw)`. This makes the loss function penalise
missing a toxic molecule `pos_weight` times more than a false alarm — directly addressing
the majority-class collapse without any data augmentation or resampling.

For heavily imbalanced datasets (e.g. Tox21_SR-ATAD5: ~4% positive):
```
pos_weight ≈ 96/4 = 24
```
The model is now strongly incentivised to find the minority class during training,
rather than discovering at evaluation time that the threshold needs lowering.

#### 4. Learning Rate: 0.01 → 0.001

With a larger model (hidden=64) and class-weighted loss changing the loss landscape,
0.01 risks overshooting. 0.001 gives smoother convergence; ReduceLROnPlateau will
further reduce it if validation loss plateaus.

### Expected Improvements

| Metric | v1 | v2 (expected) |
|---|---|---|
| LD50 R² (GIN) | 0.430 | +0.05–0.10 (more features, more capacity) |
| Tox21 macro-AUROC (GIN) | 0.753 | +0.02–0.05 |
| F1 collapse (at threshold 0.5) | ~10 datasets zero | Significantly fewer |
| hERG_Karim GCN/GAT F1 | 0.000 | Should recover with class weighting |

---

## Metric Choices

| Metric | Used for | Why |
|---|---|---|
| ROC-AUC | Primary ranking metric | Threshold-independent; handles imbalance correctly |
| F1 | Secondary | Shows whether the model actually predicts the minority class |
| Accuracy | Reported but not used for comparison | Misleading under imbalance (90% accuracy = predict all-negative) |
| R² | LD50 regression | Fraction of variance explained; scale-independent |
| RMSE | LD50 regression | Same units as the target (log-dose) |

**Key insight for the report:** On Tox21 endpoints with 5% positive rate, a model that
always predicts "non-toxic" achieves 95% accuracy. ROC-AUC of 0.5 (random) would
correctly identify this model as useless — accuracy would not.

---

## Reproducibility

- Random seed fixed at 42 across `random`, `numpy`, `torch`, and `torch.cuda`
- `drop_last=True` on training DataLoader (prevents size-1 batches that crash GIN's BatchNorm)
- All model weights saved as `.pt` files per version
- All metrics saved as CSV per version
- TensorBoard logs saved per version for training curve inspection

---

## What Was Not Done (Future Work)

| Idea | Expected gain | Complexity |
|---|---|---|
| Bond features as edge attributes | Medium | Medium — requires modifying forward() in all 4 models |
| 3rd GNN layer | Small–Medium | Low |
| DAVIS/KIBA DTI pipeline | High | High — needs protein encoder (ESM/ProtBERT) |
| Hyperparameter search (Optuna) | Medium | Medium |
| Focal loss instead of pos_weight | Similar to pos_weight | Low |
| Graph-level augmentation (DropEdge) | Small | Low |
| Scaffold-based train/test split | Better generalisation estimate | Low (TDC supports it) |
