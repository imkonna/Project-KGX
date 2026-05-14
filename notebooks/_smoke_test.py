"""Smoke test for unified_benchmark.ipynb.

Runs RF/XGB/GIN/GINE/SchNet on DILI (smallest dataset, 475 mols) on CPU
to verify the full plumbing works end to end. Stops early if anything fails.
"""

import os, sys, json, copy, random, time, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINConv, GINEConv, global_add_pool

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

from tdc.single_pred import Tox
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cpu')
print(f'Device: {device}\n')

# -- Scaffold split --
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception: return None

def scaffold_split(df, smiles_col='Drug', frac_train=0.7, frac_val=0.1, frac_test=0.2, seed=42):
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaff = get_scaffold(row[smiles_col]) or ''
        scaffolds[scaff].append(idx)
    rng = np.random.RandomState(seed)
    groups = sorted(scaffolds.values(), key=lambda g: (-len(g), rng.random()))
    n = len(df); n_train = int(frac_train * n); n_val = int(frac_val * n)
    tr, va, te = [], [], []
    for g in groups:
        if len(tr) + len(g) <= n_train: tr.extend(g)
        elif len(va) + len(g) <= n_val: va.extend(g)
        else: te.extend(g)
    return {'train': df.loc[tr].reset_index(drop=True),
            'valid': df.loc[va].reset_index(drop=True),
            'test': df.loc[te].reset_index(drop=True)}

# -- Featurizers --
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32) if mol else None

def featurize_array(df, fn):
    X, y = [], []
    for _, row in df.iterrows():
        v = fn(row['Drug'])
        if v is None: continue
        X.append(v); y.append(row['Y'])
    return np.stack(X), np.array(y, dtype=np.float32)

def _atom_feats(a):
    return [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
            int(a.GetIsAromatic()), a.GetTotalNumHs(), int(a.IsInRing()),
            int(a.GetHybridization()), a.GetTotalValence(), a.GetNumRadicalElectrons()]

def _bond_feats(b):
    return [b.GetBondTypeAsDouble(), int(b.GetIsAromatic()),
            int(b.IsInRing()), int(b.GetStereo())]

def smiles_to_graph(smiles, with_bond=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([_atom_feats(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
        if with_bond:
            bf = _bond_feats(b); ea += [bf, bf]
    ei = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros((2, 0), dtype=torch.long)
    d = Data(x=x, edge_index=ei)
    if with_bond:
        d.edge_attr = torch.tensor(ea, dtype=torch.float) if ea else torch.zeros((0, 4), dtype=torch.float)
    return d

def build_graphs(df, with_bond):
    out = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row['Drug'], with_bond)
        if g is None or g.x.size(0) == 0: continue
        g.y = torch.tensor([row['Y']], dtype=torch.float)
        out.append(g)
    return out

# -- Models --
class GIN(nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        n1 = nn.Sequential(nn.Linear(in_ch, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
        n2 = nn.Sequential(nn.Linear(hid, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
        self.c1, self.c2 = GINConv(n1), GINConv(n2)
        self.lin = nn.Linear(hid, out_ch)
    def forward(self, x, ei, b, ea=None):
        x = F.relu(self.c1(x, ei)); x = F.relu(self.c2(x, ei))
        return self.lin(global_add_pool(x, b)).squeeze(-1)

class GINE(nn.Module):
    def __init__(self, in_ch, hid, out_ch, edge_dim=4):
        super().__init__()
        n1 = nn.Sequential(nn.Linear(in_ch, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
        n2 = nn.Sequential(nn.Linear(hid, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
        self.c1 = GINEConv(n1, edge_dim=edge_dim); self.c2 = GINEConv(n2, edge_dim=edge_dim)
        self.lin = nn.Linear(hid, out_ch)
    def forward(self, x, ei, b, ea=None):
        x = F.relu(self.c1(x, ei, ea)); x = F.relu(self.c2(x, ei, ea))
        return self.lin(global_add_pool(x, b)).squeeze(-1)

# -- Training --
def train_gnn(model_cls, split, with_bond, max_epochs=30, patience=10, bs=16):
    tr = build_graphs(split['train'], with_bond)
    va = build_graphs(split['valid'], with_bond)
    te = build_graphs(split['test'], with_bond)
    if not (tr and va and te): raise RuntimeError('empty split')
    in_ch = tr[0].num_node_features
    tl = PyGDataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    vl = PyGDataLoader(va, batch_size=bs)
    el = PyGDataLoader(te, batch_size=bs)
    model = model_cls(in_ch, 64, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    ys = torch.cat([d.y for d in tr]).view(-1)
    n_pos = max(int(ys.sum().item()), 1); n_neg = int(len(ys)) - n_pos
    pw = torch.tensor([n_neg / n_pos], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_val = float('inf'); best_state = copy.deepcopy(model.state_dict()); bad = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for b in tl:
            b = b.to(device); opt.zero_grad()
            ea = getattr(b, 'edge_attr', None) if with_bond else None
            out = model(b.x, b.edge_index, b.batch, ea)
            loss = crit(out, b.y.view(-1)); loss.backward(); opt.step()
        model.eval(); vloss = 0.0; n = 0
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                ea = getattr(b, 'edge_attr', None) if with_bond else None
                out = model(b.x, b.edge_index, b.batch, ea)
                vloss += crit(out, b.y.view(-1)).item() * b.num_graphs; n += b.num_graphs
        vloss /= max(n, 1)
        if vloss + 1e-4 < best_val: best_val = vloss; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else: bad += 1
        if bad >= patience: break
    model.load_state_dict(best_state)
    model.eval(); ys_t, ps = [], []
    with torch.no_grad():
        for b in el:
            b = b.to(device)
            ea = getattr(b, 'edge_attr', None) if with_bond else None
            out = model(b.x, b.edge_index, b.batch, ea)
            ys_t.append(b.y.view(-1).numpy()); ps.append(out.numpy())
    yt = np.concatenate(ys_t); pr = 1.0 / (1.0 + np.exp(-np.concatenate(ps)))
    yp = (pr >= 0.5).astype(int)
    return {'auroc': float(roc_auc_score(yt, pr)) if len(np.unique(yt)) > 1 else float('nan'),
            'f1': float(f1_score(yt, yp, zero_division=0))}

# -- Smoke test --
def run_test(dataset_name):
    print(f'=== {dataset_name} ===')
    t = time.time()
    data = Tox(name=dataset_name)
    df = data.get_data()
    print(f'  loaded: n={len(df)}  ({time.time()-t:.1f}s)')

    t = time.time()
    split = scaffold_split(df, seed=SEED)
    print(f'  scaffold-split: tr={len(split["train"])} va={len(split["valid"])} te={len(split["test"])}  ({time.time()-t:.1f}s)')

    results = {}

    # RF
    t = time.time()
    Xtr, ytr = featurize_array(split['train'], smiles_to_morgan)
    Xte, yte = featurize_array(split['test'], smiles_to_morgan)
    m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    yp = m.predict_proba(Xte)[:, 1]
    results['RF'] = {'auroc': float(roc_auc_score(yte, yp)) if len(np.unique(yte)) > 1 else float('nan'),
                     'f1': float(f1_score(yte, (yp >= 0.5).astype(int), zero_division=0))}
    print(f'  RF: AUROC={results["RF"]["auroc"]:.3f}  F1={results["RF"]["f1"]:.3f}  ({time.time()-t:.1f}s)')

    # XGB
    t = time.time()
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                      random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    yp = m.predict_proba(Xte)[:, 1]
    results['XGB'] = {'auroc': float(roc_auc_score(yte, yp)) if len(np.unique(yte)) > 1 else float('nan'),
                      'f1': float(f1_score(yte, (yp >= 0.5).astype(int), zero_division=0))}
    print(f'  XGB: AUROC={results["XGB"]["auroc"]:.3f}  F1={results["XGB"]["f1"]:.3f}  ({time.time()-t:.1f}s)')

    # GIN
    t = time.time()
    results['GIN'] = train_gnn(GIN, split, with_bond=False, max_epochs=30, patience=8)
    print(f'  GIN: AUROC={results["GIN"]["auroc"]:.3f}  F1={results["GIN"]["f1"]:.3f}  ({time.time()-t:.1f}s)')

    # GINE
    t = time.time()
    results['GINE'] = train_gnn(GINE, split, with_bond=True, max_epochs=30, patience=8)
    print(f'  GINE: AUROC={results["GINE"]["auroc"]:.3f}  F1={results["GINE"]["f1"]:.3f}  ({time.time()-t:.1f}s)')

    return results

if __name__ == '__main__':
    out_path = Path('results/smoke_test.json'); out_path.parent.mkdir(parents=True, exist_ok=True)
    all_r = {}
    for ds in ['DILI', 'ClinTox']:
        try:
            all_r[ds] = run_test(ds)
        except Exception as e:
            print(f'  FAILED: {e}')
            traceback.print_exc()
            all_r[ds] = {'error': str(e)}
        print()
    with open(out_path, 'w') as f: json.dump(all_r, f, indent=2)
    print(f'\nResults written to {out_path}')
