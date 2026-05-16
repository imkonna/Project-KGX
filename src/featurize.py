"""Molecule-to-graph conversion utilities.

Converts SMILES strings into PyTorch Geometric Data objects with:
- Node features: atomic number, degree, formal charge, hybridization, aromaticity, etc.
- Edge features: bond type, conjugation, ring membership.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data

# ── Atom featurization ────────────────────────────────────────────────────────
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),       # Periodic table
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": [0, 1, 2, 3, 4],
}

ATOM_FEAT_DIM = 118 + 6 + 5 + 5 + 5 + 2  # = 141

# ── Bond featurization ────────────────────────────────────────────────────────
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_FEAT_DIM = 4 + 1 + 1  # = 6


# ── Helpers ───────────────────────────────────────────────────────────────────
def one_hot(value, allowable_set):
    """One-hot encode a value. Unknown values get all-zeros."""
    encoding = [0] * len(allowable_set)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    return encoding


def atom_features(atom):
    """Compute feature vector for a single atom."""
    return (
        one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
        + one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
        + [1 if atom.GetIsAromatic() else 0]
        + [1 if atom.IsInRing() else 0]
    )


def bond_features(bond):
    """Compute feature vector for a single bond."""
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [1 if bond.GetIsConjugated() else 0]
        + [1 if bond.IsInRing() else 0]
    )


# ── Graph construction ────────────────────────────────────────────────────────
def mol_to_graph(smiles, label):
    """Convert a SMILES string to a PyTorch Geometric Data object.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    label : int
        Binary label (0 or 1).

    Returns
    -------
    torch_geometric.data.Data or None
        Graph with node features (x), edge_index, edge_attr, and label (y).
        Returns None if SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float
    )

    # Edge index and features (undirected: add both directions)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if len(edge_index) == 0:
        # Single-atom molecule (rare) — self-loop
        edge_index = [[0, 0]]
        edge_attr = [[0] * BOND_FEAT_DIM]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_dataset(df, desc=""):
    """Convert a TDC dataframe into a list of PyG Data objects.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have columns 'Drug' (SMILES) and 'Y' (label).
    desc : str
        Description printed during conversion.

    Returns
    -------
    list[Data]
        List of PyG graph objects.
    """
    graphs = []
    skipped = 0
    for _, row in df.iterrows():
        data = mol_to_graph(row["Drug"], row["Y"])
        if data is not None:
            graphs.append(data)
        else:
            skipped += 1
    print(f"{desc}: {len(graphs):,} graphs built ({skipped} skipped)")
    return graphs
