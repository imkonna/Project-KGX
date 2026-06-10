"""Real molecular artwork for the poster, rendered from RDKit.

Genuine atom coordinates, not hand-drawn doodles. `conformer_3d` draws the real
ETKDG 3D conformer that SchNet consumes, with depth cues, to showcase the 3D arm
of the benchmark in the Benchmark Design panel.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

OUT = Path(__file__).parent / "assets"

ELEM = {
    "C": "#3C4A52",
    "N": "#2C6E8F",
    "O": "#C24A33",
    "S": "#C28A2B",
    "Cl": "#3E8E63",
    "F": "#3E8E63",
    "Br": "#9A5B2B",
    "P": "#B5722A",
    "default": "#5A6B73",
}


def _hex(c):
    return np.array([int(c[k:k + 2], 16) for k in (1, 3, 5)]) / 255


def _shade(col, f):
    """Darken toward the back (f=0 far, f=1 near)."""
    return tuple(_hex(col) * (0.58 + 0.42 * f))


def _lighten(col, amt):
    c = _hex(col)
    return tuple(c + (1 - c) * amt)


def conformer_3d(smiles="CC(=O)Nc1ccc(O)cc1", name="conformer_3d", figsize=6.6, spread=1.6):
    """Real 3D ETKDG conformer (the input SchNet sees), drawn with perspective depth.

    `spread` exaggerates the 3D perspective so atoms separate front-to-back.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    p = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                  for i in range(mol.GetNumAtoms())])
    p -= p.mean(0)
    # tilt to an angle that reveals the ring is not flat-on
    ax_, ay_ = 0.62, 1.0
    Rx = np.array([[1, 0, 0], [0, np.cos(ax_), -np.sin(ax_)], [0, np.sin(ax_), np.cos(ax_)]])
    Ry = np.array([[np.cos(ay_), 0, np.sin(ay_)], [0, 1, 0], [-np.sin(ay_), 0, np.cos(ay_)]])
    p = p @ Rx.T @ Ry.T
    span = np.ptp(p, axis=0).max() or 1.0
    p /= span
    z = p[:, 2]
    zn = (z - z.min()) / (np.ptp(z) or 1)  # 0 = far, 1 = near

    # perspective: bring the camera close so near atoms fan out and enlarge
    d = 2.2
    persp = d / (d - spread * z)
    px, py = p[:, 0] * persp, p[:, 1] * persp

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    bonds = sorted(mol.GetBonds(), key=lambda b: (z[b.GetBeginAtomIdx()] + z[b.GetEndAtomIdx()]) / 2)
    for b in bonds:
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        dd = (zn[i] + zn[j]) / 2
        ax.plot([px[i], px[j]], [py[i], py[j]], color="#7C8C95",
                lw=1.8 + 3.4 * dd, solid_capstyle="round", alpha=0.30 + 0.6 * dd, zorder=1 + dd)

    for i in np.argsort(z):
        sym = mol.GetAtomWithIdx(int(i)).GetSymbol()
        col = ELEM.get(sym, ELEM["default"])
        r = (0.04 + 0.045 * zn[i]) * persp[i]
        # drop shadow grounds the atom in space
        ax.add_patch(Circle((px[i] + 0.018, py[i] - 0.018), r * 0.96,
                            color="#222d34", alpha=0.10 + 0.12 * zn[i], zorder=2 + zn[i]))
        # sphere body, depth-shaded
        ax.add_patch(Circle((px[i], py[i]), r, facecolor=_shade(col, zn[i]),
                            edgecolor="white", linewidth=1.1, zorder=3 + zn[i]))
        # specular highlight -> reads as a 3D sphere
        ax.add_patch(Circle((px[i] - 0.32 * r, py[i] + 0.32 * r), 0.32 * r,
                            facecolor=_lighten(col, 0.7), edgecolor="none",
                            alpha=0.5, zorder=3.6 + zn[i]))

    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.2
    ax.set_xlim(px.min() - pad, px.max() + pad)
    ax.set_ylim(py.min() - pad, py.max() + pad)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"{name}.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    conformer_3d()
