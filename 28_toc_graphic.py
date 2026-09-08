"""
TOC / abstract graphic for the JCIM article.

ACS spec: 3.25 in wide x 1.75 in tall. One-glance message: two real, distinct
E/Z isomers (maleic vs fumaric acid) are chemically different (a bold "not
equal") yet collapse to the same ChemNet tokens, so FCD sees them as identical
(a bold "equal"). The two symbols carry the whole story.

Molecules rendered with RDKit, composed with matplotlib.
"""
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

HERE = Path(__file__).parent
INV = "#B01722"      # deeper red, stronger on white
INK = "#111111"      # near-black
GREY = "#3d3d3d"     # dark grey for secondary text/arrows (was #777/#888)


def mol_png(smi, px=420):
    m = Chem.MolFromSmiles(smi)
    d = rdMolDraw2D.MolDraw2DCairo(px, px)
    o = d.drawOptions()
    o.bondLineWidth = 4          # thicker bonds for legibility at 3.25 in
    o.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return mpimg.imread(BytesIO(d.GetDrawingText()), format="png")


fig = plt.figure(figsize=(3.25, 1.75), dpi=600)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# --- two isomers across the top, a bold not-equal between them ---
maleic = mol_png("OC(=O)/C=C\\C(=O)O")
fumaric = mol_png("OC(=O)/C=C/C(=O)O")

MOLY, MOLH, MOLW = 0.575, 0.41, 0.35
for img, x0, lab in [(maleic, 0.005, r"maleic ($Z$)"),
                     (fumaric, 0.645, r"fumaric ($E$)")]:
    axm = fig.add_axes([x0, MOLY, MOLW, MOLH]); axm.axis("off")
    axm.imshow(img)
    ax.text(x0 + MOLW / 2, 0.545, lab, ha="center", va="top",
            fontsize=9.5, color=INK, fontweight="bold")

ax.text(0.5, MOLY + MOLH / 2 + 0.02, r"$\neq$", ha="center", va="center",
        fontsize=22, color=INK, fontweight="bold")
ax.text(0.5, MOLY + MOLH / 2 - 0.155, "different\ncompounds", ha="center",
        va="center", fontsize=7.5, color=GREY, style="italic",
        linespacing=0.95)

# --- collapse arrows into the token box ---
for x in (0.19, 0.81):
    ax.add_patch(FancyArrowPatch((x, 0.455), (0.5, 0.345),
                 arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.8))

ax.add_patch(plt.Rectangle((0.145, 0.205), 0.71, 0.135, transform=ax.transAxes,
             facecolor="#ececec", edgecolor="#555", lw=1.4, zorder=2))
ax.text(0.5, 0.272, r"ChemNet:  $/$ , $\backslash$  $\rightarrow$  same token",
        ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold",
        zorder=3)

# --- verdict: the "equal" that mirrors the "not equal" above ---
ax.text(0.235, 0.075, r"$\Rightarrow$", ha="center", va="center",
        fontsize=13, color=GREY, fontweight="bold")
ax.text(0.53, 0.075, "FCD = 0", ha="center", va="center",
        fontsize=15, color=INV, fontweight="bold")
ax.text(0.83, 0.075, "(identical)", ha="center", va="center",
        fontsize=9, color=INV, fontweight="bold")

for ext in ("png", "pdf"):
    out = HERE / f"toc_graphic.{ext}"
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"wrote {out}")
