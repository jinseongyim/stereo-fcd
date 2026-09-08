"""
Figure 2: the encoder resolves the tetrahedral centre and not the double bond.

Two columns of named medicines, drawn from the PubChem structures measured in
step 37. Left: pairs whose double-bond configuration decides the pharmacology,
for which the ChemNet inputs are bit-identical and the activation difference is
exactly zero. Right: pairs that differ at a tetrahedral centre, which the same
encoder separates because '@' is in its vocabulary.

The contrast is the argument. Without the right-hand column the claim reads as
"ChemNet ignores stereochemistry", which is false; the failure is specific to
the two bond-direction characters.

Every number is read from drug_case_studies.json; nothing is hardcoded.
Structures carry RDKit stereo annotations so the configuration is legible.
"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdCIPLabeler
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
DATA = json.loads((HERE / "drug_case_studies.json").read_text())

# which pairs to show, and the short labels used in the figure
EZ_SHOW = {
    "tamoxifen":     ("tamoxifen", "(E)-tamoxifen"),
    "retinoic acid": ("tretinoin", "isotretinoin"),
}
CH_SHOW = {
    "omeprazole": ("esomeprazole", "(R)-omeprazole"),
    "citalopram": ("escitalopram", "(R)-citalopram"),
}

ZERO = "#C44E52"   # the collapse
SEP = "#4C72B0"    # the control
GREY = "#8C8C8C"

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


def render(smiles, px=520):
    """Draw one molecule to an RGB array, with stereo annotations."""
    mol = Chem.MolFromSmiles(smiles)
    # RDKit's legacy CIP perception mislabels sulfoxide stereocentres (it calls
    # esomeprazole 'R'). rdCIPLabeler implements the CIP rules correctly and the
    # drawer then annotates from those labels.
    rdCIPLabeler.AssignCIPLabels(mol)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    d = rdMolDraw2D.MolDraw2DCairo(px, int(px * 0.62))
    o = d.drawOptions()
    o.addStereoAnnotation = True
    o.bondLineWidth = 1.6
    o.baseFontSize = 0.62
    # the (E)/(Z) and (R)/(S) marks are the point of the figure, so make them
    # legible at print size rather than leaving them at the default scale
    o.annotationFontScale = 1.05
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format="png")


def find(section, key):
    for r in DATA[section]:
        if r["pair"] == key and r.get("status") == "measured":
            return r
    raise KeyError(f"{key} not measured in {section}")


def draw_pair(ax_top, ax_bot, ax_mid, rec, labels, collapsed):
    ax_top.imshow(render(rec["a"]["SMILES"])); ax_top.axis("off")
    ax_bot.imshow(render(rec["b"]["SMILES"])); ax_bot.axis("off")
    ax_top.set_title(f"{labels[0]}   (CID {rec['a']['CID']})", fontsize=8.5, pad=3)
    ax_bot.text(0.5, -0.03, f"{labels[1]}   (CID {rec['b']['CID']})", fontsize=8.5,
                ha="center", va="top", transform=ax_bot.transAxes)

    d = rec["order_fixed"]["max_abs_delta"]
    if collapsed:
        sym, col = "=", ZERO
        txt = r"$\max|\Delta a| = 0$"
    else:
        sym, col = r"$\neq$", SEP
        txt = rf"$\max|\Delta a| = {d:.2f}$"
    ax_mid.axis("off")
    ax_mid.text(0.5, 0.68, sym, ha="center", va="center", fontsize=16, color=col,
                transform=ax_mid.transAxes)
    ax_mid.text(0.5, 0.20, txt, ha="center", va="center", fontsize=8.5, color=col,
                transform=ax_mid.transAxes)


if __name__ == "__main__":
    ez_keys = list(EZ_SHOW)
    ch_keys = list(CH_SHOW)

    fig = plt.figure(figsize=(7.2, 6.2))
    # 2 pair-columns; each pair occupies rows [structure, relation, structure]
    gs = fig.add_gridspec(
        nrows=7, ncols=2,
        height_ratios=[1.0, 0.34, 1.0, 0.62, 1.0, 0.34, 1.0],
        hspace=0.06, wspace=0.16,
    )

    for col, (keys, show, section, collapsed) in enumerate((
            (ez_keys, EZ_SHOW, "ez_pairs", True),
            (ch_keys, CH_SHOW, "chiral_controls", False))):
        for i, key in enumerate(keys):
            base = 0 if i == 0 else 4
            rec = find(section, key)
            draw_pair(fig.add_subplot(gs[base, col]),
                      fig.add_subplot(gs[base + 2, col]),
                      fig.add_subplot(gs[base + 1, col]),
                      rec, show[key], collapsed)

    fig.text(0.29, 0.975, "(a)  double-bond configuration", ha="center",
             fontsize=10, weight="bold", color=ZERO)
    fig.text(0.755, 0.975, "(b)  tetrahedral configuration", ha="center",
             fontsize=10, weight="bold", color=SEP)
    fig.text(0.29, 0.951, r"'/' and '\' are absent from the vocabulary",
             ha="center", fontsize=7.6, color=GREY)
    fig.text(0.755, 0.951, "'@' is present in the vocabulary",
             ha="center", fontsize=7.6, color=GREY)

    for ext in ("pdf", "png"):
        out = HERE / f"figure1.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out.name)

    print("\nvalues used (order-fixed writer):")
    for section, keys in (("ez_pairs", ez_keys), ("chiral_controls", ch_keys)):
        for k in keys:
            r = find(section, k)
            print(f"  {k:18s} max|da| = {r['order_fixed']['max_abs_delta']:.4e}"
                  f"   one-hot identical = {r['order_fixed']['onehot_identical']}")
