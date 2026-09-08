"""
Step 36: find an E/Z inversion that actually flips every stereogenic double bond.

A double bond that carries a CIP E/Z label is stereogenic, so inverting its
geometry MUST change the label -- unless the molecule has a genuine automorphism
that maps the bond to itself with swapped substituents (rare). Any method that
leaves many labelled bonds unchanged is buggy, not detecting symmetry.

We test candidate inversion methods on the 292 mapped bonds of MolT5-large's
eligible ChEBI-20 output and report, for each, how many bond CIP labels actually
change. The correct method changes ~all of them; the residue is the genuine
symmetry count.
"""
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler, BondStereo

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
DB = Chem.BondType.DOUBLE
SS = {BondStereo.STEREOE, BondStereo.STEREOZ,
      BondStereo.STEREOCIS, BondStereo.STEREOTRANS}
FLIP = {BondStereo.STEREOE: BondStereo.STEREOZ,
        BondStereo.STEREOZ: BondStereo.STEREOE,
        BondStereo.STEREOCIS: BondStereo.STEREOTRANS,
        BondStereo.STEREOTRANS: BondStereo.STEREOCIS}


def skel(m):
    m2 = Chem.Mol(m); Chem.RemoveStereochemistry(m2); return m2


def cip(m, mapping=None):
    m = Chem.Mol(m)
    try:
        rdCIPLabeler.AssignCIPLabels(m)
    except Exception:
        pass
    out = {}
    for b in m.GetBonds():
        if b.GetBondType() == DB and b.HasProp("_CIPCode"):
            c = b.GetProp("_CIPCode")
            if c in ("E", "Z"):
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if mapping is not None:
                    i, j = mapping[i], mapping[j]
                out[tuple(sorted((i, j)))] = c
    return out


def m_setstereo_cleardir(smi):
    """Flip STEREOE/Z enum, clear bond directions, let the writer recompute."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for b in m.GetBonds():
        if b.GetStereo() in FLIP:
            b.SetStereo(FLIP[b.GetStereo()])
    for b in m.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
    Chem.SetBondStereoFromDirections(m)  # no-op if no dirs; keeps enum
    return Chem.MolFromSmiles(Chem.MolToSmiles(m))


def m_swap_slashes(smi):
    """String-level: swap / <-> \\ for exactly one directional token per bond is
    hard lexically; swapping ALL preserves geometry. Included as a negative
    control (should change ~0 CIP labels)."""
    t = smi.replace("/", "\x00").replace("\\", "/").replace("\x00", "\\")
    return Chem.MolFromSmiles(t)


def m_setstereo_only(smi):
    """Flip enum in place, force assignment (step 32/33 method)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for b in m.GetBonds():
        if b.GetStereo() in FLIP:
            b.SetStereo(FLIP[b.GetStereo()])
    Chem.AssignStereochemistry(m, cleanIt=False, force=True)
    return Chem.MolFromSmiles(Chem.MolToSmiles(m))


def m_stereogroup_wedge(smi):
    """Use SetBondStereoFromDirections after flipping direction of one neighbour
    per stereo bond, but via the stereo-atom API rather than search."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.SetBondStereoFromDirections(m)
    for b in m.GetBonds():
        if b.GetStereo() in SS:
            b.SetStereo(FLIP[b.GetStereo()])
    # rebuild directions from the flipped stereo
    Chem.SetDoubleBondNeighborDirections(m)
    return Chem.MolFromSmiles(Chem.MolToSmiles(m))


METHODS = {
    "setstereo_only": m_setstereo_only,
    "setstereo_cleardir": m_setstereo_cleardir,
    "stereogroup_wedge": m_stereogroup_wedge,
    "swap_all_slashes(ctrl)": m_swap_slashes,
}


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]
    pairs = []
    for r in recs:
        if not r["valid"]:
            continue
        mg, mp = Chem.MolFromSmiles(r["gt"]), Chem.MolFromSmiles(r["pred"])
        if mg is None or mp is None:
            continue
        if Chem.MolToSmiles(skel(mg)) != Chem.MolToSmiles(skel(mp)):
            continue
        lg = cip(mg)
        if not lg:
            continue
        mapping = skel(mg).GetSubstructMatch(skel(mp))
        if not mapping:
            continue
        pairs.append((r, mp, mapping, lg))

    total_bonds = sum(len(lg) for _, _, _, lg in pairs)
    print(f"eligible molecules {len(pairs)}, mapped gold bonds {total_bonds}")
    for name, fn in METHODS.items():
        changed = unchanged = correct_after = 0
        for r, mp, mapping, lg in pairs:
            lp = cip(mp, mapping)
            mi = fn(r["pred"])
            if mi is None:
                continue
            map_i = skel(Chem.MolFromSmiles(r["gt"])).GetSubstructMatch(skel(mi))
            li = cip(mi, map_i) if map_i else {}
            for k in lp:  # bonds the prediction actually assigned
                if k in li:
                    if li[k] != lp[k]:
                        changed += 1
                    else:
                        unchanged += 1
            correct_after += sum(1 for k, v in lg.items() if li.get(k) == v)
        print(f"  {name:26s} flipped {changed:3d}  unchanged {unchanged:3d}  "
              f"correct_vs_gold_after {correct_after:3d}/292")


if __name__ == "__main__":
    main()
