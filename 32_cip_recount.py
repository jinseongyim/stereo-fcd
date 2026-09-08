"""
Step 32: recount per-bond E/Z with canonical CIP labels (symmetry-robust).

The earlier matcher read RDKit's STEREOE/STEREOZ, which is defined relative to a
bond's stereo-atom neighbours. Mapping atom indices between two differently
canonicalized molecules without remapping those stereo atoms can flip the E/Z
reading, so a handful of bonds were miscounted under molecular symmetry -- the
reason whole-molecule exact match (4 inverted) exceeded per-bond correct
(3 inverted), which is impossible if both are read consistently.

CIP labels (rdCIPLabeler) assign each double bond a canonical "E"/"Z" descriptor
independent of atom numbering, so they can be compared across molecules by the
skeleton substructure mapping without sign ambiguity. This recomputes coverage,
precision, recall and the inversion contrast with CIP labels, and checks that
per-bond correctness and whole-molecule exact match are now consistent.
"""
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
OUT = HERE / "cip_recount.json"

DB = Chem.BondType.DOUBLE


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def skel_mol(m):
    m2 = Chem.Mol(m)
    Chem.RemoveStereochemistry(m2)
    return m2


def cip_ez(m, mapping=None):
    """{atom-pair -> 'E'/'Z'} for double bonds with a CIP E/Z label.

    `m` must already have stereo perceived (i.e. come from MolFromSmiles). Keys
    are in the target's numbering when `mapping` (pred->target) is given.
    """
    m = Chem.Mol(m)
    try:
        rdCIPLabeler.AssignCIPLabels(m)
    except Exception:
        pass
    out = {}
    for b in m.GetBonds():
        if b.GetBondType() == DB and b.HasProp("_CIPCode"):
            code = b.GetProp("_CIPCode")
            if code in ("E", "Z"):
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if mapping is not None:
                    i, j = mapping[i], mapping[j]
                out[tuple(sorted((i, j)))] = code
    return out


def invert(m):
    """Flip every E/Z bond in place, then round-trip through SMILES so the
    inverted stereo is properly perceived for CIP labelling."""
    from rdkit.Chem import BondStereo
    FLIP = {BondStereo.STEREOE: BondStereo.STEREOZ,
            BondStereo.STEREOZ: BondStereo.STEREOE,
            BondStereo.STEREOCIS: BondStereo.STEREOTRANS,
            BondStereo.STEREOTRANS: BondStereo.STEREOCIS}
    m2 = Chem.Mol(m)
    for b in m2.GetBonds():
        if b.GetStereo() in FLIP:
            b.SetStereo(FLIP[b.GetStereo()])
    Chem.AssignStereochemistry(m2, cleanIt=False, force=True)
    return Chem.MolFromSmiles(Chem.MolToSmiles(m2))


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]

    N = assigned = correct = 0
    inv_correct = 0
    exact_asis = exact_inv = 0
    n_elig = 0
    mol_cat = Counter()

    for r in recs:
        if not r["valid"]:
            continue
        mg, mp = Chem.MolFromSmiles(r["gt"]), Chem.MolFromSmiles(r["pred"])
        if mg is None or mp is None:
            continue
        if Chem.MolToSmiles(skel_mol(mg)) != Chem.MolToSmiles(skel_mol(mp)):
            continue
        lg = cip_ez(mg)
        if not lg:
            continue
        mapping = skel_mol(mg).GetSubstructMatch(skel_mol(mp))
        if not mapping:
            continue
        n_elig += 1

        lp = cip_ez(mp, mapping)
        # inverted = toggle every assigned CIP label (protocol A); this is the
        # unambiguous "invert every E/Z assignment" and needs no molecule edit,
        # so it is immune to the SetStereo no-op documented in step 34.
        li = {k: ("Z" if v == "E" else "E") for k, v in lp.items()}

        n_i = len(lg)
        a_i = sum(1 for k in lg if k in lp)
        c_i = sum(1 for k, v in lg.items() if lp.get(k) == v)
        N += n_i; assigned += a_i; correct += c_i
        inv_correct += sum(1 for k, v in lg.items() if li.get(k) == v)

        mol_cat[("omission" if a_i < n_i else "-",
                 "wrong" if a_i > c_i else "-")] += 1

        cg = canon(r["gt"])
        exact_asis += (canon(r["pred"]) == cg)
        # whole-molecule exact match after toggling every assigned label:
        # all gold bonds present and correct once toggled
        exact_inv += (len(lp) >= n_i and all(li.get(k) == v for k, v in lg.items()))

    print(f"eligible molecules: {n_elig}")
    print(f"\nper-bond, CIP labels (comparable bonds N = {N})")
    print(f"  coverage  = {assigned}/{N} = {assigned/N:.4f}")
    print(f"  precision = {correct}/{assigned} = {correct/assigned:.4f}")
    print(f"  recall    = {correct}/{N} = {correct/N:.4f}")
    print(f"  correct inverted = {inv_correct}/{N}")
    print(f"\nwhole-molecule exact isomeric match (canonical SMILES)")
    print(f"  as generated = {exact_asis}/{n_elig}")
    print(f"  inverted     = {exact_inv}/{n_elig}")
    print(f"\nconsistency: correct_inverted ({inv_correct}) must be >= "
          f"exact_inverted ({exact_inv}): {inv_correct >= exact_inv}")

    print("\nmolecule level")
    for k in sorted(mol_cat, key=lambda x: -mol_cat[x]):
        lab = {("-", "-"): "all E/Z present and correct",
               ("omission", "-"): "omission only",
               ("-", "wrong"): "mis-assignment only",
               ("omission", "wrong"): "both"}[k]
        print(f"  {lab:32s} {mol_cat[k]:4d}")

    OUT.write_text(json.dumps({
        "n_eligible": n_elig, "N_bonds": N, "assigned": assigned,
        "correct": correct, "correct_inverted": inv_correct,
        "coverage": assigned / N, "precision": correct / assigned,
        "recall": correct / N,
        "exact_asis": exact_asis, "exact_inverted": exact_inv,
        "molecule_level": {f"{a}|{b}": v for (a, b), v in mol_cat.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
