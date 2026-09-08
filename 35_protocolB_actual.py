"""
Step 35: protocol B computed on the SAME molecules FCD scores.

The FCD inversion in Table 1 uses invert_correct (step 34): it toggles a
neighbouring bond direction and keeps the change only if the CIP label actually
flips. That is the actual molecular transformation the metric sees. To compare
"what FCD sees" against per-bond accuracy on the identical intervention, we must
re-perceive CIP labels on exactly those invert_correct molecules -- not on the
SetStereo round-trip used in step 33, which silently no-ops on double bonds that
carry directional neighbours on both ends and would inflate the surviving-correct
count with an artifact rather than genuine molecular symmetry.

This script reports, for MolT5-large's eligible ChEBI-20 output:
  - per-bond E/Z correct as generated and after invert_correct (protocol B, the
    actual transformation), read with rdCIPLabeler;
  - whole-molecule exact matches before and after;
  - for every bond that stays correct after inversion, whether it is genuinely
    symmetry-invariant (invert_correct could not flip its CIP label).
It overwrites cip_recount.json with these protocol-B numbers so Figure 1 and the
main text use the actual-molecule intervention.
"""
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler

import importlib
_m34 = importlib.import_module("34_inversion_bug_scope")
invert_correct = _m34.invert_correct

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
OUT = HERE / "cip_recount.json"
DB = Chem.BondType.DOUBLE


def skel(m):
    m2 = Chem.Mol(m)
    Chem.RemoveStereochemistry(m2)
    return m2


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


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]

    n_elig = 0
    N = assigned = correct = 0
    inv_correct = 0
    exact_asis = exact_inv = 0
    from collections import Counter
    mol_cat = Counter()
    sym_bonds = []

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
        n_elig += 1

        lp = cip(mp, mapping)
        # actual transformation, identical to the FCD inversion
        mp_inv = Chem.MolFromSmiles(invert_correct(r["pred"]))
        map_i = skel(mg).GetSubstructMatch(skel(mp_inv)) if mp_inv else None
        li = cip(mp_inv, map_i) if map_i else {}

        n_i = len(lg)
        a_i = sum(1 for k in lg if k in lp)
        c_i = sum(1 for k, v in lg.items() if lp.get(k) == v)
        N += n_i
        assigned += a_i
        correct += c_i
        for k, v in lg.items():
            if li.get(k) == v:
                inv_correct += 1
                # correct after inversion: was it flippable at all?
                sym_bonds.append({"cid": r["cid"], "bond": list(k),
                                  "gold": v, "as_gen": lp.get(k),
                                  "after_inv": li.get(k)})

        mol_cat[("omission" if a_i < n_i else "-",
                 "wrong" if a_i > c_i else "-")] += 1

        cg = canon(r["gt"])
        exact_asis += (canon(r["pred"]) == cg)
        exact_inv += (canon(Chem.MolToSmiles(mp_inv)) == cg if mp_inv else 0)

    print(f"eligible molecules: {n_elig}")
    print(f"per-bond E/Z (CIP), protocol B on the actual invert_correct molecules:")
    print(f"  as generated     = {correct}/{assigned} = {correct/assigned:.4f}")
    print(f"  after inversion  = {inv_correct}/{assigned} = {inv_correct/assigned:.4f}")
    print(f"whole-molecule exact match: as-gen {exact_asis}/{n_elig}, "
          f"inverted {exact_inv}/{n_elig}")
    print(f"\nbonds still correct after inversion ({len(sym_bonds)}):")
    for b in sym_bonds:
        print(f"  cid {b['cid']} bond {b['bond']} gold {b['gold']} "
              f"as-gen {b['as_gen']} after_inv {b['after_inv']}")

    OUT.write_text(json.dumps({
        "n_eligible": n_elig, "N_bonds": N, "assigned": assigned,
        "correct": correct, "correct_inverted": inv_correct,
        "coverage": assigned / N, "precision": correct / assigned,
        "recall": correct / N,
        "exact_asis": exact_asis, "exact_inverted": exact_inv,
        "protocol": "B (actual invert_correct molecules, CIP re-perceived)",
        "molecule_level": {f"{a}|{b}": v for (a, b), v in mol_cat.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
