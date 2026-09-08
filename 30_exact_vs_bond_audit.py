"""
Step 30: resolve the 3-vs-4 discrepancy the reviewer flagged.

Section 4.3 reports, under full inversion of MolT5-large output:
  per-bond E/Z correct = 3/292
  exact isomeric match = 4/380

If a set and its full inversion together get 292 correct (289 + 3) and inversion
is an involution on E/Z, then at most 3 molecules can become exact matches after
inversion. Reporting 4 exact matches is therefore either an arithmetic error or
a sign that the whole-molecule "exact match" test and the per-bond matcher live
in different universes (e.g. canonical-SMILES string equality vs InChI, or a
molecule with no comparable bonds that matches for another reason).

This script recomputes both quantities on the inverted set with a single,
explicit definition and, for every molecule counted as an exact match after
inversion, prints its per-bond ledger so the 4th case (if real) is identified.
"""
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"

E, Z = Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS
SS = {E, Z, CIS, TRANS}
FLIP = {E: Z, Z: E, CIS: TRANS, TRANS: CIS}


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def skel_mol(m):
    m2 = Chem.Mol(m)
    Chem.RemoveStereochemistry(m2)
    return m2


def ez_labels(m, mapping=None):
    out = {}
    for b in m.GetBonds():
        if b.GetStereo() in SS:
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if mapping is not None:
                i, j = mapping[i], mapping[j]
            out[tuple(sorted((i, j)))] = "E" if b.GetStereo() in (E, TRANS) else "Z"
    return out


def invert_mol(m):
    m2 = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    for b in m2.GetBonds():
        if b.GetStereo() in SS:
            b.SetStereo(FLIP[b.GetStereo()])
    Chem.AssignStereochemistry(m2, cleanIt=False, force=True)
    return m2


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]

    n_elig = 0
    bond_tot = bond_ok_asis = bond_ok_inv = 0
    exact_asis_smiles = exact_inv_smiles = 0
    exact_asis_inchi = exact_inv_inchi = 0
    inv_exact_cases = []

    for r in recs:
        if not r["valid"]:
            continue
        mg, mp = Chem.MolFromSmiles(r["gt"]), Chem.MolFromSmiles(r["pred"])
        if mg is None or mp is None:
            continue
        if Chem.MolToSmiles(skel_mol(mg)) != Chem.MolToSmiles(skel_mol(mp)):
            continue
        lg = ez_labels(mg)
        if not lg:
            continue
        n_elig += 1

        mapping = skel_mol(mg).GetSubstructMatch(skel_mol(mp))
        if not mapping:
            continue
        lp = ez_labels(mp, mapping)
        mp_inv = invert_mol(mp)
        # recompute mapping for inverted (skeleton identical, but be safe)
        map_i = skel_mol(mg).GetSubstructMatch(skel_mol(mp_inv))
        li = ez_labels(mp_inv, map_i) if map_i else {}

        ok_asis = ok_inv = tot = 0
        for k, v in lg.items():
            if k in lp:
                tot += 1
                ok_asis += (lp[k] == v)
            if k in li:
                ok_inv += (li.get(k) == v)
        bond_tot += tot
        bond_ok_asis += ok_asis
        bond_ok_inv += sum(1 for k, v in lg.items() if li.get(k) == v)

        cg = canon(r["gt"])
        cp, ci = canon(r["pred"]), Chem.MolToSmiles(mp_inv)
        ig = Chem.MolToInchi(mg)
        ip, ii = Chem.MolToInchi(mp), Chem.MolToInchi(mp_inv)

        exact_asis_smiles += (cp == cg)
        exact_inv_smiles += (ci == cg)
        exact_asis_inchi += (ip == ig)
        exact_inv_inchi += (ii == ig)

        if ci == cg or ii == ig:
            inv_exact_cases.append({
                "cid": r["cid"],
                "gt": cg, "pred_inv": ci,
                "smiles_match": ci == cg, "inchi_match": ii == ig,
                "n_gold_bonds": len(lg),
                "gold": lg, "pred": lp, "pred_inv": li,
                "bonds_correct_inv": sum(1 for k, v in lg.items() if li.get(k) == v),
            })

    print(f"eligible molecules: {n_elig}")
    print(f"\nper-bond (comparable bonds = {bond_tot})")
    print(f"  correct as-is    : {bond_ok_asis}")
    print(f"  correct inverted : {bond_ok_inv}")
    print(f"\nexact match by SMILES : as-is {exact_asis_smiles}, inverted {exact_inv_smiles}")
    print(f"exact match by InChI  : as-is {exact_asis_inchi}, inverted {exact_inv_inchi}")

    print(f"\nmolecules that are exact matches AFTER inversion "
          f"(SMILES or InChI): {len(inv_exact_cases)}")
    for c in inv_exact_cases:
        print(f"\n  cid {c['cid']}  smiles_match={c['smiles_match']} "
              f"inchi_match={c['inchi_match']}  "
              f"gold_bonds={c['n_gold_bonds']} bonds_correct_inv={c['bonds_correct_inv']}")
        print(f"    gt       : {c['gt']}")
        print(f"    pred_inv : {c['pred_inv']}")
        print(f"    gold {c['gold']}")
        print(f"    pred {c['pred']}   pred_inv {c['pred_inv'] if isinstance(c['pred_inv'],str) else c['pred_inv']}")

    (HERE / "exact_vs_bond_audit.json").write_text(json.dumps({
        "eligible": n_elig, "bond_total": bond_tot,
        "bond_ok_asis": bond_ok_asis, "bond_ok_inv": bond_ok_inv,
        "exact_asis_smiles": exact_asis_smiles, "exact_inv_smiles": exact_inv_smiles,
        "exact_asis_inchi": exact_asis_inchi, "exact_inv_inchi": exact_inv_inchi,
        "inverted_exact_cases": inv_exact_cases,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
