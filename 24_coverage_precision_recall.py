"""
Step 24: decompose the "under-specification" claim properly.

Section S6.3 currently reports 111/380 exact isomeric match against 282/285
per-bond correctness and reads the gap as under-specification. That inference is
not safe: exact isomeric match is a whole-molecule test, so it also fails on
tetrahedral errors and on E/Z bonds the model simply omitted, and the two
denominators (380 molecules, 285 bonds) measure different things.

This script reports the bond-level decomposition against a single denominator,

    N          E/Z bonds carried by the targets
    coverage   fraction of those the prediction assigns at all
    precision  of the assigned ones, fraction correct
    recall     of all N, fraction assigned AND correct

and a molecule-level cross-tabulation that separates omission from mis-assignment
and isolates how many exact-match failures are attributable to tetrahedral
stereochemistry rather than to E/Z at all.

Scope: predictions whose stereo-free skeleton matches the target and whose
target carries at least one E/Z bond -- the same subset as step 17.

CORRECTION (step 25). An earlier version keyed bond labels by atom index taken
from each molecule's own canonical SMILES. Target and prediction are different
molecules -- same skeleton, different stereochemistry -- and RDKit's canonical
ranking uses stereochemistry, so those two numberings do not always agree and a
few bonds were matched to the wrong partner. The tell was that per-bond correct
counts for a set and its full inversion summed to 288 rather than to the 285
comparable bonds. Labels are now compared through an explicit substructure
correspondence on the stereo-free skeleton, and complementarity holds exactly.
"""
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
OUT = HERE / "coverage_precision_recall.json"

E, Z = Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS
SS = {E, Z, CIS, TRANS}


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def skeleton(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def ez_labels(m, mapping=None):
    """E/Z label per stereo bond, keyed by atom-index pair.

    If `mapping` is given (prediction atom -> target atom), keys are translated
    into the target's numbering so that labels from two different molecules are
    comparable.
    """
    out = {}
    for b in m.GetBonds():
        if b.GetStereo() in SS:
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if mapping is not None:
                i, j = mapping[i], mapping[j]
            out[tuple(sorted((i, j)))] = "E" if b.GetStereo() in (E, TRANS) else "Z"
    return out


def skeleton_mol(m):
    m2 = Chem.Mol(m)
    Chem.RemoveStereochemistry(m2)
    return m2


def tet_labels(s):
    """R/S label per tetrahedral centre, keyed by atom index."""
    m = Chem.MolFromSmiles(s)
    if m is None:
        return {}
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return {a.GetIdx(): a.GetPropsAsDict().get("_CIPCode")
            for a in m.GetAtoms() if a.HasProp("_CIPCode")}


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]

    N = assigned = correct = 0
    mol_cat = Counter()
    exact_fail_cause = Counter()
    n_eligible = 0

    for r in recs:
        if not r["valid"]:
            continue
        gt, pred = r["gt"], r["pred"]
        sg, sp = skeleton(gt), skeleton(pred)
        if sg is None or sp is None or sg != sp:
            continue
        mg, mp = Chem.MolFromSmiles(gt), Chem.MolFromSmiles(pred)
        cg, cp = canon(gt), canon(pred)
        lg = ez_labels(mg)
        if not lg:
            continue
        mapping = skeleton_mol(mg).GetSubstructMatch(skeleton_mol(mp))
        if not mapping:
            continue
        n_eligible += 1
        lp = ez_labels(mp, mapping)

        n_i = len(lg)
        assigned_i = sum(1 for k in lg if k in lp)
        correct_i = sum(1 for k, v in lg.items() if lp.get(k) == v)
        N += n_i
        assigned += assigned_i
        correct += correct_i

        omitted = assigned_i < n_i
        wrong = assigned_i > correct_i
        mol_cat[("omission" if omitted else "-",
                 "wrong" if wrong else "-")] += 1

        # of molecules whose E/Z is fully right, why does exact match still fail?
        if not omitted and not wrong:
            if cp == cg:
                exact_fail_cause["exact match"] += 1
            elif tet_labels(cg) != tet_labels(cp):
                exact_fail_cause["tetrahedral differs"] += 1
            else:
                exact_fail_cause["other"] += 1

    print(f"eligible molecules (skeleton matches, target carries E/Z): {n_eligible}")
    print("\nbond level, single denominator N = target E/Z bonds")
    print(f"  N (target E/Z bonds)   : {N}")
    print(f"  assigned by prediction : {assigned:5d}   coverage  = {assigned/N:.4f}")
    print(f"  correct among assigned : {correct:5d}   precision = {correct/assigned:.4f}")
    print(f"  correct of all N       : {correct:5d}   recall    = {correct/N:.4f}")

    print("\nmolecule level (omission = some target bond unassigned;"
          " wrong = some assigned bond incorrect)")
    for k in sorted(mol_cat, key=lambda x: -mol_cat[x]):
        lab = {("-", "-"): "all E/Z present and correct",
               ("omission", "-"): "omission only",
               ("-", "wrong"): "mis-assignment only",
               ("omission", "wrong"): "both"}[k]
        print(f"  {lab:32s} {mol_cat[k]:4d}  ({100*mol_cat[k]/n_eligible:.1f}%)")

    print("\namong molecules whose E/Z is entirely correct, exact isomeric match:")
    tot = sum(exact_fail_cause.values())
    for k, v in exact_fail_cause.most_common():
        print(f"  {k:24s} {v:4d}  ({100*v/tot:.1f}%)")

    res = {
        "n_eligible_molecules": n_eligible,
        "bond_level": {"N_target_bonds": N, "assigned": assigned,
                       "correct": correct,
                       "coverage": assigned / N, "precision": correct / assigned,
                       "recall": correct / N},
        "molecule_level": {f"{a}|{b}": v for (a, b), v in mol_cat.items()},
        "exact_match_among_ez_correct": dict(exact_fail_cause),
    }
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
