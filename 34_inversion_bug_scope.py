"""
Step 34: measure how often the SetStereo-based inversion used in 15_c2_full.py
is a no-op, and rebuild a CORRECT inversion for the FCD headline.

SetStereo(FLIP) followed by AssignStereochemistry silently fails to change a
double bond when the bond carries directional single bonds on both ends (e.g.
conjugated systems, ring-embedded C=C), so the "fully inverted" set used for the
headline actually leaves some bonds unchanged. This script:

  1. counts, over every generated E/Z-bearing molecule, how many bonds the old
     inversion actually flips (CIP label changes) vs leaves unchanged;
  2. builds a correct inversion by toggling one neighbouring bond direction per
     stereo double bond and iterating until the CIP label flips, and reports how
     many molecules become genuinely distinct isomers;
  3. re-embeds original vs correctly-inverted with ChemNet and recomputes the
     headline FCD, to confirm FCD stays blind even under a real inversion.
"""
import importlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BondStereo, rdCIPLabeler

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
from fcd.utils import calculate_frechet_distance

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
ev = importlib.import_module("14_eval_stereofcd")

STEREO_SET = {BondStereo.STEREOE, BondStereo.STEREOZ,
              BondStereo.STEREOCIS, BondStereo.STEREOTRANS}
FLIP = {BondStereo.STEREOE: BondStereo.STEREOZ, BondStereo.STEREOZ: BondStereo.STEREOE,
        BondStereo.STEREOCIS: BondStereo.STEREOTRANS, BondStereo.STEREOTRANS: BondStereo.STEREOCIS}
UP, DN = Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT


def cip_multiset(m):
    m = Chem.Mol(m)
    try:
        rdCIPLabeler.AssignCIPLabels(m)
    except Exception:
        pass
    labs = sorted(b.GetProp("_CIPCode") for b in m.GetBonds()
                  if b.GetBondType() == Chem.BondType.DOUBLE and b.HasProp("_CIPCode")
                  and b.GetProp("_CIPCode") in ("E", "Z"))
    return labs


def invert_old(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    for b in m.GetBonds():
        if b.GetStereo() in STEREO_SET:
            b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(m, cleanIt=False, force=True)
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


def invert_correct(smi):
    """Invert every stereo double bond by flipping its STEREOE/Z enum and then
    rebuilding the bond directions from the flipped stereo with
    SetDoubleBondNeighborDirections. Unlike SetStereo+AssignStereochemistry
    (which no-ops when both ends already carry directional neighbours) and unlike
    the earlier direction-search heuristic (which left ~12%% of bonds unflipped),
    this flips the CIP label on every stereogenic double bond; only genuine
    automorphism-invariant bonds, if any, survive. Verified on known cases
    (C/C=C/C E->Z, F/C=C/F E->Z) and by step 36."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    Chem.SetBondStereoFromDirections(m)
    for b in m.GetBonds():
        if b.GetStereo() in STEREO_SET:
            b.SetStereo(FLIP[b.GetStereo()])
    Chem.SetDoubleBondNeighborDirections(m)
    return Chem.MolToSmiles(m)


def _one_cip(m, bidx):
    mm = Chem.Mol(m)
    try:
        rdCIPLabeler.AssignCIPLabels(mm)
    except Exception:
        return None
    b = mm.GetBondWithIdx(bidx)
    return b.GetProp("_CIPCode") if b.HasProp("_CIPCode") else None


def main():
    d = json.loads(PRED.read_text(encoding="utf-8"))
    recs = d["records"] if "records" in d else d
    gen = [r["pred"] if isinstance(r, dict) else r for r in recs]

    old_bonds_flipped = old_bonds_total = 0
    correct_bonds_flipped = 0
    mol_changed_old = mol_changed_correct = 0
    n_ez = 0
    orig_smiles, inv_old_smiles, inv_correct_smiles = [], [], []

    for s in gen:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        cm0 = cip_multiset(m)
        if not cm0:
            continue
        n_ez += 1
        cs = Chem.MolToSmiles(m)
        so = invert_old(cs)
        sc = invert_correct(cs)
        mo, mc = Chem.MolFromSmiles(so), Chem.MolFromSmiles(sc)
        cmo = cip_multiset(mo) if mo else cm0
        cmc = cip_multiset(mc) if mc else cm0

        # count bonds whose CIP label changed
        old_bonds_total += len(cm0)
        old_bonds_flipped += sum(1 for a, b in zip(cm0, cmo) if a != b)
        correct_bonds_flipped += sum(1 for a, b in zip(cm0, cmc) if a != b)
        mol_changed_old += (Chem.MolToSmiles(mo) != cs) if mo else 0
        mol_changed_correct += (Chem.MolToSmiles(mc) != cs) if mc else 0

        orig_smiles.append(cs)
        inv_old_smiles.append(so)
        inv_correct_smiles.append(sc)

    print(f"E/Z-bearing generated molecules: {n_ez}")
    print(f"total stereo bonds: {old_bonds_total}")
    print(f"  old (SetStereo) inversion flipped CIP label on: {old_bonds_flipped} "
          f"({100*old_bonds_flipped/old_bonds_total:.1f}%)")
    print(f"  correct inversion flipped CIP label on:         {correct_bonds_flipped} "
          f"({100*correct_bonds_flipped/old_bonds_total:.1f}%)")
    print(f"molecules made distinct: old {mol_changed_old}, correct {mol_changed_correct} "
          f"(of {n_ez})")

    # ---- FCD: original vs correctly-inverted, against the same reference ----
    ref_path = HERE / "chebi20_test.txt"
    refs = [ln.split("\t")[1] for ln in ref_path.read_text(encoding="utf-8").splitlines()
            if "\t" in ln]
    refs = [s for s in refs if len(s) < 340]
    # keep orig/inverted aligned and within pad length
    pairs = [(o, c) for o, c in zip(orig_smiles, inv_correct_smiles)
             if len(o) < 340 and len(c) < 340]
    orig_f = [o for o, _ in pairs]
    inv_f = [c for _, c in pairs]
    sfcd = ev.StereoFCD()
    E_ref = sfcd.embed(refs, correction=False)
    E_o = sfcd.embed(orig_f, correction=False)
    E_c = sfcd.embed(inv_f, correction=False)

    def frechet(x, y):
        return calculate_frechet_distance(x.mean(0), np.cov(x.T), y.mean(0), np.cov(y.T))

    fcd_o = float(frechet(E_o, E_ref))
    fcd_c = float(frechet(E_c, E_ref))
    # embedding identity check
    max_abs = float(np.abs(E_o - E_c).max())
    print(f"\nFCD original vs reference          : {fcd_o:.6f}")
    print(f"FCD correctly-inverted vs reference: {fcd_c:.6f}")
    print(f"delta FCD (correct inversion)      : {abs(fcd_c-fcd_o):.6e}")
    print(f"max |embedding difference| orig vs correctly-inverted: {max_abs:.3e}")

    (HERE / "inversion_bug_scope.json").write_text(json.dumps({
        "n_ez": n_ez, "total_bonds": old_bonds_total,
        "old_flipped": old_bonds_flipped, "correct_flipped": correct_bonds_flipped,
        "mol_changed_old": mol_changed_old, "mol_changed_correct": mol_changed_correct,
        "fcd_orig": fcd_o, "fcd_correct_inv": fcd_c,
        "delta_fcd_correct": abs(fcd_c - fcd_o), "max_emb_diff": max_abs,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
