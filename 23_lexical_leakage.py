"""
Step 23: is the residual semantic or lexical?

Section S2 reports that 99.72% of ChEMBL E/Z pairs have bit-identical ChemNet
activations, and Section S4 reports an inversion effect whose CI excludes zero.
Both are attributed to canonical reordering rather than to stereochemical
sensitivity. That attribution is currently an argument, not a measurement.

This makes it a measurement. For each molecule we produce the E and the Z form
in two ways:

  canonical      Chem.MolToSmiles(mol)                  -- atom order may change
  order-fixed    Chem.MolToSmiles(mol, canonical=False) -- input atom order kept

Under the order-fixed writer the two strings differ *only* where a bond
direction character appears, and both '/' and '\\' tokenize to the unknown
token. If ChemNet has no representation of E/Z direction, the order-fixed pairs
must be identical at a rate of exactly 100%, while the canonical pairs retain
the small residual. Any order-fixed pair that differs would falsify the
attribution and mean ChemNet does carry some direction information.
"""
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_predictions, load_ref_model

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
STEREO = HERE / "chembl_stereo.smi"
OUT = HERE / "lexical_leakage.json"

N_MOL = 4000
SEED = 0

SS = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
      Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def forms(smi, canonical):
    """Return (original, inverted) SMILES written with the same writer."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bonds = [b for b in mol.GetBonds() if b.GetStereo() in SS]
    if not bonds:
        return None
    a = Chem.MolToSmiles(mol, canonical=canonical)
    for b in bonds:
        b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    except Exception:
        return None
    b_ = Chem.MolToSmiles(mol, canonical=canonical)
    return a, b_


def strip_direction(s):
    return s.replace("/", "").replace("\\", "")


def main():
    import random
    rng = random.Random(SEED)
    smis = [x.strip() for x in STEREO.read_text(encoding="utf-8").splitlines() if x.strip()]
    rng.shuffle(smis)

    pairs = {"canonical": [], "order_fixed": []}
    kept = []
    for s in smis:
        if len(kept) >= N_MOL:
            break
        c = forms(s, True)
        o = forms(s, False)
        if c is None or o is None:
            continue
        if len(c[0]) > 330 or len(o[0]) > 330:
            continue
        kept.append(s)
        pairs["canonical"].append(c)
        pairs["order_fixed"].append(o)

    n = len(kept)
    print(f"molecules: {n}")

    model = load_ref_model()
    res = {"n": n}

    for mode in ("canonical", "order_fixed"):
        P = pairs[mode]
        # how often does the writer change anything other than direction chars?
        same_skeleton_string = sum(
            1 for a, b in P if strip_direction(a) == strip_direction(b))
        A = get_predictions(model, [a for a, _ in P], n_jobs=0)
        B = get_predictions(model, [b for _, b in P], n_jobs=0)
        d = np.abs(A - B).max(axis=1)
        identical = int((d == 0).sum())
        print(f"\n{mode}")
        print(f"  strings differ only in '/' and '\\\\' : "
              f"{same_skeleton_string}/{n} = {100*same_skeleton_string/n:.2f}%")
        print(f"  activations bit-identical           : "
              f"{identical}/{n} = {100*identical/n:.2f}%")
        print(f"  max |delta| over all pairs          : {d.max():.3e}")
        res[mode] = {
            "direction_only_string_diff": same_skeleton_string,
            "identical_activations": identical,
            "frac_identical": identical / n,
            "max_abs_delta": float(d.max()),
            "mean_abs_delta": float(d.mean()),
        }

    print("\n" + "=" * 64)
    cf = res["canonical"]["frac_identical"]
    of = res["order_fixed"]["frac_identical"]
    print(f"  canonical writer   : {100*cf:.2f}% identical")
    print(f"  order-fixed writer : {100*of:.2f}% identical")
    if of == 1.0:
        print("  -> residual is entirely lexical (canonical reordering).")
        print("     ChemNet carries no E/Z direction information.")
    else:
        print("  -> order-fixed pairs also differ: the attribution to")
        print("     canonicalization is INCOMPLETE. Investigate.")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
