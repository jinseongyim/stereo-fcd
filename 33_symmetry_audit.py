"""
Step 33: explain why per-bond correct-after-inversion is 6/292, not 3/292.

Naive complementarity says: toggle every assigned E/Z label and the 289 correct
bonds become wrong while the 3 wrong bonds become correct -> 3 correct. We
observe 6. This script audits every bond and molecule that is correct after
inversion, printing:

  - gold CIP label, prediction CIP label as generated, prediction CIP label
    after inversion (all read with rdCIPLabeler, numbering-independent);
  - whether the WHOLE prediction is invariant to the simultaneous inversion
    under molecular symmetry, i.e. canonical isomeric SMILES(pred) ==
    canonical isomeric SMILES(inverted pred);
  - the count under two evaluation protocols:
      (A) FIXED mapping + literal label toggle (no re-canonicalization);
      (B) symmetry-aware: flip, re-canonicalize, re-derive CIP, re-match.

If protocol A gives 3 and protocol B gives 6, the 3 extra correct bonds live in
predictions that are symmetry-invariant to the toggle, and the manuscript must
describe the operation as "toggle every assigned label" rather than "give every
molecule the wrong geometry".
"""
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler, BondStereo

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
DB = Chem.BondType.DOUBLE
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


def invert_roundtrip(m):
    m2 = Chem.Mol(m)
    for b in m2.GetBonds():
        if b.GetStereo() in FLIP:
            b.SetStereo(FLIP[b.GetStereo()])
    Chem.AssignStereochemistry(m2, cleanIt=False, force=True)
    return Chem.MolFromSmiles(Chem.MolToSmiles(m2))


def cip_fixed_toggle(m, mapping):
    """Protocol A: read as-generated CIP labels in gold numbering, then literally
    toggle each label. No re-canonicalization."""
    lp = cip(m, mapping)
    return {k: ("Z" if v == "E" else "E") for k, v in lp.items()}


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]

    A_correct = B_correct = 0
    exact_inv = 0
    cases = []

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

        lp = cip(mp, mapping)
        mp_inv = invert_roundtrip(mp)
        map_i = skel(mg).GetSubstructMatch(skel(mp_inv))
        li = cip(mp_inv, map_i) if map_i else {}
        la = cip_fixed_toggle(mp, mapping)  # protocol A

        a_corr = sum(1 for k, v in lg.items() if la.get(k) == v)
        b_corr = sum(1 for k, v in lg.items() if li.get(k) == v)
        A_correct += a_corr
        B_correct += b_corr

        sym_invariant = (Chem.MolToSmiles(mp) == Chem.MolToSmiles(mp_inv))
        is_exact_inv = (Chem.MolToSmiles(mp_inv) ==
                        Chem.MolToSmiles(Chem.MolFromSmiles(r["gt"])))
        exact_inv += is_exact_inv

        if b_corr > 0 or is_exact_inv:
            cases.append({
                "cid": r["cid"], "b_corr": b_corr, "a_corr": a_corr,
                "sym_invariant": sym_invariant, "exact_inv": is_exact_inv,
                "gold": lg, "pred": lp, "pred_toggle_A": la, "pred_inv_B": li,
                "gt": Chem.MolToSmiles(mg),
            })

    print(f"protocol A (fixed mapping + literal toggle): {A_correct}/292 correct after inversion")
    print(f"protocol B (symmetry-aware re-canonicalize): {B_correct}/292 correct after inversion")
    print(f"whole-molecule exact matches after inversion: {exact_inv}/380\n")

    for c in sorted(cases, key=lambda x: (-x["exact_inv"], -x["b_corr"])):
        print(f"cid {c['cid']}  A={c['a_corr']} B={c['b_corr']} "
              f"sym_invariant={c['sym_invariant']} exact_inv={c['exact_inv']}")
        print(f"   gold        {c['gold']}")
        print(f"   pred(as-gen){c['pred']}")
        print(f"   toggle (A)  {c['pred_toggle_A']}")
        print(f"   inv    (B)  {c['pred_inv_B']}")
        print(f"   gt: {c['gt']}")


if __name__ == "__main__":
    main()
