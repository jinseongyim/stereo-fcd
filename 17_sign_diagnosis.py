"""
Step 17: why does inverting MolT5's stereochemistry IMPROVE Stereo-FCD?

C2 on the full test set gave a decisive but backwards result: inverting every
double bond moved Stereo-FCD from 0.5844 to 0.5617 (-17.5 sd), while the
ChEMBL corruption sweep had shown the metric rising with damage.

Two incompatible explanations:

  A  The metric is right and the model is wrong. MolT5's E/Z exact-match rate
     was 13%, well below the 50% a coin flip would give, so its stereochemistry
     may be systematically inverted relative to the truth. Then flipping it
     genuinely moves the distribution closer, and Stereo-FCD is correctly
     reporting "this model's stereochemistry is worse than chance".

  B  The metric is wrong. The learned correction encodes a direction unrelated
     to correctness, and the flipped set is closer in h-space for no meaningful
     reason.

The discriminator is per-molecule ground truth, which needs no metric at all:
restrict to predictions whose stereo-free skeleton matches the target, then ask
whether inverting RAISES the fraction of correct E/Z assignments.

  inverted accuracy > as-is accuracy  ->  A (metric vindicated)
  inverted accuracy <= as-is accuracy ->  B (metric must be discarded)
"""
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
OUT = HERE / "sign_diagnosis.json"

E, Z = Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS
SS = {E, Z, CIS, TRANS}
FLIP = {E: Z, Z: E, CIS: TRANS, TRANS: CIS}


def skeleton(smi):
    """canonical SMILES with all stereochemistry removed"""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def invert(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    bs = [b for b in m.GetBonds() if b.GetStereo() in SS]
    if not bs:
        return Chem.MolToSmiles(m)
    for b in bs:
        b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(m, cleanIt=False, force=True)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def bond_labels(smi):
    """E/Z label per stereo bond, keyed by the atom pair, order-independent."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {}
    out = {}
    for b in m.GetBonds():
        if b.GetStereo() in SS:
            key = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            out[key] = "E" if b.GetStereo() in (E, TRANS) else "Z"
    return out


def main():
    d = json.loads(PRED.read_text(encoding="utf-8"))
    recs = d["records"]
    print(f"records: {len(recs)}")

    # ---------- distributional E/Z composition ----------
    def composition(smis, label):
        c = Counter()
        nmol = 0
        for s in smis:
            lab = bond_labels(s)
            if lab:
                nmol += 1
                c.update(lab.values())
        tot = c["E"] + c["Z"]
        print(f"  {label:28s} molecules with E/Z {nmol:5d}   bonds E {c['E']:5d} "
              f"Z {c['Z']:5d}   E-share {c['E']/max(tot,1):.3f}")
        return {"n_mol": nmol, "E": c["E"], "Z": c["Z"],
                "E_share": c["E"] / max(tot, 1)}

    gts = [r["gt"] for r in recs]
    preds = [r["pred"] for r in recs if r["valid"]]
    inv_preds = [x for x in (invert(s) for s in preds) if x]

    print("\nE/Z composition")
    comp = {
        "reference": composition(gts, "ChEBI-20 test (truth)"),
        "molt5": composition(preds, "MolT5-large output"),
        "molt5_inverted": composition(inv_preds, "MolT5-large inverted"),
    }

    # ---------- the discriminator: per-molecule ground truth ----------
    n_same_skel = n_gt_stereo = 0
    exact_asis = exact_inv = 0
    bond_tot = bond_ok_asis = bond_ok_inv = 0

    for r in recs:
        if not r["valid"]:
            continue
        gt, pred = r["gt"], r["pred"]
        sg, sp = skeleton(gt), skeleton(pred)
        if sg is None or sp is None or sg != sp:
            continue                       # different molecule -- stereo is moot
        n_same_skel += 1

        lg = bond_labels(canon(gt))
        if not lg:
            continue                       # target carries no E/Z
        n_gt_stereo += 1

        cp = canon(pred)
        ci = invert(pred)
        if cp == canon(gt):
            exact_asis += 1
        if ci == canon(gt):
            exact_inv += 1

        lp, li = bond_labels(cp), bond_labels(ci) if ci else {}
        for k, v in lg.items():
            if k in lp:
                bond_tot += 1
                bond_ok_asis += (lp[k] == v)
                bond_ok_inv += (li.get(k) == v)

    print("\ndiscriminator: predictions whose skeleton matches the target")
    print(f"  same skeleton                 : {n_same_skel}")
    print(f"  ... and target carries E/Z    : {n_gt_stereo}")
    if n_gt_stereo:
        print(f"\n  exact isomeric match, as-is   : {exact_asis}/{n_gt_stereo} "
              f"= {100*exact_asis/n_gt_stereo:.2f}%")
        print(f"  exact isomeric match, inverted: {exact_inv}/{n_gt_stereo} "
              f"= {100*exact_inv/n_gt_stereo:.2f}%")
    if bond_tot:
        print(f"\n  per-bond E/Z correct, as-is   : {bond_ok_asis}/{bond_tot} "
              f"= {100*bond_ok_asis/bond_tot:.2f}%")
        print(f"  per-bond E/Z correct, inverted: {bond_ok_inv}/{bond_tot} "
              f"= {100*bond_ok_inv/bond_tot:.2f}%")

    verdict = "INCONCLUSIVE"
    if bond_tot:
        if bond_ok_inv > bond_ok_asis:
            verdict = ("A -- inverting genuinely improves correctness; the model's "
                       "stereochemistry is anti-correlated with truth and the metric "
                       "is reporting that correctly")
        else:
            verdict = ("B -- inverting does NOT improve correctness, so the metric's "
                       "preference for the inverted set is spurious")
    print(f"\nVERDICT: {verdict}")

    OUT.write_text(json.dumps({
        "composition": comp, "n_same_skeleton": n_same_skel,
        "n_gt_stereo": n_gt_stereo,
        "exact_asis": exact_asis, "exact_inverted": exact_inv,
        "bond_total": bond_tot, "bond_ok_asis": bond_ok_asis,
        "bond_ok_inverted": bond_ok_inv, "verdict": verdict,
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
