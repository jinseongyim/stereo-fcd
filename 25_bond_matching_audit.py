"""
Step 25: audit the per-bond matching used in steps 17 and 24.

A referee observed an inconsistency. If 282 of 285 comparable bonds are correct
as generated, then inverting every assignment should leave exactly 285-282 = 3
correct. Step 17 reports 6. Under a clean involution that is impossible, so
either inversion is not an involution on these labels or the matching is wrong.

The suspect is the matching. Steps 17/24 key bond labels by atom index taken
from each molecule's *own* canonical SMILES:

    lg = labels(canon(gt))     indices in gt's canonical numbering
    lp = labels(canon(pred))   indices in pred's canonical numbering
    li = labels(canon(invert(pred)))   indices after re-canonicalization

RDKit's canonical ranking can take stereochemistry into account, so these three
numberings need not agree -- and Section S2 has already shown that inverting a
bond sometimes changes canonical atom order. Where that happens, `li.get(k)`
either misses or refers to a different bond.

This script recomputes everything with an explicit atom mapping and without
re-canonicalizing the inverted form:

  * map prediction atoms onto target atoms with a substructure match on the
    stereo-free skeleton, so labels are compared through a known correspondence;
  * build the inverted labels by flipping the prediction's own bond objects and
    reading them at the same indices, so inversion is an exact involution.

It reports the corrected counts and how many molecules were affected.
"""
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
OUT = HERE / "bond_matching_audit.json"

E, Z = Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS
SS = {E, Z, CIS, TRANS}
FLIP = {E: Z, Z: E, CIS: TRANS, TRANS: CIS}


def lab(b):
    return "E" if b.GetStereo() in (E, TRANS) else "Z"


def skeleton_mol(m):
    """Copy with all stereochemistry removed, for substructure matching."""
    m2 = Chem.Mol(m)
    Chem.RemoveStereochemistry(m2)
    return m2


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]

    n_elig = 0
    n_misaligned = 0
    tot = ok_asis = ok_inv = 0
    tot_old = ok_asis_old = ok_inv_old = 0

    for r in recs:
        if not r["valid"]:
            continue
        g = Chem.MolFromSmiles(r["gt"])
        p = Chem.MolFromSmiles(r["pred"])
        if g is None or p is None:
            continue
        sg, sp = skeleton_mol(g), skeleton_mol(p)
        if Chem.MolToSmiles(sg) != Chem.MolToSmiles(sp):
            continue
        gl = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): lab(b)
              for b in g.GetBonds() if b.GetStereo() in SS}
        if not gl:
            continue
        n_elig += 1

        # --- explicit correspondence: pred atom i  ->  gt atom match[i] ---
        match = sg.GetSubstructMatch(sp)          # tuple: pred idx -> gt idx
        if not match:
            continue

        # prediction labels, expressed in GT indices
        pl = {}
        for b in p.GetBonds():
            if b.GetStereo() in SS:
                k = tuple(sorted((match[b.GetBeginAtomIdx()],
                                  match[b.GetEndAtomIdx()])))
                pl[k] = lab(b)
        # inverted labels: flip the same bond objects, same indices -> exact involution
        il = {k: ("Z" if v == "E" else "E") for k, v in pl.items()}

        for k, v in gl.items():
            if k in pl:
                tot += 1
                ok_asis += (pl[k] == v)
                ok_inv += (il[k] == v)

        # --- reproduce the old index-by-own-canonical-SMILES matching ---
        def canon_labels(smi):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return {}
            return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): lab(b)
                    for b in m.GetBonds() if b.GetStereo() in SS}

        glo = canon_labels(Chem.MolToSmiles(g))
        plo = canon_labels(Chem.MolToSmiles(p))
        pinv = Chem.Mol(p)
        for b in pinv.GetBonds():
            if b.GetStereo() in SS:
                b.SetStereo(FLIP[b.GetStereo()])
        Chem.AssignStereochemistry(pinv, cleanIt=False, force=True)
        ilo = canon_labels(Chem.MolToSmiles(pinv))
        if set(plo.keys()) != set(ilo.keys()):
            n_misaligned += 1
        for k, v in glo.items():
            if k in plo:
                tot_old += 1
                ok_asis_old += (plo[k] == v)
                ok_inv_old += (ilo.get(k) == v)

    print(f"eligible molecules: {n_elig}")
    print(f"molecules whose canonical bond keys change under inversion: "
          f"{n_misaligned}  ({100*n_misaligned/max(n_elig,1):.1f}%)")

    print("\nold matching (atom indices from each molecule's own canonical SMILES)")
    print(f"  comparable bonds {tot_old}   as-is {ok_asis_old}   inverted {ok_inv_old}"
          f"   sum {ok_asis_old + ok_inv_old}")

    print("\ncorrected matching (explicit substructure correspondence, "
          "inversion applied in place)")
    print(f"  comparable bonds {tot}   as-is {ok_asis}   inverted {ok_inv}"
          f"   sum {ok_asis + ok_inv}")
    print(f"  complementarity holds: {ok_asis + ok_inv == tot}")
    if tot:
        print(f"  as-is    {100*ok_asis/tot:.2f}%")
        print(f"  inverted {100*ok_inv/tot:.2f}%")

    OUT.write_text(json.dumps({
        "n_eligible": n_elig,
        "n_canonical_key_change_under_inversion": n_misaligned,
        "old": {"bonds": tot_old, "as_is": ok_asis_old, "inverted": ok_inv_old},
        "corrected": {"bonds": tot, "as_is": ok_asis, "inverted": ok_inv,
                      "complementary": ok_asis + ok_inv == tot},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
