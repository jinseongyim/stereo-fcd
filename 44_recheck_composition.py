"""Recompute Table S12's inverted column with the audited inversion helper.

Script 17 built that column with SetStereo + AssignStereochemistry, which step 34
documents as a no-op when both ends of the bond already carry directional
neighbours. The published column is therefore not an exact complement of the
as-is column: it reports E = 76 where a complete inversion must give E = 71.
This recomputes it with invert_correct and states the complement check.
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
BS = Chem.BondStereo
E, Z = BS.STEREOE, BS.STEREOZ
SS = {E, Z}                      # what composition() counts, as in script 17
# SetBondStereoFromDirections yields CIS/TRANS, not E/Z, so the audited helper
# must flip all four kinds -- omitting CIS/TRANS makes it a silent no-op
INV_SET = {E, Z, BS.STEREOCIS, BS.STEREOTRANS}
FLIP = {E: Z, Z: E, BS.STEREOCIS: BS.STEREOTRANS, BS.STEREOTRANS: BS.STEREOCIS}


def invert_correct(smi):
    """The audited inverter of step 34: flip the enum, then rebuild directions."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    Chem.SetBondStereoFromDirections(m)
    for b in m.GetBonds():
        if b.GetStereo() in INV_SET:
            b.SetStereo(FLIP[b.GetStereo()])
    Chem.SetDoubleBondNeighborDirections(m)
    return Chem.MolToSmiles(m)


def invert_old(smi):
    """What script 17 actually used."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for b in m.GetBonds():
        if b.GetStereo() in SS:
            b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(m, cleanIt=False, force=True)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def composition(smis):
    c, n_mol = Counter(), 0
    for s in smis:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        bs = [b for b in m.GetBonds() if b.GetStereo() in SS]
        if bs:
            n_mol += 1
        for b in bs:
            c[b.GetStereo()] += 1
    tot = c[E] + c[Z]
    return {"n_mol": n_mol, "E": c[E], "Z": c[Z], "total": tot,
            "E_share": c[E] / max(tot, 1)}


def main():
    d = json.loads((HERE / "molt5_outputs_molt5-large-caption2smiles.json")
                   .read_text(encoding="utf-8"))
    # same selection as script 17, so the comparison is like for like
    smis = [r["pred"] for r in d["records"] if r["valid"]]
    print(f"  {len(smis)} valid predictions of {len(d['records'])}")

    asis = composition(smis)
    new = composition([invert_correct(s) for s in smis])
    old = composition([invert_old(s) for s in smis])

    print(f"\n  {'':22s} {'n_mol':>6} {'E':>6} {'Z':>6} {'total':>7} {'E-share':>9}")
    for name, c in (("as generated", asis), ("inverted (audited)", new),
                    ("inverted (script 17)", old)):
        print(f"  {name:22s} {c['n_mol']:6d} {c['E']:6d} {c['Z']:6d} "
              f"{c['total']:7d} {c['E_share']:9.3f}")

    print(f"\n  exact complement requires  E = {asis['Z']}, Z = {asis['E']}, "
          f"E-share = {asis['Z']/max(asis['total'],1):.3f}")
    for name, c in (("audited", new), ("script 17", old)):
        ok = (c["E"], c["Z"]) == (asis["Z"], asis["E"])
        off = abs(c["E"] - asis["Z"])
        print(f"    {name:10s} {'EXACT COMPLEMENT' if ok else f'off by {off} bond(s)'}")

    # the argument the table supports: does inversion still move toward truth?
    truth = 0.430
    print(f"\n  distance to the ChEBI-20 truth E-share ({truth}):")
    print(f"    as generated       |{asis['E_share']:.3f} - {truth}| = "
          f"{abs(asis['E_share']-truth):.3f}")
    print(f"    inverted (audited) |{new['E_share']:.3f} - {truth}| = "
          f"{abs(new['E_share']-truth):.3f}")
    print("    the inverted set is still closer, so the table's argument stands"
          if abs(new["E_share"] - truth) < abs(asis["E_share"] - truth)
          else "    ARGUMENT NO LONGER HOLDS")

    (HERE / "composition_recheck.json").write_text(
        json.dumps({"as_generated": asis, "inverted_audited": new,
                    "inverted_script17": old}, indent=1), encoding="utf-8")
    print("\n  written composition_recheck.json")


if __name__ == "__main__":
    main()
