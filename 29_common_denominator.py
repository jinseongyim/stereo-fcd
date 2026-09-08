"""
Step 29: put the cross-model stereo comparison on common denominators.

Table 3 as first written reports coverage/recall over each model's own
skeleton-matched, E/Z-bearing predictions, so the denominators differ
(522/1041/1022) and the recall comparison is conditional on each model
recovering its own skeletons -- a selection effect could inflate the gap. This
script recomputes the comparison two ways with denominators that do NOT depend
on the model:

1. END-TO-END recall. Denominator = every E/Z bond carried by the 3,300 gold
   molecules (model-independent). A bond counts as recovered only if the
   prediction has the correct stereo-free skeleton AND assigns the correct
   configuration; a wrong skeleton, an omitted bond, or a wrong bond all count
   as failures. This is the honest "how often does the model get the whole
   thing right" number.

2. COMMON-SUPPORT recall. For a pair of models, restrict to the molecules where
   BOTH recovered the stereo-free skeleton and the target carries E/Z, and
   compute each model's recall over that shared bond set. Same denominator for
   both, so the comparison is direct.

Matching uses the corrected substructure correspondence of step 25.
"""
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "common_denominator.json"

MODELS = {
    "MolT5-base":  HERE / "molt5_outputs_molt5-base-caption2smiles.json",
    "MolT5-large": HERE / "molt5_outputs_molt5-large-caption2smiles.json",
    "BioT5-base":  HERE / "molt5_outputs_biot5-base-text2mol.json",
}

E, Z = Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS
SS = {E, Z, CIS, TRANS}


def skeleton_mol(m):
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


def per_caption(recs):
    """For each record, return (n_gold_ez, assigned, correct, skeleton_ok).

    n_gold_ez is the target's E/Z bond count (model-independent). assigned and
    correct are 0 unless the skeleton matches. Keyed by cid.
    """
    out = {}
    for r in recs:
        cid = r["cid"]
        mg = Chem.MolFromSmiles(r["gt"])
        if mg is None:
            continue
        lg = ez_labels(mg)
        n_gold = len(lg)
        rec = {"n_gold": n_gold, "assigned": 0, "correct": 0,
               "skeleton_ok": False, "has_ez": n_gold > 0}
        if r.get("valid") and n_gold:
            mp = Chem.MolFromSmiles(r["pred"])
            if mp is not None and \
               Chem.MolToSmiles(skeleton_mol(mg)) == Chem.MolToSmiles(skeleton_mol(mp)):
                mapping = skeleton_mol(mg).GetSubstructMatch(skeleton_mol(mp))
                if mapping:
                    rec["skeleton_ok"] = True
                    lp = ez_labels(mp, mapping)
                    rec["assigned"] = sum(1 for k in lg if k in lp)
                    rec["correct"] = sum(1 for k, v in lg.items() if lp.get(k) == v)
        out[cid] = rec
    return out


def main():
    data = {name: per_caption(json.loads(p.read_text(encoding="utf-8"))["records"])
            for name, p in MODELS.items()}

    # gold E/Z bonds are model-independent; take them from any model's map
    ref = data["MolT5-large"]
    total_gold_ez = sum(v["n_gold"] for v in ref.values())
    n_ez_mols = sum(1 for v in ref.values() if v["has_ez"])
    print(f"gold molecules with E/Z: {n_ez_mols}   total gold E/Z bonds: {total_gold_ez}")

    res = {"total_gold_ez_bonds": total_gold_ez, "n_ez_molecules": n_ez_mols,
           "end_to_end": {}, "common_support": {}}

    # ---- 1. end-to-end recall, common denominator = total_gold_ez ----
    print("\nend-to-end (denominator = all gold E/Z bonds, skeleton miss = failure)")
    print(f"{'model':12s} {'recall':>8} {'coverage':>9} {'precision':>10}")
    for name, d in data.items():
        correct = sum(v["correct"] for v in d.values())
        assigned = sum(v["assigned"] for v in d.values())
        rec = correct / total_gold_ez
        cov = assigned / total_gold_ez
        prec = correct / assigned if assigned else None
        res["end_to_end"][name] = {"recall": rec, "coverage": cov,
                                   "precision": prec, "correct": correct,
                                   "assigned": assigned, "N": total_gold_ez}
        print(f"{name:12s} {rec:8.4f} {cov:9.4f} "
              f"{prec:10.4f}" if prec else f"{name:12s} {rec:8.4f}")

    # ---- 2. common-support recall for the decisive pair ----
    for a, b in [("MolT5-large", "BioT5-base"), ("MolT5-base", "BioT5-base"),
                 ("MolT5-base", "MolT5-large")]:
        da, db = data[a], data[b]
        cids = [c for c in da if c in db
                and da[c]["has_ez"] and da[c]["skeleton_ok"] and db[c]["skeleton_ok"]]
        # common denominator: gold E/Z bonds on those shared molecules
        N = sum(da[c]["n_gold"] for c in cids)
        ca = sum(da[c]["correct"] for c in cids)
        cb = sum(db[c]["correct"] for c in cids)
        res["common_support"][f"{a} vs {b}"] = {
            "n_molecules": len(cids), "N_bonds": N,
            f"recall_{a}": ca / N if N else None,
            f"recall_{b}": cb / N if N else None,
            f"correct_{a}": ca, f"correct_{b}": cb}
        print(f"\ncommon support: {a} vs {b}")
        print(f"  molecules both recovered (with E/Z): {len(cids)}   "
              f"shared gold E/Z bonds: {N}")
        if N:
            print(f"  recall {a:12s}: {ca/N:.4f}  ({ca}/{N})")
            print(f"  recall {b:12s}: {cb/N:.4f}  ({cb}/{N})")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
