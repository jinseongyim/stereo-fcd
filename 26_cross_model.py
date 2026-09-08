"""
Step 26: do FCD and stereochemical fidelity rank two real models the same way?

The strongest remaining objection is that the argument rests on one model plus an
artificial inversion. This addresses it with two independently trained
generators and no perturbation: for MolT5-base and MolT5-large we report, on the
same ChEBI-20 test set, the metric people actually use (FCD, lower is better)
beside the per-molecule stereochemistry it cannot see (coverage / precision /
recall against ground truth, computed exactly as in step 24 with an explicit
substructure correspondence).

FCD is computed by the SAME path as the main-text Table 1: reference set from
ev.prep(chebi20_test.txt), generations canonicalized and length-filtered, frozen
ChemNet embeddings, the fcd package's own Frechet distance. This guarantees
MolT5-large's number here equals the main-text baseline (0.386789).
"""
import importlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd.utils import calculate_frechet_distance

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "cross_model.json"
ev = importlib.import_module("14_eval_stereofcd")

MODELS = {
    "MolT5-base":  HERE / "molt5_outputs_molt5-base-caption2smiles.json",
    "MolT5-large": HERE / "molt5_outputs_molt5-large-caption2smiles.json",
    "BioT5-base":  HERE / "molt5_outputs_biot5-base-text2mol.json",
}

SS = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
      Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}


def frechet(x, y):
    return calculate_frechet_distance(x.mean(0), np.cov(x.T), y.mean(0), np.cov(y.T))


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
            out[tuple(sorted((i, j)))] = "E" if b.GetStereo() in (
                Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS) else "Z"
    return out


def coverage_precision_recall(recs):
    N = assigned = correct = 0
    for r in recs:
        if not r.get("valid"):
            continue
        mg, mp = Chem.MolFromSmiles(r["gt"]), Chem.MolFromSmiles(r["pred"])
        if mg is None or mp is None:
            continue
        if Chem.MolToSmiles(skeleton_mol(mg)) != Chem.MolToSmiles(skeleton_mol(mp)):
            continue
        lg = ez_labels(mg)
        if not lg:
            continue
        mapping = skeleton_mol(mg).GetSubstructMatch(skeleton_mol(mp))
        if not mapping:
            continue
        lp = ez_labels(mp, mapping)
        N += len(lg)
        assigned += sum(1 for k in lg if k in lp)
        correct += sum(1 for k, v in lg.items() if lp.get(k) == v)
    return {"N": N, "assigned": assigned, "correct": correct,
            "coverage": assigned / N, "precision": correct / assigned,
            "recall": correct / N}


def main():
    sfcd = ev.StereoFCD()
    ref = ev.prep(HERE / "chebi20_test.txt")
    E_ref = sfcd.embed(ref, correction=False)
    print(f"reference {len(ref)}")

    res = {}
    for name, path in MODELS.items():
        recs = json.loads(path.read_text(encoding="utf-8"))["records"]
        gen = []
        for r in recs:
            if not r["valid"]:
                continue
            m = Chem.MolFromSmiles(r["pred"])
            if m is None:
                continue
            c = Chem.MolToSmiles(m)
            if len(c) < 340:
                gen.append(c)
        E_gen = sfcd.embed(gen, correction=False)
        fcd = frechet(E_gen, E_ref)
        cpr = coverage_precision_recall(recs)
        validity = sum(1 for r in recs if r["valid"]) / len(recs)
        res[name] = {"n": len(recs), "n_gen": len(gen), "validity": validity,
                     "fcd": float(fcd), **cpr}
        print(f"\n{name}  (n={len(recs)}, usable gen {len(gen)}, "
              f"validity {validity:.3f})")
        print(f"  FCD (lower=better) : {fcd:.4f}")
        print(f"  coverage           : {cpr['coverage']:.4f}")
        print(f"  precision          : {cpr['precision']:.4f}")
        print(f"  recall             : {cpr['recall']:.4f}")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    names = list(res)
    by_fcd = sorted(names, key=lambda k: res[k]["fcd"])
    by_rec = sorted(names, key=lambda k: -res[k]["recall"])
    print("\n" + "=" * 60)
    print("ranking by FCD (best first)  :", " > ".join(by_fcd))
    print("ranking by stereo recall     :", " > ".join(by_rec))
    print("the two orderings agree      :", by_fcd == by_rec)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
