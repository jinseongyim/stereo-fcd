"""
Step 18: why does the marginal E/Z composition disagree with per-bond accuracy?

Step 17 established:
  - conditional on a matching skeleton, MolT5's E/Z is 98.95% correct
  - yet the output's marginal E-share is 0.881 against a truth of 0.430,
    and inverting moves it to 0.128, which is *closer* to the truth

Hypothesis: the two populations are different molecules. Where MolT5 recovers
the target skeleton its stereochemistry is nearly right; where it does not,
it emits E-heavy stereochemistry that no target constrains. A distributional
metric pools both, so it is dominated by the second group and cannot tell
"wrong stereochemistry" from "wrong molecule".

If true, Stereo-FCD's failure is a design flaw, not a training bug: any
*marginal* stereo statistic conflates the two.
"""
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
OUT = HERE / "composition_split.json"

E, Z = Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS
SS = {E, Z, CIS, TRANS}


def skeleton(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def stereo_counts(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return Counter()
    c = Counter()
    for b in m.GetBonds():
        if b.GetStereo() in SS:
            c["E" if b.GetStereo() in (E, TRANS) else "Z"] += 1
    return c


def report(name, c, nmol):
    tot = c["E"] + c["Z"]
    share = c["E"] / tot if tot else float("nan")
    print(f"  {name:34s} mols {nmol:5d}  bonds E {c['E']:5d} Z {c['Z']:5d} "
          f" total {tot:5d}  E-share {share:.3f}")
    return {"n_mol": nmol, "E": c["E"], "Z": c["Z"], "total": tot,
            "E_share": share}


def main():
    recs = json.loads(PRED.read_text(encoding="utf-8"))["records"]

    match = Counter()
    miss = Counter()
    n_match = n_miss = 0
    n_match_stereo = n_miss_stereo = 0
    truth = Counter()
    n_truth = 0

    for r in recs:
        gt = r["gt"]
        tc = stereo_counts(gt)
        if tc:
            truth += tc
            n_truth += 1
        if not r["valid"]:
            continue
        pred = r["pred"]
        sg, sp = skeleton(gt), skeleton(pred)
        if sg is None or sp is None:
            continue
        pc = stereo_counts(pred)
        if sg == sp:
            n_match += 1
            if pc:
                n_match_stereo += 1
            match += pc
        else:
            n_miss += 1
            if pc:
                n_miss_stereo += 1
            miss += pc

    print("E/Z composition, split by whether the skeleton matches the target\n")
    out = {
        "truth": report("ChEBI-20 truth", truth, n_truth),
        "skeleton_match": report("pred, skeleton MATCHES target",
                                 match, n_match_stereo),
        "skeleton_miss": report("pred, skeleton MISSES target",
                                miss, n_miss_stereo),
    }
    out["n_skeleton_match"] = n_match
    out["n_skeleton_miss"] = n_miss

    print(f"\n  molecules: skeleton match {n_match}, miss {n_miss}")
    m_tot = match["E"] + match["Z"]
    x_tot = miss["E"] + miss["Z"]
    if m_tot + x_tot:
        print(f"  share of all emitted stereo bonds coming from MISSED "
              f"skeletons: {x_tot/(m_tot+x_tot):.3f}")
        out["miss_share_of_stereo_bonds"] = x_tot / (m_tot + x_tot)

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
