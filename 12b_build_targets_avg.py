"""
Step 12b: rebuild the training target with conformer averaging.

Step 13d showed the single-conformer target had a signal-to-noise ratio of only
1.47 -- the E/Z difference was barely above ETKDG sampling noise, which is why
the paired model correctly concluded there was nothing to learn.

Averaging K conformers cuts the noise by sqrt(K), so K=10 lifts the ratio to
roughly 4.6. We trade molecule count for conformers per molecule, which is the
right trade: the quantity being learned is a within-pair difference, so a
cleaner target on fewer skeletons beats a noisy one on more.

EmbedMultipleConfs amortises setup across conformers, so this is far cheaper
than K independent embeddings.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "targets_avg.jsonl"
N_MOL = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
N_CONF = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = 0xC0FFEE

DESC = ["Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
        "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"]

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def flip_all(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bonds = [b for b in mol.GetBonds() if b.GetStereo() in STEREO_SET]
    if not bonds:
        return None
    for b in bonds:
        b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def descriptors_avg(smi):
    """Mean 3D descriptors over N_CONF ETKDG conformers. Also returns the
    within-molecule spread so the trainer can weight noisy rows down."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED
    ps.pruneRmsThresh = -1.0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, params=ps)
    if len(cids) == 0:
        return None, None
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=300)
    except Exception:
        pass
    vals = []
    for cid in cids:
        try:
            vals.append([float(getattr(Descriptors3D, n)(mol, confId=cid)) for n in DESC])
        except Exception:
            continue
    if not vals:
        return None, None
    v = np.array(vals)
    return v.mean(0).tolist(), v.std(0).tolist()


def work(smi):
    other = flip_all(smi)
    if other is None or other == smi:
        return []
    out = []
    for s, flipped in ((smi, False), (other, True)):
        m, sdv = descriptors_avg(s)
        if m is None:
            return []
        out.append({"smiles": s, "flipped": flipped, "desc": m, "sd": sdv})
    return out


def main():
    src = [json.loads(l)["smiles"] for l in (HERE / "targets.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    seen, mols = set(), []
    for s in src:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        key = Chem.MolToSmiles(m, isomericSmiles=False)
        if key in seen:
            continue
        seen.add(key)
        mols.append(s)
        if len(mols) >= N_MOL:
            break
    print(f"skeletons to process: {len(mols)}  ({N_CONF} conformers per configuration)")

    n = 0
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool, \
         open(OUT, "w", encoding="utf-8") as f:
        for i, recs in enumerate(pool.imap_unordered(work, mols, chunksize=8), 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
                n += 1
            if i % 250 == 0:
                f.flush()
                print(f"  {i}/{len(mols)} -> {n} rows", flush=True)
    print(f"\nwrote {n} rows ({n//2} pairs) -> {OUT}")


if __name__ == "__main__":
    main()
