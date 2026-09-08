"""
Step 12: build the training signal for the stereo correction term.

For each molecule carrying E/Z we emit BOTH configurations (as written, and
fully inverted) and compute 3D shape descriptors from an ETKDG conformer.
Those descriptors are exactly the quantity double-bond geometry controls and
the frozen ChemNet provably cannot see, so they are what the correction term
should learn.

Output: targets.jsonl  {smiles, flipped, desc: [...]}
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "targets.jsonl"
N_MOL = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEED = 0xC0FFEE

DESC_NAMES = ["Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
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


def descriptors(smi):
    """ETKDG conformer -> 3D shape descriptors. None if embedding fails."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED
    ps.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, ps) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        pass
    try:
        return [float(getattr(Descriptors3D, n)(mol)) for n in DESC_NAMES]
    except Exception:
        return None


def work(smi):
    """Return the two configurations of one molecule with their descriptors."""
    other = flip_all(smi)
    if other is None or other == smi:
        return []
    out = []
    for s, flipped in ((smi, False), (other, True)):
        d = descriptors(s)
        if d is not None:
            out.append({"smiles": s, "flipped": flipped, "desc": d})
    return out if len(out) == 2 else []   # keep only complete pairs


def load_sources(n):
    """ChEBI-20 train (the benchmark's own distribution) + ChEMBL stereo."""
    import random
    rng = random.Random(0)
    pool = []

    p = HERE / "chebi20_train.txt"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and ("/" in parts[1] or "\\" in parts[1]):
                    pool.append(parts[1])
    print(f"  ChEBI-20 train stereo molecules: {len(pool)}")

    chembl = [s.strip() for s in (HERE / "chembl_stereo.smi").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    rng.shuffle(chembl)
    print(f"  ChEMBL stereo pool: {len(chembl)}")

    pool = pool + chembl
    # canonicalise, keep length-safe
    seen, out = set(), []
    for s in pool:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if len(c) >= 340 or c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= n:
            break
    return out


def main():
    print("collecting source molecules ...")
    mols = load_sources(N_MOL)
    print(f"  usable: {len(mols)}")

    n_written = 0
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool, \
         open(OUT, "w", encoding="utf-8") as f:
        for i, recs in enumerate(pool.imap_unordered(work, mols, chunksize=16), 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
                n_written += 1
            if i % 500 == 0:
                f.flush()
                print(f"  {i}/{len(mols)} molecules -> {n_written} rows", flush=True)

    print(f"\nwrote {n_written} rows ({n_written//2} complete E/Z pairs) -> {OUT}")


if __name__ == "__main__":
    main()
