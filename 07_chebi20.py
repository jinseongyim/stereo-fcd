"""
Step 7: verify the load-bearing claim on ChEBI-20 -- the benchmark that the
text-to-molecule literature (MolT5, BioT5, 3D-MolT5, LaMolT5, ...) actually
reports FCD on.

Measures, from the primary data:
  1. E/Z prevalence in ChEBI-20 train/test
  2. FCD(reference, ground-truth test set) vs FCD(reference, all-E/Z-flipped)
  3. how that compares to the published SOTA margins on this benchmark
"""
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_fcd, load_ref_model
from fcd.fcd import get_predictions

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "chebi20_results.json"
SEED = 0

RAW = "https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data/{}.txt"
SPLITS = ["train", "validation", "test"]

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def fetch(split):
    p = HERE / f"chebi20_{split}.txt"
    if not p.exists():
        print(f"  downloading {split} ...", flush=True)
        urllib.request.urlretrieve(RAW.format(split), p)
    return p


def read_smiles(path):
    """ChEBI-20 files are TSV: CID <tab> SMILES <tab> description."""
    out = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                out.append(parts[1])
    return out, header


def flip_all(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, 0
    bonds = [b for b in mol.GetBonds() if b.GetStereo() in STEREO_SET]
    if not bonds:
        return Chem.MolToSmiles(mol), 0
    for b in bonds:
        b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        return Chem.MolToSmiles(mol), len(bonds)
    except Exception:
        return None, len(bonds)


def main():
    rng = random.Random(SEED)
    stats = {}

    print("ChEBI-20 prevalence")
    data = {}
    for sp in SPLITS:
        path = fetch(sp)
        smis, header = read_smiles(path)
        data[sp] = smis
        n = len(smis)
        ez = sum(1 for s in smis if "/" in s or "\\" in s)
        at = sum(1 for s in smis if "@" in s)
        stats[sp] = {"n": n, "ez": ez, "ez_frac": ez / n, "at": at, "at_frac": at / n}
        print(f"  {sp:11s} n={n:6d}   E/Z {ez:6d} ({100*ez/n:5.2f}%)   '@' {at:6d} ({100*at/n:5.2f}%)")

    test = data["test"]
    train = data["train"]

    # canonicalize + keep parseable, length-safe
    def prep(smis):
        out = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            c = Chem.MolToSmiles(m)
            if len(c) < 340:
                out.append(c)
        return out

    T = prep(test)
    print(f"\nusable canonical test molecules: {len(T)} / {len(test)}")

    # flipped version of the test set
    F, n_flipped_mols, n_bonds = [], 0, 0
    for s in T:
        f, nb = flip_all(s)
        if f is None:
            F.append(s)
            continue
        F.append(f)
        if nb:
            n_bonds += nb
            if f != s:
                n_flipped_mols += 1
    print(f"molecules made stereochemically wrong: {n_flipped_mols} "
          f"({100*n_flipped_mols/len(T):.2f}%), {n_bonds} bonds inverted")

    model = load_ref_model()

    # reference = training-set molecules (what the metric compares against)
    R = prep(train)
    rng.shuffle(R)
    R = R[:len(T) * 2]
    print(f"reference set: {len(R)} training molecules")

    print("\ncomputing FCD ...")
    fcd_true = get_fcd(T, R, model=model)
    fcd_flip = get_fcd(F, R, model=model)

    # bit-level check
    a_t = get_predictions(model, T)
    a_f = get_predictions(model, F)
    max_abs = float(np.abs(a_t - a_f).max())
    identical = bool(np.array_equal(a_t, a_f))

    # noise floor: FCD of equal-size resamples of the reference pool
    pool = prep(train)
    rng.shuffle(pool)
    scores = []
    for i in range(6):
        sub = rng.sample(pool, len(T))
        scores.append(get_fcd(sub, R, model=model))
    noise_sd = float(np.std(scores, ddof=1))

    print("\n" + "=" * 68)
    print("ChEBI-20: does FCD notice a systematically stereo-inverted generator?")
    print("=" * 68)
    print(f"  FCD(test TRUE stereo,     ref) = {fcd_true:.6f}")
    print(f"  FCD(test ALL E/Z FLIPPED, ref) = {fcd_flip:.6f}")
    print(f"  difference                     = {abs(fcd_flip-fcd_true):.3e}")
    print(f"  activations bit-identical      = {identical}   max|delta| = {max_abs:.3e}")
    print(f"  resampling noise sd            = {noise_sd:.4f}")
    print("\n  published SOTA margins on this benchmark for reference:")
    print("    3D-MolT5 0.41 / BioT5 0.43 / MolXPT 0.45   -> winning margin 0.02")
    print("    LaMolT5 0.352 ... Text+Chem T5 0.499       -> whole leaderboard in 0.15")

    stats["fcd"] = {
        "fcd_true": fcd_true, "fcd_flipped": fcd_flip,
        "difference": abs(fcd_flip - fcd_true),
        "activations_identical": identical, "max_abs_activation_delta": max_abs,
        "noise_sd": noise_sd,
        "n_test_used": len(T), "n_molecules_inverted": n_flipped_mols,
        "n_bonds_inverted": n_bonds,
    }
    OUT.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
