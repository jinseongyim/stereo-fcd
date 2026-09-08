"""
Step 5 (training-free form): can FCD rank a stereo-faithful generator against a
stereo-randomizing one?

Construction (no model training required):
  R  reference set   : held-out ChEMBL molecules carrying E/Z stereo
  A  "good" model    : molecules reproducing the reference stereo exactly
  B  "broken" model  : SAME molecular graphs, E/Z reassigned at random

B is chemically a different set of compounds -- different canonical SMILES,
different 3D shape, different bioactivity. If FCD cannot separate A from B,
then FCD assigns identical scores to a generator that has learned double-bond
stereochemistry and one that has not.

Reported alongside: the standard companion metrics from MOSES/GuacaMol
(validity, uniqueness, novelty, internal diversity, and stereo accuracy)
to show which of them, if any, notice.
"""
import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_fcd, load_ref_model

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
SAMPLE = HERE / "chembl_stereo.smi"   # bulk-extracted: molecules carrying '/' or '\'
OUT = HERE / "rank_tie_results.json"
SEED = 0
SUBSAMPLE = 20000

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}

mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def stereo_bonds(mol):
    return [b for b in mol.GetBonds() if b.GetStereo() in STEREO_SET]


def randomize_stereo(smi, rng):
    """Reassign every specified E/Z bond by a coin flip. Returns (smiles, n_bonds, n_flipped)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, 0, 0
    bonds = stereo_bonds(mol)
    if not bonds:
        return None, 0, 0
    # mutate in place: an RWMol copy keeps the original BondDir, which the SMILES
    # writer prefers over bond stereo, silently returning the input unchanged.
    n_flipped = 0
    for b in bonds:
        if rng.random() < 0.5:
            b.SetStereo(FLIP[b.GetStereo()])
            n_flipped += 1
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        return Chem.MolToSmiles(mol), len(bonds), n_flipped
    except Exception:
        return None, len(bonds), n_flipped


def stereo_accuracy(gen, ref_lookup):
    """Fraction of generated molecules whose E/Z assignment matches the reference."""
    ok = sum(1 for g, r in zip(gen, ref_lookup) if g == r)
    return ok / len(gen)


def internal_diversity(smis, rng, k=500):
    sub = smis if len(smis) <= k else rng.sample(smis, k)
    fps = [mgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in sub]
    tot, n = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        tot += sum(sims); n += len(sims)
    return 1 - tot / n


def main():
    rng = random.Random(SEED)
    smiles = [s.strip() for s in SAMPLE.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(smiles) > SUBSAMPLE:
        smiles = rng.sample(smiles, SUBSAMPLE)
        print(f"subsampled to {len(smiles)} for ChemNet inference")

    stereo_mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if stereo_bonds(m):
            can = Chem.MolToSmiles(m)
            if len(can) < 340:
                stereo_mols.append(can)
    stereo_mols = sorted(set(stereo_mols))
    rng.shuffle(stereo_mols)
    print(f"ChEMBL molecules with specified E/Z stereo, usable: {len(stereo_mols)}")

    half = len(stereo_mols) // 2
    R = stereo_mols[:half]          # reference / "real" distribution
    A = stereo_mols[half:]          # stereo-faithful generator output
    print(f"  |R| = {len(R)}   |A| = {len(A)}")

    B, nb_tot, nf_tot = [], 0, 0
    for s in A:
        r, nb, nf = randomize_stereo(s, rng)
        B.append(r if r else s)
        nb_tot += nb; nf_tot += nf
    print(f"  built B: {nf_tot}/{nb_tot} stereo bonds flipped ({100*nf_tot/max(nb_tot,1):.1f}%)")

    model = load_ref_model()
    print("\ncomputing FCD ...")
    fcd_A = get_fcd(A, R, model=model)
    fcd_B = get_fcd(B, R, model=model)

    # companion metrics
    def validity(x):
        return sum(1 for s in x if Chem.MolFromSmiles(s) is not None) / len(x)
    refset = set(R)
    def novelty(x):
        return sum(1 for s in x if s not in refset) / len(x)

    ident = {
        "A_vs_B_identical_smiles": sum(1 for a, b in zip(A, B) if a == b) / len(A),
    }

    res = {
        "n_ref": len(R), "n_gen": len(A),
        "stereo_bonds_total": nb_tot, "stereo_bonds_flipped": nf_tot,
        "FCD_A_vs_R": fcd_A, "FCD_B_vs_R": fcd_B,
        "FCD_difference": fcd_B - fcd_A,
        "validity_A": validity(A), "validity_B": validity(B),
        "uniqueness_A": len(set(A)) / len(A), "uniqueness_B": len(set(B)) / len(B),
        "novelty_A": novelty(A), "novelty_B": novelty(B),
        "intdiv_A": internal_diversity(A, rng), "intdiv_B": internal_diversity(B, rng),
        "stereo_accuracy_A": 1.0,
        "stereo_accuracy_B": ident["A_vs_B_identical_smiles"],
    }

    print("\n" + "=" * 66)
    print("CAN FCD RANK A (stereo-faithful) ABOVE B (stereo-randomized)?")
    print("=" * 66)
    print(f"  FCD(A, R) = {res['FCD_A_vs_R']:.10f}")
    print(f"  FCD(B, R) = {res['FCD_B_vs_R']:.10f}")
    print(f"  difference = {res['FCD_difference']:.3e}"
          + ("   <-- EXACT TIE" if res["FCD_difference"] == 0.0 else ""))
    print(f"\n  stereo accuracy   A = {res['stereo_accuracy_A']:.4f}   B = {res['stereo_accuracy_B']:.4f}")
    print(f"  validity          A = {res['validity_A']:.4f}   B = {res['validity_B']:.4f}")
    print(f"  uniqueness        A = {res['uniqueness_A']:.4f}   B = {res['uniqueness_B']:.4f}")
    print(f"  novelty           A = {res['novelty_A']:.4f}   B = {res['novelty_B']:.4f}")
    print(f"  int. diversity    A = {res['intdiv_A']:.4f}   B = {res['intdiv_B']:.4f}")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
