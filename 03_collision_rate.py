"""
Step 3: measure the standard-metric collision rate on REAL molecules.

For every ChEMBL molecule carrying specified double-bond stereochemistry,
build its E/Z-flipped counterpart and ask, per metric, whether the flip is
invisible:

  ChemNet activation   identical  -> FCD cannot see it
  Morgan / MACCS / RDKit-fp Tanimoto == 1.0
  Morgan(includeChirality=True)     -> control, expected to SEE it

Reports prevalence (how many real molecules even have E/Z) and, among those,
the per-metric collision rate.
"""
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, MACCSkeys, rdFingerprintGenerator

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import load_ref_model
from fcd.fcd import get_predictions

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
SAMPLE = HERE / "chembl_stereo.smi"   # bulk-extracted: molecules carrying '/' or '\'
OUT = HERE / "collision_results.json"
SUBSAMPLE = 20000                     # cap for ChemNet inference cost
SEED = 0

mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mgen_ch = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def stereo_bonds(mol):
    return [b for b in mol.GetBonds() if b.GetStereo() in STEREO_SET]


def flip_all(smi):
    """Return canonical SMILES with every specified E/Z bond inverted, or None.

    NB: mutate the parsed mol in place and force re-assignment. Copying into an
    RWMol keeps the original BondDir on the neighbouring single bonds, and the
    writer prefers those over the bond stereo, silently producing the input back.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, 0
    bonds = stereo_bonds(mol)
    if not bonds:
        return None, 0
    for b in bonds:
        b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        return Chem.MolToSmiles(mol), len(bonds)
    except Exception:
        return None, len(bonds)


def main():
    import random
    smiles = [s.strip() for s in SAMPLE.read_text(encoding="utf-8").splitlines() if s.strip()]
    print(f"loaded {len(smiles)} ChEMBL SMILES carrying '/' or '\\'")
    if len(smiles) > SUBSAMPLE:
        smiles = random.Random(SEED).sample(smiles, SUBSAMPLE)
        print(f"  subsampled to {len(smiles)} for ChemNet inference")

    pairs = []          # (orig_canonical, flipped_canonical, n_stereo_bonds)
    n_parsed = n_with_stereo = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n_parsed += 1
        can = Chem.MolToSmiles(m)
        if not stereo_bonds(m):
            continue
        n_with_stereo += 1
        flipped, nb = flip_all(can)
        if flipped and flipped != can:
            pairs.append((can, flipped, nb))

    print(f"  parsed                       : {n_parsed}")
    print(f"  with specified E/Z stereo    : {n_with_stereo}  ({100*n_with_stereo/n_parsed:.2f}%)")
    print(f"  usable flipped pairs         : {len(pairs)}")
    if not pairs:
        print("no pairs -- stopping")
        return

    # length guard: ChemNet default pad_len is 350
    pairs = [p for p in pairs if max(len(p[0]), len(p[1])) < 340]
    print(f"  after length filter (<340)   : {len(pairs)}")

    model = load_ref_model()
    origs = [p[0] for p in pairs]
    flips = [p[1] for p in pairs]

    print("\ncomputing ChemNet activations ...")
    a_o = get_predictions(model, origs, device="cpu")
    a_f = get_predictions(model, flips, device="cpu")
    dact = np.abs(a_o - a_f).max(axis=1)

    print("computing fingerprints ...")
    rows = []
    for (o, f, nb), d in zip(pairs, dact):
        mo, mf = Chem.MolFromSmiles(o), Chem.MolFromSmiles(f)
        rows.append({
            "orig": o, "flip": f, "n_stereo_bonds": nb,
            "chemnet_maxabs": float(d),
            "morgan": DataStructs.TanimotoSimilarity(mgen.GetFingerprint(mo), mgen.GetFingerprint(mf)),
            "morgan_chiral": DataStructs.TanimotoSimilarity(mgen_ch.GetFingerprint(mo), mgen_ch.GetFingerprint(mf)),
            "maccs": DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(mo), MACCSkeys.GenMACCSKeys(mf)),
            "rdkit_fp": DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(mo), Chem.RDKFingerprint(mf)),
        })

    n = len(rows)
    print("\n" + "=" * 62)
    print(f"COLLISION RATES over {n} real E/Z-flip pairs from ChEMBL")
    print("=" * 62)
    c_chemnet = sum(1 for r in rows if r["chemnet_maxabs"] == 0.0)
    print(f"  ChemNet activation exactly identical : {c_chemnet}/{n} = {100*c_chemnet/n:.2f}%")
    for key in ["morgan", "morgan_chiral", "maccs", "rdkit_fp"]:
        c = sum(1 for r in rows if r[key] == 1.0)
        print(f"  Tanimoto {key:14s} == 1.0      : {c}/{n} = {100*c/n:.2f}%")

    OUT.write_text(json.dumps({
        "n_sampled": len(smiles), "n_parsed": n_parsed,
        "n_with_stereo": n_with_stereo, "n_pairs": n,
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
