"""
Step 13d: is the training target signal or noise?

The paired model collapsed to predicting zero, i.e. it concluded there was
nothing learnable. Two candidate explanations:

  (a) a bug -- the E and Z forms reach the network with identical encodings,
      so no gradient exists;
  (b) the target is dominated by ETKDG conformer sampling noise, so the E/Z
      difference we asked it to predict is not actually there.

This separates them. For the same molecule we compute descriptors under several
random conformer seeds (pure noise) and compare that spread against the
difference between the E and Z configurations (the signal we want).

If noise >> signal, single-conformer 3D descriptors are the wrong target and
the fix is conformer averaging or a different, geometry-exact target.
"""
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
DESC = ["Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
        "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"]
N_MOL = 150
N_SEED = 4

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def desc(smi, seed):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    if AllChem.EmbedMolecule(mol, ps) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        pass
    try:
        return np.array([float(getattr(Descriptors3D, n)(mol)) for n in DESC])
    except Exception:
        return None


def flip(smi):
    m = Chem.MolFromSmiles(smi)
    bs = [b for b in m.GetBonds() if b.GetStereo() in STEREO_SET]
    if not bs:
        return None
    for b in bs:
        b.SetStereo(FLIP[b.GetStereo()])
    Chem.AssignStereochemistry(m, cleanIt=False, force=True)
    return Chem.MolToSmiles(m)


def main():
    # ---- (a) encoding check: do E and Z differ at the tokenizer? ----
    import fcd.utils as U
    VOCAB = [v for k, v in vars(U).items() if k.endswith("__vocab") and isinstance(v, list)][0]
    EXT = {k: i for i, k in enumerate(VOCAB + ["/", "\\"])}
    a, b = r"C/C=C/C", r"C/C=C\C"
    ta = [EXT.get(t, EXT["X"]) for t in U.tokenize(a + ".")]
    tb = [EXT.get(t, EXT["X"]) for t in U.tokenize(b + ".")]
    print("(a) extended-token sequences for an E/Z pair")
    print(f"    E: {ta}")
    print(f"    Z: {tb}")
    print(f"    differ: {ta != tb}\n")

    # ---- (b) signal vs conformer noise ----
    rows = [json.loads(l) for l in (HERE / "targets.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    smis = [r["smiles"] for r in rows if not r["flipped"]][:N_MOL]

    sig, noi = [], []
    for s in smis:
        o = flip(s)
        if o is None or o == s:
            continue
        ds = [desc(s, 1000 + k) for k in range(N_SEED)]
        do = [desc(o, 1000 + k) for k in range(N_SEED)]
        if any(x is None for x in ds + do):
            continue
        ds, do = np.array(ds), np.array(do)
        # noise: spread across seeds within one configuration
        noi.append(np.abs(ds - ds.mean(0)).mean(0))
        # signal: difference between configuration means
        sig.append(np.abs(ds.mean(0) - do.mean(0)))

    sig, noi = np.array(sig), np.array(noi)
    print(f"(b) over {len(sig)} molecules, {N_SEED} conformer seeds each\n")
    print(f"    {'descriptor':22s} {'|E-Z| signal':>14s} {'conformer noise':>16s} {'ratio':>8s}")
    for i, n in enumerate(DESC):
        s_, n_ = sig[:, i].mean(), noi[:, i].mean()
        print(f"    {n:22s} {s_:14.4f} {n_:16.4f} {s_/max(n_,1e-12):8.2f}")
    print(f"\n    {'MEAN RATIO':22s} {'':14s} {'':16s} "
          f"{(sig.mean(0)/np.maximum(noi.mean(0),1e-12)).mean():8.2f}")
    print("\n    ratio >> 1 : E/Z difference dominates -> target is sound")
    print("    ratio ~ 1  : conformer noise swamps the signal -> target is the problem")


if __name__ == "__main__":
    main()
