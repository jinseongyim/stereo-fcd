"""
Step 5b: is the stereo effect on FCD actually zero, or just small?

A reviewer will ask what 4.3e-05 means. So we calibrate it against two scales:

  NOISE FLOOR   spread of FCD(A_i, R) over independent equal-size resamples of A.
                Anything below this is invisible to the metric in practice.

  SENSITIVITY   FCD(A_corrupted_p, R) where a fraction p of A is swapped for
                unrelated ChEMBL molecules -- a perturbation FCD *can* see.
                Tells us what fraction of graph-level corruption is equivalent
                to 50% stereo corruption.

If Delta_stereo << noise floor, the honest claim is "FCD cannot rank these",
not merely "FCD scores them similarly".
"""
import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_fcd, load_ref_model

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
STEREO = HERE / "chembl_stereo.smi"
ALL_GZ = HERE / "chembl_37_chemreps.txt.gz"
OUT = HERE / "noise_floor_results.json"
SEED = 0
N = 9981                      # match step 4 set sizes
N_RESAMPLE = 8
CORRUPT_FRACTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def canon_pool(path, limit, rng):
    smis = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    rng.shuffle(smis)
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if len(c) < 340:
            out.append(c)
        if len(out) >= limit:
            break
    return out


def randomize_stereo(smi, rng):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    bonds = [b for b in mol.GetBonds() if b.GetStereo() in STEREO_SET]
    if not bonds:
        return smi
    for b in bonds:
        if rng.random() < 0.5:
            b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def load_nonstereo(limit, rng):
    """Unrelated molecules (no E/Z) used as the corruption source."""
    import gzip
    out = []
    with gzip.open(ALL_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            smi = p[1]
            if "/" in smi or "\\" in smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            c = Chem.MolToSmiles(m)
            if len(c) < 340:
                out.append(c)
            if len(out) >= limit * 4:
                break
    rng.shuffle(out)
    return out


def main():
    rng = random.Random(SEED)
    model = load_ref_model()

    pool = canon_pool(STEREO, 4 * N, rng)
    print(f"stereo pool: {len(pool)}")
    R = pool[:N]
    A_pool = pool[N:]
    A = A_pool[:N]

    fcd_A = get_fcd(A, R, model=model)
    B = [randomize_stereo(s, rng) for s in A]
    fcd_B = get_fcd(B, R, model=model)
    d_stereo = abs(fcd_B - fcd_A)
    print(f"\nFCD(A,R) = {fcd_A:.8f}   FCD(B,R) = {fcd_B:.8f}   |delta| = {d_stereo:.3e}")

    print(f"\nnoise floor: {N_RESAMPLE} independent resamples of A")
    scores = []
    for i in range(N_RESAMPLE):
        sub = rng.sample(A_pool, N)
        s = get_fcd(sub, R, model=model)
        scores.append(s)
        print(f"  resample {i+1}: FCD = {s:.8f}")
    scores = np.array(scores)
    noise_sd = float(scores.std(ddof=1))
    noise_range = float(scores.max() - scores.min())
    print(f"  sd = {noise_sd:.3e}   range = {noise_range:.3e}")

    print("\nsensitivity: replacing a fraction of A with unrelated (non-stereo) molecules")
    others = load_nonstereo(N, rng)
    sens = {}
    for p in CORRUPT_FRACTIONS:
        k = int(round(p * N))
        C = A[:N - k] + others[:k]
        s = get_fcd(C, R, model=model)
        sens[p] = s
        print(f"  p={p:<6.3f} (k={k:4d})  FCD = {s:.8f}   delta = {abs(s-fcd_A):.3e}")

    res = {
        "n": N, "fcd_A": fcd_A, "fcd_B_stereo_randomized": fcd_B,
        "delta_stereo": d_stereo,
        "noise_resamples": scores.tolist(),
        "noise_sd": noise_sd, "noise_range": noise_range,
        "delta_stereo_in_sd_units": d_stereo / noise_sd if noise_sd else None,
        "sensitivity": {str(k): v for k, v in sens.items()},
    }
    print("\n" + "=" * 64)
    print(f"  delta(50% stereo corruption) = {d_stereo:.3e}")
    print(f"  FCD resampling noise sd      = {noise_sd:.3e}")
    print(f"  ratio                        = {d_stereo/noise_sd:.4f} sd")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
