"""
Step 21: flip versus strip, in ONE pool with ONE noise estimate.

The paper's most quotable claim is that FCD penalizes *deleting* stereochemistry
far more than *getting it wrong*. Until now that rested on two separate
experiments with differently-constructed pools (05_noise_floor.py and
06_strip_penalty.py), which invites the objection that the two numbers are not
comparable.

This script removes the objection. A single reference set R and a single
candidate set A are drawn from one pool; every condition is a transformation of
the *same* A, scored against the *same* R, and calibrated against one noise
floor estimated from resamples of the same pool.

Conditions
  baseline    A unchanged
  flip100     every stereo double bond inverted
  flip50      each stereo double bond inverted with probability 1/2
  strip_ez    E/Z removed, tetrahedral centres left intact  <- the clean control
  strip_all   all stereochemistry removed (isomericSmiles=False)
  swap2/swap5 2% / 5% of molecules replaced by unrelated ones (scale reference)

strip_ez is the decisive comparison: relative to flip100 the molecules differ
only in whether their E/Z assignments are wrong or absent. Everything else --
the molecules, the tetrahedral centres, the set size -- is held fixed.
"""
import gzip
import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_fcd, load_ref_model

import importlib as _il
_m34=_il.import_module('34_inversion_bug_scope')
invert_correct=_m34.invert_correct
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
STEREO = HERE / "chembl_stereo.smi"
ALL_GZ = HERE / "chembl_37_chemreps.txt.gz"
OUT = HERE / "flip_vs_strip_onepool.json"

SEED = 0
N = 9981
N_RESAMPLE = 8

SS = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
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


def _edit(smi, fn):
    """Parse, mutate in place, reassign. Never via RWMol copy -- a copied
    BondDir wins in the SMILES writer and silently undoes the edit."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    bonds = [b for b in mol.GetBonds() if b.GetStereo() in SS]
    if not bonds:
        return smi
    fn(mol, bonds)
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def flip_all(smi):
    return invert_correct(smi)


def flip_half(smi, rng):
    def f(m, bs):
        for b in bs:
            if rng.random() < 0.5:
                b.SetStereo(FLIP[b.GetStereo()])
    return _edit(smi, f)


def strip_ez(smi):
    """Remove double-bond geometry only; leave @ untouched."""
    def f(m, bs):
        for b in bs:
            b.SetStereo(Chem.BondStereo.STEREONONE)
        for b in m.GetBonds():
            b.SetBondDir(Chem.BondDir.NONE)
    return _edit(smi, f)


def strip_all(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else smi


def load_nonstereo(limit, rng):
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


def changed(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    rng = random.Random(SEED)
    model = load_ref_model()

    pool = canon_pool(STEREO, 4 * N, rng)
    print(f"pool: {len(pool)}")
    R, A_pool = pool[:N], pool[N:]
    A = A_pool[:N]

    others = load_nonstereo(N, rng)

    conds = {
        "baseline":  A,
        "flip100":   [flip_all(s) for s in A],
        "flip50":    [flip_half(s, rng) for s in A],
        "strip_ez":  [strip_ez(s) for s in A],
        "strip_all": [strip_all(s) for s in A],
        "swap2":     A[:N - int(0.02 * N)] + others[:int(0.02 * N)],
        "swap5":     A[:N - int(0.05 * N)] + others[:int(0.05 * N)],
    }

    print("\nnoise floor: %d resamples of the same pool" % N_RESAMPLE)
    scores = []
    for i in range(N_RESAMPLE):
        s = get_fcd(rng.sample(A_pool, N), R, model=model)
        scores.append(s)
        print(f"  resample {i+1}: {s:.8f}")
    noise_sd = float(np.array(scores).std(ddof=1))
    print(f"  sd = {noise_sd:.3e}")

    base = None
    res = {"n": N, "noise_sd": noise_sd, "noise_resamples": scores,
           "conditions": {}}
    print(f"\n{'condition':<11} {'FCD':>11} {'delta':>12} {'d/sigma':>9}  changed")
    for name, S in conds.items():
        f = get_fcd(S, R, model=model)
        if base is None:
            base = f
        d = f - base
        nch = changed(A, S)
        print(f"{name:<11} {f:11.8f} {d:12.3e} {d/noise_sd:9.3f}  {nch}")
        res["conditions"][name] = {"fcd": f, "delta": d,
                                   "delta_in_sd": d / noise_sd,
                                   "n_changed": nch}

    fl = res["conditions"]["flip100"]["delta_in_sd"]
    st = res["conditions"]["strip_ez"]["delta_in_sd"]
    res["ratio_stripez_over_flip100"] = abs(st) / abs(fl) if fl else None
    print("\n" + "=" * 62)
    print(f"  flip100  (every E/Z wrong)   : {fl:+.4f} sigma")
    print(f"  strip_ez (every E/Z absent)  : {st:+.4f} sigma")
    if fl:
        print(f"  ratio                        : {abs(st)/abs(fl):.0f}x")
    print("  same pool, same reference set, same noise estimate.")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
