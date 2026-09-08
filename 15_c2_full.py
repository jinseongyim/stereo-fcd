"""
Step 15: the C2 experiment done properly, on the full ChEBI-20 test set.

The earlier version used the 800-molecule subsample and reported a bare delta
with no uncertainty, which is why its SIGN could not be trusted. Here we run
MolT5-large's full 3,300-molecule output and attach:

  * a paired bootstrap CI on the delta (resampling molecules, so the real and
    inverted sets move together and the comparison stays paired);
  * the metric's own resampling noise floor at the SAME n, which is the bar the
    delta has to clear;
  * the published SOTA margins on this benchmark, which is the bar that matters
    scientifically.

A metric passes only if the stereo delta clears its noise floor. FCD is expected
to fail this by construction: its delta is exactly zero.
"""
import json
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

# fcd 1.2.2 calls scipy.linalg.sqrtm(A, disp=False); scipy dropped `disp` in
# recent versions and now returns the matrix alone. Restore the old two-value
# contract so the vendored Frechet code keeps working.
import inspect as _inspect
import scipy.linalg as _sla
if "disp" not in _inspect.signature(_sla.sqrtm).parameters:
    _sqrtm_orig = _sla.sqrtm

    def _sqrtm_comp(A, disp=True, blocksize=64):
        r = _sqrtm_orig(A)
        return r if disp else (r, 0.0)

    _sla.sqrtm = _sqrtm_comp

from fcd.utils import calculate_frechet_distance
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "c2_full_results.json"
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
N_BOOT = 200
SEED = 0

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}

import importlib
ev = importlib.import_module("14_eval_stereofcd")


def invert(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    bs = [b for b in m.GetBonds() if b.GetStereo() in STEREO_SET]
    if not bs:
        return smi
    for b in bs:
        b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(m, cleanIt=False, force=True)
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


def frechet(x, y):
    return calculate_frechet_distance(x.mean(0), np.cov(x.T), y.mean(0), np.cov(y.T))


def main():
    rng = np.random.default_rng(SEED)
    d = json.loads(PRED.read_text(encoding="utf-8"))
    print(f"MolT5 records: {d['n']}  (valid {d['n_valid']}, E/Z-bearing {d['n_pred_ez']})")

    gen = []
    for r in d["records"]:
        if not r["valid"]:
            continue
        m = Chem.MolFromSmiles(r["pred"])
        c = Chem.MolToSmiles(m)
        if len(c) < 340:
            gen.append(c)
    inv = [invert(s) for s in gen]
    n_changed = sum(1 for a, b in zip(gen, inv) if a != b)
    print(f"usable generations {len(gen)}, of which {n_changed} "
          f"({100*n_changed/len(gen):.2f}%) change under inversion")

    ref = ev.prep(HERE / "chebi20_test.txt")
    print(f"reference {len(ref)}")

    sfcd = ev.StereoFCD()
    print("embedding ...", flush=True)
    E_ref_f = sfcd.embed(ref, correction=False)
    E_ref_s = sfcd.embed(ref, correction=True)
    E_gen_f = sfcd.embed(gen, correction=False)
    E_gen_s = sfcd.embed(gen, correction=True)
    E_inv_f = sfcd.embed(inv, correction=False)
    E_inv_s = sfcd.embed(inv, correction=True)

    res = {"n_gen": len(gen), "n_changed": n_changed, "n_ref": len(ref)}
    for name, (Eg, Ei, Er) in {
        "FCD": (E_gen_f, E_inv_f, E_ref_f),
        "Stereo-FCD": (E_gen_s, E_inv_s, E_ref_s),
    }.items():
        real = frechet(Eg, Er)
        flip = frechet(Ei, Er)
        delta = flip - real

        # paired bootstrap over molecules
        deltas = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(Eg), len(Eg))
            deltas.append(frechet(Ei[idx], Er) - frechet(Eg[idx], Er))
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        # noise floor at the same n: split the reference and score one half
        floor = []
        for _ in range(12):
            a = rng.choice(len(Er), len(Eg), replace=False)
            floor.append(frechet(Er[a], Er))
        sd = float(np.std(floor, ddof=1))

        res[name] = {"real": real, "inverted": flip, "delta": delta,
                     "ci95": [float(lo), float(hi)], "noise_sd": sd,
                     "delta_in_sd": float(delta / sd) if sd else None}
        print(f"\n{name}")
        print(f"  real     = {real:.6f}")
        print(f"  inverted = {flip:.6f}")
        print(f"  delta    = {delta:+.6e}   95% CI [{lo:+.3e}, {hi:+.3e}]")
        print(f"  noise sd = {sd:.3e}   ->  {delta/sd if sd else float('nan'):+.2f} sd")
        print(f"  clears its own noise floor: {abs(delta) > 2*sd}")

    print("\npublished SOTA margins on ChEBI-20 for scale:")
    print("  3D-MolT5 0.41 / BioT5 0.43 / MolXPT 0.45  -> winning margin 0.02")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
