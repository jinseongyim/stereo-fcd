"""
Step 22: the single-pool comparison, repeated over independent draws.

Step 21 established the contrast within one pool but from a single (R, A) draw,
with a noise floor estimated from only 8 resamples. Reporting "36.8 sigma"
against an 8-sample standard deviation is fragile, and the size of the stripping
penalty depends on set composition in a way one draw cannot show.

This script fixes both with a cleaner design. We take D independent (R, A) draws
from the same pool. Within each draw every condition is a transformation of that
draw's own A, scored against that draw's own R, so the comparison is *paired*.
Two quantities then come out of the same experiment:

  noise floor      the spread of baseline FCD across draws -- this IS the
                   sampling variability, estimated from D draws rather than
                   from resamples inside one draw

  effect           the paired difference (condition - baseline) within each
                   draw, reported as a mean with a bootstrap 95% CI over draws

Paired differences are the right estimator here: draw-to-draw variation in
baseline FCD is common to both terms and cancels, so the CI reflects uncertainty
in the *effect* rather than in the baseline.
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
OUT = HERE / "flip_vs_strip_repeated.json"

SEED = 1
N = 9981
N_DRAWS = 15
N_BOOT = 10000

SS = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
      Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def load_nonstereo(limit, rng):
    """Unrelated molecules with no E/Z direction, for the substitution scale
    reference. Matches load_nonstereo in 21_flip_vs_strip_onepool.py."""
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
            if len(out) >= limit:
                break
    rng.shuffle(out)
    return out


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


def strip_ez(smi):
    def f(m, bs):
        for b in bs:
            b.SetStereo(Chem.BondStereo.STEREONONE)
        for b in m.GetBonds():
            b.SetBondDir(Chem.BondDir.NONE)
    return _edit(smi, f)


def boot_ci(x, rng, n=N_BOOT, alpha=0.05):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def main():
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)
    model = load_ref_model()

    pool = canon_pool(STEREO, 40000, rng)
    print(f"pool: {len(pool)}  draws: {N_DRAWS}  n per set: {N}")
    if len(pool) < 2 * N:
        raise SystemExit("pool too small")

    # unrelated non-stereo molecules for the substitution conditions; sampled
    # fresh per draw so swap2/swap5 carry the same 15-draw uncertainty as the
    # stereo conditions (fixes the protocol mix in Figure 1c). A SEPARATE rng is
    # used so the (R, A) draw sequence is byte-for-byte identical to the original
    # flip/strip-only run -- flip100 and strip_ez per-draw values are unchanged.
    orng = random.Random(SEED + 1)
    n2, n5 = int(0.02 * N), int(0.05 * N)
    others_pool = load_nonstereo(N_DRAWS * n5 + n5, orng)
    print(f"others pool (non-stereo): {len(others_pool)}")

    rows = []
    for d in range(N_DRAWS):
        sample = rng.sample(pool, 2 * N)
        R, A = sample[:N], sample[N:]
        others = orng.sample(others_pool, n5)
        f_base = get_fcd(A, R, model=model)
        f_flip = get_fcd([flip_all(s) for s in A], R, model=model)
        f_strip = get_fcd([strip_ez(s) for s in A], R, model=model)
        f_swap2 = get_fcd(A[:N - n2] + others[:n2], R, model=model)
        f_swap5 = get_fcd(A[:N - n5] + others[:n5], R, model=model)
        rows.append({"draw": d, "baseline": f_base,
                     "flip100": f_flip, "strip_ez": f_strip,
                     "swap2": f_swap2, "swap5": f_swap5,
                     "d_flip": f_flip - f_base, "d_strip": f_strip - f_base,
                     "d_swap2": f_swap2 - f_base, "d_swap5": f_swap5 - f_base})
        print(f"  draw {d:2d}: base {f_base:.6f}  flip {f_flip:.6f} "
              f"(d {f_flip-f_base:+.3e})  strip {f_strip:.6f} "
              f"(d {f_strip-f_base:+.3e})  swap2 {f_swap2-f_base:+.3e} "
              f"swap5 {f_swap5-f_base:+.3e}")

    base = np.array([r["baseline"] for r in rows])
    d_flip = np.array([r["d_flip"] for r in rows])
    d_strip = np.array([r["d_strip"] for r in rows])
    d_swap2 = np.array([r["d_swap2"] for r in rows])
    d_swap5 = np.array([r["d_swap5"] for r in rows])

    sigma = float(base.std(ddof=1))
    print(f"\nnoise floor: sd of baseline FCD across {N_DRAWS} draws = {sigma:.3e}")

    res = {"n": N, "n_draws": N_DRAWS, "seed": SEED,
           "noise_sd_across_draws": sigma, "rows": rows}

    print(f"\n{'condition':<10} {'mean delta':>12} {'95% CI':>28} {'mean d/sigma':>13}")
    for name, d in (("flip100", d_flip), ("strip_ez", d_strip),
                    ("swap2", d_swap2), ("swap5", d_swap5)):
        lo, hi = boot_ci(d, nprng)
        res[name] = {"mean_delta": float(d.mean()),
                     "sd_delta": float(d.std(ddof=1)),
                     "ci95_delta": [lo, hi],
                     "mean_delta_in_sd": float(d.mean()) / sigma,
                     "ci95_delta_in_sd": [lo / sigma, hi / sigma],
                     "per_draw": d.tolist()}
        print(f"{name:<10} {d.mean():12.3e} "
              f"[{lo:11.3e}, {hi:11.3e}] {d.mean()/sigma:13.3f}")

    ratio = float(d_strip.mean() / d_flip.mean()) if d_flip.mean() else None
    # bootstrap the ratio jointly, resampling draws (keeps the pairing)
    idx = nprng.integers(0, N_DRAWS, size=(N_BOOT, N_DRAWS))
    rat = d_strip[idx].mean(axis=1) / d_flip[idx].mean(axis=1)
    rlo, rhi = float(np.quantile(rat, 0.025)), float(np.quantile(rat, 0.975))
    res["ratio_strip_over_flip"] = {"point": ratio, "ci95": [rlo, rhi]}

    print("\n" + "=" * 66)
    print(f"  flip100  : {d_flip.mean()/sigma:+.3f} sigma  "
          f"(95% CI {boot_ci(d_flip, nprng)[0]/sigma:+.3f}, "
          f"{boot_ci(d_flip, nprng)[1]/sigma:+.3f})")
    print(f"  strip_ez : {d_strip.mean()/sigma:+.2f} sigma  "
          f"(95% CI {boot_ci(d_strip, nprng)[0]/sigma:+.2f}, "
          f"{boot_ci(d_strip, nprng)[1]/sigma:+.2f})")
    print(f"  ratio    : {ratio:.0f}x  (95% CI {rlo:.0f}, {rhi:.0f})")
    print(f"  paired within draw; sigma from {N_DRAWS} independent draws.")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
