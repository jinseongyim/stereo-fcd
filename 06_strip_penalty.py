"""
Step 6: the ZINC250k scenario.

Real setup: the reference set retains E/Z on a minority of molecules (ZINC250k
6.65%, ChEMBL 10.58%), while graph-based generators encode bond TYPE only and
emit nothing stereochemical. So the realistic failure is not an E<->Z flip
(same token length, activations bit-identical) but a STRIP (characters deleted,
sequence shortened, activations genuinely different).

Question: does FCD charge a meaningful penalty for a generator that has lost
100% of the stereochemistry the reference set carries?

Calibrated against the same two scales as step 5b: the FCD resampling noise
floor, and the graph-corruption sensitivity curve.
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
OUT = HERE / "strip_penalty_results.json"
SEED = 0
N = 10000
N_RESAMPLE = 6
PREVALENCES = [0.0665, 0.1058, 0.25, 0.50, 1.00]  # ZINC250k, ChEMBL, then stress


def strip_stereo(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, isomericSmiles=False)


def load_pool(path, limit, rng, keep_stereo=True):
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


def load_nonstereo(limit, rng):
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
            if len(out) >= limit:
                break
    rng.shuffle(out)
    return out


def build_mixed(stereo_pool, plain_pool, n, prevalence, offset_s, offset_p):
    k = int(round(prevalence * n))
    return stereo_pool[offset_s:offset_s + k] + plain_pool[offset_p:offset_p + (n - k)]


def main():
    rng = random.Random(SEED)
    model = load_ref_model()

    print("loading pools ...")
    stereo_pool = load_pool(STEREO, 6 * N, rng)
    plain_pool = load_nonstereo(6 * N, rng)
    print(f"  stereo pool {len(stereo_pool)}   plain pool {len(plain_pool)}")

    results = {}
    for prev in PREVALENCES:
        # reference and generator drawn from the same mixed distribution
        R = build_mixed(stereo_pool, plain_pool, N, prev, 0, 0)
        A = build_mixed(stereo_pool, plain_pool, N, prev, 2 * N, 2 * N)
        A_strip = [strip_stereo(s) for s in A]

        n_changed = sum(1 for a, b in zip(A, A_strip) if a != b)
        fcd_A = get_fcd(A, R, model=model)
        fcd_S = get_fcd(A_strip, R, model=model)

        # noise floor at this prevalence
        scores = []
        for i in range(N_RESAMPLE):
            off_s = rng.randrange(0, max(1, len(stereo_pool) - N))
            off_p = rng.randrange(0, max(1, len(plain_pool) - N))
            sub = build_mixed(stereo_pool, plain_pool, N, prev, off_s, off_p)
            scores.append(get_fcd(sub, R, model=model))
        sd = float(np.std(scores, ddof=1))

        d = abs(fcd_S - fcd_A)
        results[prev] = {
            "prevalence": prev, "n_molecules_changed_by_strip": n_changed,
            "fcd_with_stereo": fcd_A, "fcd_stripped": fcd_S,
            "delta": d, "noise_sd": sd, "delta_in_sd": d / sd if sd else None,
        }
        print(f"\nprevalence {prev:.4f}  ({n_changed}/{N} molecules altered by stripping)")
        print(f"  FCD(with stereo, R) = {fcd_A:.8f}")
        print(f"  FCD(stripped   , R) = {fcd_S:.8f}")
        print(f"  delta = {d:.3e}   noise sd = {sd:.3e}   = {d/sd if sd else float('nan'):.3f} sd")

    print("\n" + "=" * 70)
    print("PENALTY FCD CHARGES FOR LOSING ALL STEREOCHEMISTRY")
    print("=" * 70)
    print(f"{'prevalence':>12} {'delta FCD':>14} {'noise sd':>12} {'in sd units':>13}")
    for prev, r in results.items():
        print(f"{prev:>12.4f} {r['delta']:>14.3e} {r['noise_sd']:>12.3e} {r['delta_in_sd']:>13.3f}")

    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
