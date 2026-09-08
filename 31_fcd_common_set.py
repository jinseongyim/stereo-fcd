"""
Step 31: recompute MolT5-large vs BioT5-base FCD on a common caption set.

Table 3's FCD is over each model's valid output, so the caption sets and sample
counts differ (validity 0.946 vs 1.000). Frechet estimators have model-dependent
finite-sample bias (Chong & Forsyth, CVPR 2020), so this leaves a selection
confound in the FCD column that the stereo column no longer has.

Fix: restrict to captions where the reference and BOTH models produce an
FCD-eligible molecule (valid, canonical length < 340). Score both models against
the same reference subset, with the same sample count, same canonicalization and
length filter. Then paired-bootstrap over caption indices to get a CI on
  Delta = FCD(BioT5) - FCD(MolT5-large)
and the fraction of resamples in which MolT5-large is preferred (Delta > 0).
"""
import importlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd.utils import calculate_frechet_distance

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
ev = importlib.import_module("14_eval_stereofcd")
OUT = HERE / "fcd_common_set.json"

LARGE = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
BIOT5 = HERE / "molt5_outputs_biot5-base-text2mol.json"
TEST = HERE / "chebi20_test.txt"
N_BOOT = 2000
SEED = 0


def eligible_smiles(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    c = Chem.MolToSmiles(m)
    return c if len(c) < 340 else None


def load(path):
    return {r["cid"]: r for r in json.loads(path.read_text(encoding="utf-8"))["records"]}


def frechet(x, y):
    return calculate_frechet_distance(x.mean(0), np.cov(x.T), y.mean(0), np.cov(y.T))


def main():
    L, B = load(LARGE), load(BIOT5)
    cids = [c for c in L if c in B]

    rows = []
    for c in cids:
        gt = eligible_smiles(L[c]["gt"])
        pl = eligible_smiles(L[c]["pred"]) if L[c]["valid"] else None
        pb = eligible_smiles(B[c]["pred"]) if B[c]["valid"] else None
        if gt and pl and pb:
            rows.append((gt, pl, pb))
    print(f"captions where reference + both models are FCD-eligible: {len(rows)}")

    ref = [r[0] for r in rows]
    gl = [r[1] for r in rows]
    gb = [r[2] for r in rows]

    sfcd = ev.StereoFCD()
    E_ref = sfcd.embed(ref, correction=False)
    E_l = sfcd.embed(gl, correction=False)
    E_b = sfcd.embed(gb, correction=False)

    fcd_l = float(frechet(E_l, E_ref))
    fcd_b = float(frechet(E_b, E_ref))
    print(f"\ncommon-set FCD (n={len(rows)}, same references, same sample count)")
    print(f"  MolT5-large : {fcd_l:.4f}")
    print(f"  BioT5-base  : {fcd_b:.4f}")
    print(f"  Delta = FCD(BioT5) - FCD(MolT5-large) = {fcd_b - fcd_l:+.4f}")

    # paired bootstrap over caption indices
    rng = np.random.default_rng(SEED)
    n = len(rows)
    deltas = np.empty(N_BOOT)
    for t in range(N_BOOT):
        idx = rng.integers(0, n, n)
        er, el, eb = E_ref[idx], E_l[idx], E_b[idx]
        deltas[t] = frechet(eb, er) - frechet(el, er)
        if t % 50 == 0:
            print(f"    boot {t}/{N_BOOT}", flush=True)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    frac_large = float((deltas > 0).mean())
    print(f"\npaired bootstrap ({N_BOOT} resamples of caption index)")
    print(f"  Delta 95% CI : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(MolT5-large preferred, i.e. Delta>0) : {frac_large:.3f}")

    OUT.write_text(json.dumps({
        "n_common": len(rows),
        "fcd_molt5_large": fcd_l, "fcd_biot5_base": fcd_b,
        "delta_biot5_minus_large": fcd_b - fcd_l,
        "boot_ci95": [float(lo), float(hi)],
        "p_large_preferred": frac_large,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
