"""
Step 15b: rerun the headline FCD experiment (15_c2_full.py) with a CORRECT
E/Z inversion, to confirm the "FCD is blind to inversion" result is not an
artefact of an inversion that silently left some bonds unchanged.

Same pool, reference, embedding, noise floor, and bootstrap as 15_c2_full.py;
only the inversion function is replaced by invert_correct (step 34), which
toggles a neighbouring bond direction per stereo double bond and verifies the
CIP label actually flips.
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
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
ev = importlib.import_module("14_eval_stereofcd")
m34 = importlib.import_module("34_inversion_bug_scope")
invert_correct = m34.invert_correct
SEED, N_BOOT = 0, 1000


def frechet(x, y):
    return calculate_frechet_distance(x.mean(0), np.cov(x.T), y.mean(0), np.cov(y.T))


def main():
    rng = np.random.default_rng(SEED)
    d = json.loads(PRED.read_text(encoding="utf-8"))

    gen = []
    for r in d["records"]:
        if not r["valid"]:
            continue
        m = Chem.MolFromSmiles(r["pred"])
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if len(c) < 340:
            gen.append(c)
    inv = [invert_correct(s) for s in gen]
    inv = [s if s and len(s) < 340 else o for o, s in zip(gen, inv)]
    n_changed = sum(1 for a, b in zip(gen, inv) if a != b)
    print(f"usable generations {len(gen)}, of which {n_changed} "
          f"({100*n_changed/len(gen):.2f}%) change under CORRECT inversion")

    ref = ev.prep(HERE / "chebi20_test.txt")
    sfcd = ev.StereoFCD()
    print("embedding ...", flush=True)
    out = {"n_gen": len(gen), "n_changed": n_changed}
    for name, corr in [("FCD", False), ("Stereo-FCD", True)]:
        E_ref = sfcd.embed(ref, correction=corr)
        E_gen = sfcd.embed(gen, correction=corr)
        E_inv = sfcd.embed(inv, correction=corr)
        real = frechet(E_gen, E_ref)
        flip = frechet(E_inv, E_ref)
        delta = flip - real
        floor = []
        for _ in range(40):
            a = rng.choice(len(E_ref), len(E_gen), replace=False)
            floor.append(frechet(E_ref[a], E_ref))
        sd = float(np.std(floor, ddof=1))
        print(f"\n{name} (correct inversion): real {real:.6f}  inverted {flip:.6f}  "
              f"delta {delta:+.6e}  noise_sd {sd:.3e}  delta_in_sd {delta/sd:+.4f}")
        out[name] = {"real": real, "inverted": flip, "delta": delta,
                     "noise_sd": sd, "delta_in_sd": float(delta / sd)}
    # keep flat keys for the FCD row (figure reads these)
    out.update(out["FCD"])
    (HERE / "c2_full_correct.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
