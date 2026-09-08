"""
Step 13b: the diagnostic that matters.

Overall descriptor MSE is dominated by between-molecule variance, where the
frozen encoder is already competent. The quantity Stereo-FCD must capture is
the WITHIN-PAIR difference: given the same skeleton, how do the 3D descriptors
change between the E and the Z configuration?

Frozen ChemNet predicts that difference as EXACTLY zero, because it encodes the
two isomers identically. So any correlation above zero is signal that did not
exist before.
"""
import json
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import load_ref_model
import fcd.utils as U
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAD = 350
train_mod = import_module("13_train_stereo")
StereoHead = train_mod.StereoHead
VOCAB = [v for k, v in vars(U).items() if k.endswith("__vocab") and isinstance(v, list)][0]
ORIG_W = len(VOCAB)
EXT_VOCAB = VOCAB + ["/", "\\"]
EXT_C2I = {k: i for i, k in enumerate(EXT_VOCAB)}
UNK = EXT_C2I["X"]


def ext_one_hot(s):
    s = s + "."
    oh = np.zeros((PAD, len(EXT_VOCAB)), dtype=np.float32)
    for pos, tok in enumerate(U.tokenize(s)):
        if pos >= PAD:
            break
        oh[pos, EXT_C2I.get(tok, UNK)] = 1
    return oh / ORIG_W


def main():
    rows = [json.loads(l) for l in (HERE / "targets.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    Y = np.array([r["desc"] for r in rows], dtype=np.float32)
    Y = np.sign(Y) * np.log1p(np.abs(Y))
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    Y = (Y - mu) / sd

    # group into E/Z pairs by stereo-free skeleton
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        m = Chem.MolFromSmiles(r["smiles"])
        key = Chem.MolToSmiles(m, isomericSmiles=False) if m else r["smiles"]
        groups[key].append(i)
    pairs = [(v[0], v[1]) for v in groups.values() if len(v) == 2]
    print(f"complete E/Z pairs: {len(pairs)}")

    chemnet = load_ref_model().to(DEV).eval()
    head = StereoHead().to(DEV).eval()
    head.load_state_dict(torch.load(HERE / "stereo_head.pt", map_location=DEV)["head"])

    smis = [r["smiles"] for r in rows]

    @torch.no_grad()
    def embed(idx, correction):
        out = np.zeros((len(idx), 512), dtype=np.float32)
        for i in range(0, len(idx), 128):
            chunk = [smis[j] for j in idx[i:i + 128]]
            x0 = torch.tensor(np.stack([U.get_one_hot(s, pad_len=PAD) / ORIG_W for s in chunk]),
                              dtype=torch.float32).transpose(1, 2).to(DEV)
            e = chemnet(x0)
            if correction:
                xe = torch.tensor(np.stack([ext_one_hot(s) for s in chunk])).to(DEV)
                g = torch.tensor([1.0 if ("/" in s or "\\" in s) else 0.0 for s in chunk],
                                 device=DEV).unsqueeze(1)
                e = e + head(xe) * g
            out[i:i + 128] = e.cpu().numpy()
        return out

    ia = [p[0] for p in pairs]
    ib = [p[1] for p in pairs]

    # 1. are the two isomers now distinguishable at all?
    e0a, e0b = embed(ia, False), embed(ib, False)
    e1a, e1b = embed(ia, True), embed(ib, True)
    d0 = np.linalg.norm(e0a - e0b, axis=1)
    d1 = np.linalg.norm(e1a - e1b, axis=1)
    print(f"\nembedding distance between the E and Z form of the same molecule")
    print(f"  frozen ChemNet : mean {d0.mean():.6f}   max {d0.max():.6f}   "
          f"exactly zero for {int((d0==0).sum())}/{len(d0)}")
    print(f"  Stereo-FCD     : mean {d1.mean():.6f}   max {d1.max():.6f}   "
          f"exactly zero for {int((d1==0).sum())}/{len(d1)}")

    # 2. does the induced difference carry the right geometric information?
    #    fit a linear read-out on the within-pair difference, held out by skeleton
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pairs))
    cut = int(0.8 * len(pairs))
    tr, te = perm[:cut], perm[cut:]

    dY = Y[ia] - Y[ib]
    dE = e1a - e1b

    # least squares on train split, evaluate R^2 per descriptor on test
    Xtr, Ytr = dE[tr], dY[tr]
    W, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    pred = dE[te] @ W
    ss_res = ((dY[te] - pred) ** 2).sum(0)
    ss_tot = ((dY[te] - dY[te].mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)

    names = train_mod.__dict__.get("DESC_NAMES") or [
        "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
        "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"]
    print("\nwithin-pair R^2 on held-out skeletons")
    print("  (frozen ChemNet is identically 0 for every descriptor, by construction)")
    for n, v in zip(names, r2):
        print(f"    {n:22s} {v:+.4f}")
    print(f"    {'MEAN':22s} {r2.mean():+.4f}")

    (HERE / "within_pair.json").write_text(json.dumps({
        "n_pairs": len(pairs),
        "dist_frozen_mean": float(d0.mean()), "dist_stereo_mean": float(d1.mean()),
        "r2": {n: float(v) for n, v in zip(names, r2)}, "r2_mean": float(r2.mean()),
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
