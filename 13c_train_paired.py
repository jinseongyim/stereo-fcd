"""
Step 13c: retrain the correction with a PAIRED objective.

Why the first attempt failed: h(x) sees the whole molecule, so it could lower
the descriptor loss by re-learning ordinary structure -- information the frozen
encoder already carries. The E/Z-dependent part is a tiny share of the total
variance, so it was simply ignored, and h(E) came out equal to h(Z).

The fix is to make the objective *only* the part ChemNet cannot represent.
Since ChemNet embeds the two isomers identically, the model's entire ability to
tell them apart is W(h(E) - h(Z)). So train that difference directly against the
measured difference in 3D descriptors:

    L = || W (h(E) - h(Z)) - (y_E - y_Z) ||^2  +  lambda (||h(E)||^2 + ||h(Z)||^2)

The regulariser keeps the correction small relative to the ChemNet embedding, so
the metric stays close to FCD rather than being dominated by the new term.
"""
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

import fcd.utils as U
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
CKPT = HERE / "stereo_head.pt"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
BATCH = 64
PAD = 350
LR = 2e-3
LAMBDA = 1e-6

StereoHead = import_module("13_train_stereo").StereoHead
VOCAB = [v for k, v in vars(U).items() if k.endswith("__vocab") and isinstance(v, list)][0]
ORIG_W = len(VOCAB)
EXT_VOCAB = VOCAB + ["/", "\\"]
EXT_C2I = {k: i for i, k in enumerate(EXT_VOCAB)}
UNK = EXT_C2I["X"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DESC = ["Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
        "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"]


def ext_one_hot(s):
    s = s + "."
    oh = np.zeros((PAD, len(EXT_VOCAB)), dtype=np.float32)
    for pos, tok in enumerate(U.tokenize(s)):
        if pos >= PAD:
            break
        oh[pos, EXT_C2I.get(tok, UNK)] = 1
    return oh / ORIG_W


def main():
    src = HERE / "targets_avg.jsonl"
    if not src.exists():
        src = HERE / "targets.jsonl"
        print("WARNING: conformer-averaged targets missing, falling back to the "
              "single-conformer file (SNR 1.47 -- expect collapse)")
    rows = [json.loads(l) for l in src.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    print(f"target file: {src.name}")
    Y = np.array([r["desc"] for r in rows], dtype=np.float32)
    Y = np.sign(Y) * np.log1p(np.abs(Y))
    scale = Y.std(0) + 1e-8
    Y = (Y - Y.mean(0)) / scale
    smis = [r["smiles"] for r in rows]

    # per-descriptor weighting: down-weight descriptors whose conformer spread is
    # large relative to their spread across molecules, i.e. mostly sampling noise
    if "sd" in rows[0]:
        SD = np.array([r["sd"] for r in rows], dtype=np.float32)
        noise = (SD.mean(0) / np.sqrt(10)) / scale        # K=10 conformers averaged
        Wt = 1.0 / (1.0 + noise ** 2)
        print("  descriptor weights (1 = clean, 0 = pure noise):")
        for n, w, nz in zip(DESC, Wt, noise):
            print(f"    {n:22s} w={w:.3f}   rel.noise={nz:.3f}")
    else:
        Wt = np.ones(Y.shape[1], dtype=np.float32)

    groups = defaultdict(list)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        groups[Chem.MolToSmiles(m, isomericSmiles=False) if m else s].append(i)
    pairs = [(v[0], v[1]) for v in groups.values() if len(v) == 2]
    print(f"E/Z pairs: {len(pairs)}")

    dY = Y[[p[0] for p in pairs]] - Y[[p[1] for p in pairs]]
    print(f"  within-pair |dY| mean = {np.abs(dY).mean():.4f}  "
          f"(std of the signal we are trying to predict)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pairs))
    cut = int(0.85 * len(pairs))
    tr, te = perm[:cut], perm[cut:]
    print(f"  train {len(tr)}   held-out {len(te)}  (split by skeleton)")

    # precompute one-hots once; 20k x 350 x 37 float32 ~ 1 GB, so store as uint8 index
    print("precomputing token indices ...", flush=True)
    # -1 marks padding. Filling padding with UNK instead leaves ~250 identical
    # 'X' tokens at the tail, the LSTM saturates to the same fixed point for both
    # isomers, and h(E) - h(Z) is exactly 0 -- no gradient, loss frozen from
    # epoch 1. Padding must be the all-zero vector, as in fcd's get_one_hot.
    IDX = np.full((len(smis), PAD), -1, dtype=np.int16)
    for i, s in enumerate(smis):
        for pos, tok in enumerate(U.tokenize(s + ".")):
            if pos >= PAD:
                break
            IDX[i, pos] = EXT_C2I.get(tok, UNK)

    def batch_oh(idx):
        oh = np.zeros((len(idx), PAD, len(EXT_VOCAB)), dtype=np.float32)
        sub = IDX[idx]
        r, c = np.nonzero(sub >= 0)
        oh[r, c, sub[r, c]] = 1.0
        return torch.tensor(oh / ORIG_W)

    head = StereoHead().to(DEV)
    W = nn.Linear(512, Y.shape[1], bias=False).to(DEV)
    opt = torch.optim.Adam(list(head.parameters()) + list(W.parameters()), lr=LR)
    dYt = torch.tensor(dY)
    Wtt = torch.tensor(Wt, device=DEV)

    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])

    def run(split, train):
        head.train(train); W.train(train)
        tot, n = 0.0, 0
        order = np.random.permutation(split) if train else split
        for i in range(0, len(order), BATCH):
            sel = order[i:i + BATCH]
            xa = batch_oh(ia[sel]).to(DEV)
            xb = batch_oh(ib[sel]).to(DEV)
            ha, hb = head(xa), head(xb)
            pred = W(ha - hb)
            y = dYt[sel].to(DEV)
            loss = (Wtt * (pred - y) ** 2).mean() \
                + LAMBDA * ((ha ** 2).sum(1) + (hb ** 2).sum(1)).mean()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sel); n += len(sel)
        return tot / n

    # trivial baseline: predicting zero difference, which is what FCD does
    zero_mse = float((Wtt.cpu() * dYt[te] ** 2).mean())
    print(f"\nbaseline: predicting zero difference (= frozen ChemNet) MSE = {zero_mse:.4f}")

    best = 1e9
    for ep in range(1, EPOCHS + 1):
        trl = run(tr, True)
        with torch.no_grad():
            tel = run(te, False)
        flag = ""
        if tel < best:
            best = tel
            torch.save({"head": head.state_dict()}, CKPT)
            flag = "  *saved"
        print(f"  epoch {ep:2d}  train {trl:.4f}   held-out {tel:.4f}{flag}", flush=True)

    print(f"\nheld-out MSE {best:.4f} vs zero-prediction {zero_mse:.4f} "
          f"-> {100*(1-best/zero_mse):.1f}% of the within-pair signal captured")
    print(f"wrote {CKPT}")


if __name__ == "__main__":
    main()
