"""
Step 13: train the stereo correction term of Stereo-FCD.

    E_stereo(x) = E_chemnet(x) + g(x) * h(x),     g(x) = 1[ x contains '/' or '\' ]

ChemNet is untouched and frozen, so for any molecule without double-bond
stereochemistry the embedding is bit-identical to the original -- every FCD
value ever published on MOSES/GuacaMol remains exactly valid, by construction
rather than by empirical agreement (cf. Clean-FID, which matches legacy FID
only to ~2e-6).

h is trained so that the combined embedding predicts 3D shape descriptors of an
ETKDG conformer. Those descriptors are precisely what double-bond geometry
controls and what the frozen encoder provably cannot represent, so h learns the
geometric component ChemNet is missing rather than an arbitrary separation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import load_ref_model
import fcd.utils as U

HERE = Path(__file__).parent
TARGETS = HERE / "targets.jsonl"
CKPT = HERE / "stereo_head.pt"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BATCH = 64
PAD = 350
LR = 1e-3
LAMBDA_NORM = 1e-3          # keeps the correction from dominating the embedding

VOCAB = [v for k, v in vars(U).items() if k.endswith("__vocab") and isinstance(v, list)][0]
ORIG_W = len(VOCAB)
EXT_VOCAB = VOCAB + ["/", "\\"]
EXT_W = len(EXT_VOCAB)
EXT_C2I = {k: i for i, k in enumerate(EXT_VOCAB)}
UNK = EXT_C2I["X"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def orig_one_hot(smi):
    return U.get_one_hot(smi, pad_len=PAD) / ORIG_W


def ext_one_hot(smi):
    s = smi + "."
    oh = np.zeros((PAD, EXT_W), dtype=np.float32)
    for pos, tok in enumerate(U.tokenize(s)):
        if pos >= PAD:
            break
        oh[pos, EXT_C2I.get(tok, UNK)] = 1
    return oh / ORIG_W


def has_stereo(smi):
    return ("/" in smi) or ("\\" in smi)


class StereoHead(nn.Module):
    """h: extended one-hot -> 512-d additive correction."""

    def __init__(self, width=EXT_W, out=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(width, 48, 4, stride=2, padding=2), nn.SELU(),
            nn.Conv1d(48, 48, 4, stride=2, padding=2), nn.SELU(),
        )
        self.rnn = nn.LSTM(48, 192, batch_first=True)
        self.out = nn.Linear(192, out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)      # start as an exact no-op everywhere

    def forward(self, x):                  # x: (B, PAD, EXT_W)
        # true sequence length: padding rows are all-zero
        lens = (x.sum(-1) > 0).sum(1)                       # (B,)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)    # (B, T, C), T = PAD/4
        z, _ = self.rnn(z)
        # Read the state at the LAST REAL token. Taking z[:, -1] instead reads a
        # fixed point produced by ~250 steps of zero padding, which is identical
        # for the E and Z forms -- h(E) - h(Z) becomes exactly 0 and the paired
        # objective has no gradient at all. (ChemNet dodges this with Reverse.)
        idx = torch.clamp((lens + 3) // 4 - 1, min=0, max=z.shape[1] - 1)
        last = z[torch.arange(z.shape[0], device=z.device), idx]
        return self.out(last)


def load_data():
    rows = [json.loads(l) for l in TARGETS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"rows: {len(rows)}")
    smis = [r["smiles"] for r in rows]
    Y = np.array([r["desc"] for r in rows], dtype=np.float32)

    # robust standardisation (PMI values are heavy-tailed)
    Y = np.sign(Y) * np.log1p(np.abs(Y))
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    Y = (Y - mu) / sd

    # split by molecular skeleton so an E/Z pair never straddles the split
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    keys = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        keys.append(Chem.MolToSmiles(m, isomericSmiles=False) if m else s)
    uniq = sorted(set(keys))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    val_keys = set(uniq[:max(1, len(uniq) // 10)])
    val_idx = [i for i, k in enumerate(keys) if k in val_keys]
    trn_idx = [i for i, k in enumerate(keys) if k not in val_keys]
    print(f"  train {len(trn_idx)}   val {len(val_idx)}   (split on stereo-free skeleton)")
    return smis, Y, trn_idx, val_idx, (mu, sd)


def featurise(smis):
    ext = np.stack([ext_one_hot(s) for s in smis])
    org = np.stack([orig_one_hot(s) for s in smis]).astype(np.float32)
    g = np.array([1.0 if has_stereo(s) else 0.0 for s in smis], dtype=np.float32)
    return ext, org, g


def main():
    smis, Y, trn, val, _ = load_data()

    print("precomputing frozen ChemNet embeddings ...", flush=True)
    chemnet = load_ref_model().to(DEV).eval()
    for p in chemnet.parameters():
        p.requires_grad_(False)

    E = np.zeros((len(smis), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(smis), 256):
            _, org, _ = featurise(smis[i:i + 256])
            x = torch.tensor(org).transpose(1, 2).to(DEV)
            E[i:i + 256] = chemnet(x).cpu().numpy()
    print(f"  done. ||E|| mean = {np.linalg.norm(E, axis=1).mean():.3f}")

    head = StereoHead().to(DEV)
    lin = nn.Linear(512, Y.shape[1]).to(DEV)
    opt = torch.optim.Adam(list(head.parameters()) + list(lin.parameters()), lr=LR)

    Yt = torch.tensor(Y)
    Et = torch.tensor(E)

    def run_epoch(idx, train):
        head.train(train); lin.train(train)
        tot, n = 0.0, 0
        order = np.random.permutation(idx) if train else np.array(idx)
        for i in range(0, len(order), BATCH):
            sel = order[i:i + BATCH]
            ext, _, g = featurise([smis[j] for j in sel])
            xe = torch.tensor(ext).to(DEV)
            gg = torch.tensor(g).to(DEV).unsqueeze(1)
            e0 = Et[sel].to(DEV)
            y = Yt[sel].to(DEV)

            corr = head(xe) * gg
            pred = lin(e0 + corr)
            loss = ((pred - y) ** 2).mean() + LAMBDA_NORM * (corr ** 2).sum(1).mean()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sel); n += len(sel)
        return tot / n

    # baseline: frozen ChemNet alone (correction disabled)
    with torch.no_grad():
        lin0 = nn.Linear(512, Y.shape[1]).to(DEV)
        o0 = torch.optim.Adam(lin0.parameters(), lr=1e-2)
    for _ in range(60):
        sel = np.random.choice(trn, min(4096, len(trn)), replace=False)
        p = lin0(Et[sel].to(DEV)); l = ((p - Yt[sel].to(DEV)) ** 2).mean()
        o0.zero_grad(); l.backward(); o0.step()
    with torch.no_grad():
        base_val = float(((lin0(Et[val].to(DEV)) - Yt[val].to(DEV)) ** 2).mean())
    print(f"\nbaseline (frozen ChemNet only) val MSE = {base_val:.4f}")

    best = 1e9
    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(trn, True)
        with torch.no_grad():
            va = run_epoch(val, False)
        flag = ""
        if va < best:
            best = va
            torch.save({"head": head.state_dict()}, CKPT)
            flag = "  *saved"
        print(f"  epoch {ep:2d}  train {tr:.4f}   val {va:.4f}{flag}", flush=True)

    print(f"\nbest val MSE {best:.4f}  vs frozen-ChemNet baseline {base_val:.4f}"
          f"   ({100*(1-best/base_val):.1f}% reduction)")
    print(f"wrote {CKPT}")


if __name__ == "__main__":
    main()
