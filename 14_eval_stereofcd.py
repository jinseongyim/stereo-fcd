"""
Step 14: evaluate Stereo-FCD.

Four claims, in order of importance:

  C1 EXACT BACKWARD COMPATIBILITY. On molecules with no '/' or '\', Stereo-FCD
     equals FCD bit-for-bit. Checked on MOSES-like stereo-free molecules.

  C2 SENSITIVITY. Flipping the double-bond geometry of MolT5-large's actual
     output changed FCD by exactly 0. What does Stereo-FCD do?

  C3 MONOTONICITY. Corrupting a growing fraction of the stereo assignments
     should raise the score monotonically. A metric that merely *reacts* to
     stereo is not enough; it has to react in proportion to the damage.

  C4 CALIBRATION. The size of the stereo penalty should be commensurate with
     the metric's own noise floor and with penalties for graph-level corruption
     -- otherwise we have simply replaced blindness with hypersensitivity.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

# scipy dropped sqrtm's `disp` argument; fcd 1.2.2 still passes it.
import inspect as _inspect
import scipy.linalg as _sla
if "disp" not in _inspect.signature(_sla.sqrtm).parameters:
    _sqrtm_orig = _sla.sqrtm

    def _sqrtm_comp(A, disp=True, blocksize=64):
        r = _sqrtm_orig(A)
        return r if disp else (r, 0.0)

    _sla.sqrtm = _sqrtm_comp

from fcd import load_ref_model
from fcd.utils import calculate_frechet_distance
import fcd.utils as U
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
CKPT = HERE / "stereo_head.pt"
OUT = HERE / "stereofcd_eval.json"
PAD = 350
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

VOCAB = [v for k, v in vars(U).items() if k.endswith("__vocab") and isinstance(v, list)][0]
ORIG_W = len(VOCAB)
EXT_VOCAB = VOCAB + ["/", "\\"]
EXT_C2I = {k: i for i, k in enumerate(EXT_VOCAB)}
UNK = EXT_C2I["X"]

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}

from importlib import import_module
StereoHead = import_module("13_train_stereo").StereoHead


def orig_one_hot(s):
    return U.get_one_hot(s, pad_len=PAD) / ORIG_W


def ext_one_hot(s):
    s = s + "."
    oh = np.zeros((PAD, len(EXT_VOCAB)), dtype=np.float32)
    for pos, tok in enumerate(U.tokenize(s)):
        if pos >= PAD:
            break
        oh[pos, EXT_C2I.get(tok, UNK)] = 1
    return oh / ORIG_W


def has_stereo(s):
    return ("/" in s) or ("\\" in s)


class StereoFCD:
    def __init__(self):
        self.chemnet = load_ref_model().to(DEV).eval()
        self.head = StereoHead().to(DEV).eval()
        self.head.load_state_dict(torch.load(CKPT, map_location=DEV)["head"])

    @torch.no_grad()
    def embed(self, smis, batch=128, correction=True):
        out = np.zeros((len(smis), 512), dtype=np.float32)
        for i in range(0, len(smis), batch):
            chunk = smis[i:i + batch]
            x0 = torch.tensor(np.stack([orig_one_hot(s) for s in chunk]),
                              dtype=torch.float32).transpose(1, 2).to(DEV)
            e = self.chemnet(x0)
            if correction:
                xe = torch.tensor(np.stack([ext_one_hot(s) for s in chunk])).to(DEV)
                g = torch.tensor([1.0 if has_stereo(s) else 0.0 for s in chunk],
                                 device=DEV).unsqueeze(1)
                e = e + self.head(xe) * g
            out[i:i + batch] = e.cpu().numpy()
        return out

    def score(self, a, b, correction=True):
        x, y = self.embed(a, correction=correction), self.embed(b, correction=correction)
        return calculate_frechet_distance(x.mean(0), np.cov(x.T), y.mean(0), np.cov(y.T))


def corrupt(smis, frac, rng):
    """Randomly reassign E/Z on a fraction of the molecules."""
    out, n_hit = [], 0
    for s in smis:
        if rng.random() >= frac:
            out.append(s)
            continue
        mol = Chem.MolFromSmiles(s)
        bonds = [b for b in mol.GetBonds() if b.GetStereo() in STEREO_SET] if mol else []
        if not bonds:
            out.append(s)
            continue
        changed = False
        for b in bonds:
            if rng.random() < 0.5:
                b.SetStereo(FLIP[b.GetStereo()])
                changed = True
        if not changed:
            out.append(s)
            continue
        try:
            Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
            c = Chem.MolToSmiles(mol)
            out.append(c)
            n_hit += (c != s)
        except Exception:
            out.append(s)
    return out, n_hit


def prep(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                m = Chem.MolFromSmiles(p[1])
                if m is not None:
                    c = Chem.MolToSmiles(m)
                    if len(c) < 340:
                        out.append(c)
            if limit and len(out) >= limit:
                break
    return out


def main():
    rng = random.Random(SEED)
    sfcd = StereoFCD()
    res = {}

    ref = prep(HERE / "chebi20_test.txt")
    train = prep(HERE / "chebi20_train.txt", 8000)
    print(f"reference {len(ref)}   train pool {len(train)}")

    # ---- C1 exact backward compatibility -------------------------------
    free = [s for s in train if not has_stereo(s)][:2000]
    e_plain = sfcd.embed(free, correction=False)
    e_stereo = sfcd.embed(free, correction=True)
    identical = bool(np.array_equal(e_plain, e_stereo))
    print(f"\nC1 backward compatibility on {len(free)} stereo-free molecules")
    print(f"   embeddings bit-identical: {identical}   max|delta| = "
          f"{np.abs(e_plain-e_stereo).max():.3e}")
    res["C1_bit_identical"] = identical

    # ---- C2 sensitivity on real model output ---------------------------
    pred_file = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
    if pred_file.exists():
        d = json.loads(pred_file.read_text(encoding="utf-8"))
        gen = []
        for r in d["records"]:
            if r["valid"]:
                m = Chem.MolFromSmiles(r["pred"])
                c = Chem.MolToSmiles(m)
                if len(c) < 340:
                    gen.append(c)
        flipped, _ = corrupt(gen, 1.0, random.Random(1))
        # fully invert rather than randomise, to match step 10
        inv = []
        for s in gen:
            m = Chem.MolFromSmiles(s)
            bs = [b for b in m.GetBonds() if b.GetStereo() in STEREO_SET] if m else []
            if not bs:
                inv.append(s); continue
            for b in bs:
                b.SetStereo(FLIP[b.GetStereo()])
            try:
                Chem.AssignStereochemistry(m, cleanIt=False, force=True)
                inv.append(Chem.MolToSmiles(m))
            except Exception:
                inv.append(s)
        fcd_r = sfcd.score(gen, ref, correction=False)
        fcd_i = sfcd.score(inv, ref, correction=False)
        s_r = sfcd.score(gen, ref, correction=True)
        s_i = sfcd.score(inv, ref, correction=True)
        print(f"\nC2 MolT5-large output vs. the same output fully inverted (n={len(gen)})")
        print(f"   FCD        : {fcd_r:.6f} -> {fcd_i:.6f}   delta {abs(fcd_i-fcd_r):.3e}")
        print(f"   Stereo-FCD : {s_r:.6f} -> {s_i:.6f}   delta {abs(s_i-s_r):.3e}")
        res["C2"] = {"fcd": [fcd_r, fcd_i], "stereo_fcd": [s_r, s_i]}

    # ---- C3 monotonicity + C4 calibration ------------------------------
    # Draw from the full ChEMBL stereo pool, not just ChEBI-20's stereo subset.
    # With |A| ~ 900 the FCD resampling noise floor is 4e-2, forty times the
    # effect being measured, so no metric can show monotonicity at that size.
    chembl = HERE / "chembl_stereo.smi"
    if chembl.exists():
        raw = [s.strip() for s in chembl.read_text(encoding="utf-8").splitlines() if s.strip()]
        rng.shuffle(raw)
        pool = []
        for s in raw[:40000]:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            c = Chem.MolToSmiles(m)
            if len(c) < 340 and has_stereo(c):
                pool.append(c)
            if len(pool) >= 24000:
                break
    else:
        pool = [s for s in train if has_stereo(s)]
    rng.shuffle(pool)
    n = min(10000, len(pool) // 2)
    R, A = pool[:n], pool[n:2 * n]
    print(f"\nC3 corruption sweep (|R|={len(R)}, |A|={len(A)})")
    sweep = {}
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        C, hit = corrupt(A, frac, random.Random(100 + int(frac * 100)))
        f_ = sfcd.score(C, R, correction=False)
        s_ = sfcd.score(C, R, correction=True)
        sweep[frac] = {"fcd": f_, "stereo_fcd": s_, "n_changed": hit}
        print(f"   frac={frac:<5.2f} changed={hit:5d}   FCD {f_:.6f}   Stereo-FCD {s_:.6f}")
    res["C3_sweep"] = {str(k): v for k, v in sweep.items()}

    # noise floor for both metrics
    print("\nC4 noise floor (independent resamples, no corruption)")
    fs, ss = [], []
    for i in range(5):
        sub = random.Random(200 + i).sample(pool, len(A))
        fs.append(sfcd.score(sub, R, correction=False))
        ss.append(sfcd.score(sub, R, correction=True))
    print(f"   FCD        sd = {np.std(fs, ddof=1):.4e}")
    print(f"   Stereo-FCD sd = {np.std(ss, ddof=1):.4e}")
    res["C4_noise"] = {"fcd_sd": float(np.std(fs, ddof=1)),
                       "stereo_fcd_sd": float(np.std(ss, ddof=1))}

    d_f = abs(sweep[1.0]["fcd"] - sweep[0.0]["fcd"])
    d_s = abs(sweep[1.0]["stereo_fcd"] - sweep[0.0]["stereo_fcd"])
    print("\n" + "=" * 60)
    print(f"  full stereo corruption costs  FCD        {d_f:.3e}"
          f"  = {d_f/max(np.std(fs, ddof=1),1e-12):.2f} sd")
    print(f"                                Stereo-FCD {d_s:.3e}"
          f"  = {d_s/max(np.std(ss, ddof=1),1e-12):.2f} sd")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
