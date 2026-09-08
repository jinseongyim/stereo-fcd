"""
Step 8: is the residual FCD difference on ChEBI-20 stereochemical signal, or a
canonical-ordering artifact?

An E/Z flip cannot change the ChemNet input unless it changes the SMILES token
sequence, because '/' and '\' share one token. It can do that only by changing
RDKit's canonical atom ordering (same multiset of tokens, different order) or
the count of directional-bond characters.

So we partition the flipped pairs:
  A. token multiset identical  -> activation MUST be identical (pure re-order or no-op)
  B. token multiset different  -> the writer emitted a different number of '/'
                                  or '\' characters; this is the only channel
                                  through which any residual can flow

and report how much of the total FCD residual each explains.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_fcd, load_ref_model
from fcd.fcd import get_predictions
from fcd.utils import tokenize

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "residual_origin.json"

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


def read_smiles(path):
    out = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                out.append(p[1])
    return out


def prep(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if len(c) < 340:
            out.append(c)
    return out


def flip_all(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, 0
    bonds = [b for b in mol.GetBonds() if b.GetStereo() in STEREO_SET]
    if not bonds:
        return Chem.MolToSmiles(mol), 0
    for b in bonds:
        b.SetStereo(FLIP[b.GetStereo()])
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        return Chem.MolToSmiles(mol), len(bonds)
    except Exception:
        return None, len(bonds)


def main():
    T = prep(read_smiles(HERE / "chebi20_test.txt"))
    model = load_ref_model()

    pairs = []
    for s in T:
        f, nb = flip_all(s)
        pairs.append((s, f if f else s, nb))

    a_o = get_predictions(model, [p[0] for p in pairs])
    a_f = get_predictions(model, [p[1] for p in pairs])
    dmax = np.abs(a_o - a_f).max(axis=1)

    same_tok, diff_tok = [], []
    for i, (o, f, nb) in enumerate(pairs):
        if o == f:
            continue                      # no stereo, or flip was a no-op
        if Counter(tokenize(o)) == Counter(tokenize(f)):
            same_tok.append(i)
        else:
            diff_tok.append(i)

    changed = [i for i, (o, f, nb) in enumerate(pairs) if o != f]
    print(f"test molecules                 : {len(pairs)}")
    print(f"  actually flipped             : {len(changed)} ({100*len(changed)/len(pairs):.2f}%)")
    print(f"  same token multiset          : {len(same_tok)}")
    print(f"  different token multiset     : {len(diff_tok)}")

    def summarize(name, idx):
        if not idx:
            print(f"  {name}: none")
            return {"n": 0}
        d = dmax[idx]
        n_ident = int((d == 0.0).sum())
        print(f"  {name}: n={len(idx)}  activation identical {n_ident}/{len(idx)}"
              f"  max|delta| median={np.median(d):.3e} max={d.max():.3e}")
        return {"n": len(idx), "n_identical": n_ident,
                "median_maxabs": float(np.median(d)), "max_maxabs": float(d.max())}

    print("\nactivation differences by partition:")
    s_same = summarize("same-token   ", same_tok)
    s_diff = summarize("diff-token   ", diff_tok)

    # how much FCD residual survives if we drop the diff-token molecules?
    ref = prep(read_smiles(HERE / "chebi20_train.txt"))[:6486]
    keep = [i for i in range(len(pairs)) if i not in set(diff_tok)]
    T_keep = [pairs[i][0] for i in keep]
    F_keep = [pairs[i][1] for i in keep]
    fcd_t = get_fcd(T_keep, ref, model=model)
    fcd_f = get_fcd(F_keep, ref, model=model)
    print(f"\nFCD restricted to same-token molecules (n={len(keep)}):")
    print(f"  true    = {fcd_t:.8f}")
    print(f"  flipped = {fcd_f:.8f}")
    print(f"  diff    = {abs(fcd_f-fcd_t):.3e}")

    res = {"n_test": len(pairs), "n_flipped": len(changed),
           "same_token": s_same, "diff_token": s_diff,
           "fcd_same_token_only_true": fcd_t, "fcd_same_token_only_flipped": fcd_f,
           "fcd_same_token_only_diff": abs(fcd_f - fcd_t)}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
