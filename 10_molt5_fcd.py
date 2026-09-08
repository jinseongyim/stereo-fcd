"""
Step 10: the demonstration on real model output.

MolT5-large emits E/Z on ~19% of its generations. So ask the question the
benchmark actually asks: if that model had produced the same molecules with
every double bond inverted, would its reported FCD change?
"""
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import load_ref_model
from fcd.fcd import get_predictions
from fcd.utils import calculate_frechet_distance


def acts(model, smis):
    # n_jobs=0 -> no DataLoader worker processes; the default of 1 hangs on Windows
    return get_predictions(model, smis, n_jobs=0, device="cuda")


def fcd_of(model, a, b_mu, b_sig):
    x = acts(model, a)
    return calculate_frechet_distance(np.mean(x, axis=0), np.cov(x.T), b_mu, b_sig), x

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PRED = HERE / "molt5_outputs_molt5-large-caption2smiles.json"
OUT = HERE / "molt5_fcd_results.json"

STEREO_SET = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
FLIP = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}


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


def read_ref():
    out = []
    with open(HERE / "chebi20_test.txt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                m = Chem.MolFromSmiles(p[1])
                if m is not None:
                    c = Chem.MolToSmiles(m)
                    if len(c) < 340:
                        out.append(c)
    return out


def main():
    d = json.loads(PRED.read_text(encoding="utf-8"))
    gen = []
    for r in d["records"]:
        if not r["valid"]:
            continue
        m = Chem.MolFromSmiles(r["pred"])
        c = Chem.MolToSmiles(m)
        if len(c) < 340:
            gen.append(c)
    print(f"valid MolT5-large generations: {len(gen)}")

    flipped, n_changed, n_bonds = [], 0, 0
    for s in gen:
        f, nb = flip_all(s)
        if f is None:
            flipped.append(s)
            continue
        flipped.append(f)
        n_bonds += nb
        if f != s:
            n_changed += 1
    print(f"generations made stereochemically wrong: {n_changed} "
          f"({100*n_changed/len(gen):.2f}%), {n_bonds} bonds inverted")

    ref = read_ref()
    print(f"reference (ChEBI-20 test ground truth): {len(ref)}")

    model = load_ref_model()
    print("computing reference activations ...", flush=True)
    ar = acts(model, ref)
    mu_r, sig_r = np.mean(ar, axis=0), np.cov(ar.T)

    print("computing generation activations ...", flush=True)
    fcd_real, a1 = fcd_of(model, gen, mu_r, sig_r)
    fcd_flip, a2 = fcd_of(model, flipped, mu_r, sig_r)
    n_ident = int((np.abs(a1 - a2).max(axis=1) == 0.0).sum())

    print("\n" + "=" * 66)
    print("MolT5-large as actually evaluated, vs. the same model with all")
    print("double-bond geometry inverted")
    print("=" * 66)
    print(f"  FCD(MolT5 output,          ChEBI-20 test) = {fcd_real:.6f}")
    print(f"  FCD(MolT5 output inverted, ChEBI-20 test) = {fcd_flip:.6f}")
    print(f"  difference                                = {abs(fcd_flip-fcd_real):.3e}")
    print(f"  molecules with identical ChemNet activation: {n_ident}/{len(gen)} "
          f"({100*n_ident/len(gen):.2f}%)")

    res = {"n_gen": len(gen), "n_changed": n_changed, "n_bonds": n_bonds,
           "fcd_real": fcd_real, "fcd_flipped": fcd_flip,
           "difference": abs(fcd_flip - fcd_real),
           "n_identical_activation": n_ident}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
