"""
Step 0b: reproduce the ChemNet / FCD stereochemical collapse at the source.

Three levels of evidence, weakest to strongest:
  L1 tokenizer  : do '/' and '\\' map to the same index?
  L2 one-hot    : are the encoded tensors bit-identical for an E/Z pair?
  L3 activation : is the 512-d ChemNet activation difference exactly 0?

Also checks what ELSE the vocab silently drops (tetrahedral @, ring closure 9, ...)
so we do not overclaim that stereo is the only casualty.
"""
import numpy as np
from rdkit import Chem, RDLogger

# fcd 1.2.2 calls np.row_stack, removed in numpy 2.x
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_fcd, load_ref_model
from fcd.fcd import get_predictions
from fcd.utils import get_one_hot, tokenize
import fcd.utils as U

RDLogger.DisableLog("rdApp.*")

VOCAB = U._utils__vocab if hasattr(U, "_utils__vocab") else getattr(U, "_U__vocab", None)
# name mangling: module-level __vocab -> _<nothing>; access via module dict
VOCAB = [v for k, v in vars(U).items() if k.endswith("__vocab") and isinstance(v, list)][0]
C2I = [v for k, v in vars(U).items() if k.endswith("__vocab_c2i")][0]
UNK = [v for k, v in vars(U).items() if k.endswith("__unk")][0]

PAIRS = [
    ("2-butene",         r"C/C=C/C",                    r"C/C=C\C"),
    ("stilbene",         r"c1ccccc1/C=C/c1ccccc1",      r"c1ccccc1/C=C\c1ccccc1"),
    ("2-pentenoic acid", r"CC/C=C/C(=O)O",              r"CC/C=C\C(=O)O"),
    ("cinnamaldehyde",   r"O=C/C=C/c1ccccc1",           r"O=C/C=C\c1ccccc1"),
    ("oleic-like C8",    r"CCC/C=C/CCC",                r"CCC/C=C\CCC"),
    ("chalcone-ish",     r"O=C(/C=C/c1ccccc1)c1ccccc1", r"O=C(/C=C\c1ccccc1)c1ccccc1"),
]

# tetrahedral chirality control: @ IS in the vocab, @@ tokenizes as two @
CHIRAL_PAIRS = [
    ("alanine",   "C[C@H](N)C(=O)O",  "C[C@@H](N)C(=O)O"),
    ("2-butanol", "CC[C@H](C)O",      "CC[C@@H](C)O"),
]


def canon(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def l1_tokenizer():
    print("\n[L1] tokenizer / vocabulary")
    print(f"  vocab size = {len(VOCAB)}, unk index = {UNK} (symbol {VOCAB[UNK]!r})")
    for ch in ["/", "\\", "@", "9", "%", "Se", "B", "b", "p"]:
        idx = C2I.get(ch, UNK)
        status = "IN VOCAB" if ch in C2I else f"-> UNK({UNK})"
        print(f"  {ch!r:6s} {status}")
    same = C2I.get("/", UNK) == C2I.get("\\", UNK)
    print(f"  '/' and '\\' collide : {same}")
    return same


def l2_onehot():
    print("\n[L2] one-hot encoding (on RDKit-canonical SMILES, as FCD is normally used)")
    results = []
    for name, e, z in PAIRS:
        ce, cz = canon(e), canon(z)
        oe, oz = get_one_hot(ce, pad_len=350), get_one_hot(cz, pad_len=350)
        identical = np.array_equal(oe, oz)
        results.append(identical)
        print(f"  {name:18s} canon differ={ce != cz}  one-hot identical={identical}"
              + ("   <-- COLLAPSE" if identical else ""))
    return results


def l3_activation(model):
    print("\n[L3] ChemNet 512-d activations")
    rows = []
    for name, e, z in PAIRS:
        ce, cz = canon(e), canon(z)
        acts = get_predictions(model, [ce, cz], device="cpu")
        d = float(np.abs(acts[0] - acts[1]).max())
        rows.append(d)
        print(f"  {name:18s} max|Δactivation| = {d:.3e}"
              + ("   <-- EXACTLY ZERO" if d == 0.0 else ""))
    return rows


def l3_chiral_control(model):
    print("\n[L3-control] tetrahedral chirality (@ vs @@) -- expected NOT to collapse")
    for name, a, b in CHIRAL_PAIRS:
        ca, cb = canon(a), canon(b)
        acts = get_predictions(model, [ca, cb], device="cpu")
        d = float(np.abs(acts[0] - acts[1]).max())
        print(f"  {name:12s} canon differ={ca != cb}  max|Δ| = {d:.3e}"
              + ("   <-- ALSO COLLAPSES" if d == 0.0 else ""))


def l4_set_level_fcd(model):
    """The metric-level consequence: FCD between an all-E set and its all-Z flip."""
    print("\n[L4] set-level FCD: all-E set vs same set with every double bond flipped to Z")
    e_set = [canon(e) for _, e, _ in PAIRS] * 20
    z_set = [canon(z) for _, _, z in PAIRS] * 20
    score = get_fcd(e_set, z_set, model=model)
    print(f"  FCD(all-E, all-Z) = {score:.6e}"
          + ("   <-- ZERO: metric cannot see the flip" if abs(score) < 1e-8 else ""))
    return score


if __name__ == "__main__":
    print("=" * 70)
    print("ChemNet / FCD stereochemical collapse -- source-level verification")
    print("=" * 70)
    collide = l1_tokenizer()
    oh = l2_onehot()
    model = load_ref_model()
    acts = l3_activation(model)
    l3_chiral_control(model)
    fcd_score = l4_set_level_fcd(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  '/' == '\\' at tokenizer      : {collide}")
    print(f"  one-hot identical            : {sum(oh)}/{len(oh)}")
    print(f"  activation diff exactly 0    : {sum(1 for d in acts if d == 0.0)}/{len(acts)}")
    print(f"  set-level FCD(E-set, Z-set)  : {fcd_score:.6e}")
