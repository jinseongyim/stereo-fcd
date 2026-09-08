"""
Step 11: is a backward-compatible vocabulary extension of ChemNet possible?

Design under test. The encoder is not inherently stereo-blind -- '@' is in the
vocabulary and tetrahedral chirality survives. Only two characters are missing.
So extend the input alphabet 35 -> 37, giving '/' and '\' their own channels,
and initialise the two new input weights to ZERO.

Two properties must hold for this to be a drop-in replacement:

  P1 EXACT BACKWARD COMPATIBILITY. For any molecule containing no '/' or '\',
     the extended model must produce bit-identical activations. If so, every
     FCD number ever published on MOSES/GuacaMol (which contain zero
     stereochemistry) remains exactly valid.

  P2 A LIVE CHANNEL. For molecules that do carry E/Z, the encoding must now
     differ from the original -- i.e. the information reaches the network
     instead of being collapsed onto the unknown token.

Note the normalisation trap: SmilesDataset returns `features / features.shape[1]`,
so widening the alphabet silently rescales EVERY channel by 35/37 and would
destroy P1. We keep dividing by the original width.

At zero-init, E and Z are still tied to each other (both new channels are dead);
that is the honest starting point, not the finished metric. This script only
establishes that the surgery is safe and that the channel exists to be trained.
"""
import numpy as np
import torch
from rdkit import Chem, RDLogger

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import load_ref_model
import fcd.utils as U

RDLogger.DisableLog("rdApp.*")

VOCAB = [v for k, v in vars(U).items() if k.endswith("__vocab") and isinstance(v, list)][0]
ORIG_W = len(VOCAB)                      # 35
NEW_TOKENS = ["/", "\\"]
EXT_VOCAB = VOCAB + NEW_TOKENS
EXT_W = len(EXT_VOCAB)                   # 37
EXT_C2I = {k: i for i, k in enumerate(EXT_VOCAB)}
UNK = EXT_C2I["X"]

STEREO_FREE = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
               "C[C@H](N)C(=O)O", "CC[C@@H](C)O"]          # includes tetrahedral '@'
STEREO = [r"C/C=C/C", r"C/C=C\C", r"CC/C=C/C(=O)O", r"O=C/C=C\c1ccccc1"]


def ext_one_hot(smiles, pad_len=350):
    """Same as fcd.utils.get_one_hot but over the extended alphabet, and
    deliberately still dividing by the ORIGINAL width."""
    smiles = smiles + "."
    oh = np.zeros((pad_len, EXT_W))
    for pos, tok in enumerate(U.tokenize(smiles)):
        oh[pos, EXT_C2I.get(tok, UNK)] = 1
    return oh


def widen(model):
    """Return a copy of ChemNet whose first Conv1d accepts EXT_W channels, with
    the two new input weights zeroed."""
    import copy
    m = copy.deepcopy(model)
    conv = None
    for layer in m:
        if isinstance(layer, torch.nn.Conv1d):
            conv = layer
            break
    assert conv is not None, "no Conv1d found"
    assert conv.in_channels == ORIG_W, f"expected {ORIG_W} in-channels, got {conv.in_channels}"

    new = torch.nn.Conv1d(EXT_W, conv.out_channels, conv.kernel_size,
                          stride=conv.stride, padding=conv.padding,
                          bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :ORIG_W, :] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)

    for i, layer in enumerate(m):
        if layer is conv:
            m[i] = new
            break
    return m


def run(model, one_hots):
    x = torch.tensor(np.stack(one_hots), dtype=torch.float32).transpose(1, 2)
    with torch.no_grad():
        return model(x).numpy()


def main():
    print(f"original alphabet width {ORIG_W} -> extended {EXT_W}")
    print(f"new tokens: {NEW_TOKENS}\n")

    base = load_ref_model()
    ext = widen(base)

    # --- P1: exact backward compatibility on stereo-free molecules ---
    a_base = run(base, [U.get_one_hot(s, pad_len=350) / ORIG_W for s in STEREO_FREE])
    a_ext = run(ext, [ext_one_hot(s) / ORIG_W for s in STEREO_FREE])
    d = np.abs(a_base - a_ext)
    print("P1  exact backward compatibility (molecules with no '/' or '\\')")
    for s, dd in zip(STEREO_FREE, d.max(axis=1)):
        print(f"      max|delta| = {dd:.3e}   {s}")
    p1 = bool((d == 0).all())
    print(f"    -> bit-identical for all: {p1}\n")

    # --- P2: the stereo characters now reach a live channel ---
    a_base_s = run(base, [U.get_one_hot(s, pad_len=350) / ORIG_W for s in STEREO])
    a_ext_s = run(ext, [ext_one_hot(s) / ORIG_W for s in STEREO])
    ds = np.abs(a_base_s - a_ext_s).max(axis=1)
    print("P2  encoding of E/Z-bearing molecules changes vs. the original")
    for s, dd in zip(STEREO, ds):
        print(f"      max|delta| = {dd:.3e}   {s}")
    print(f"    -> all changed: {bool((ds > 0).all())}\n")

    # --- honest limitation: at zero-init E and Z are still tied ---
    ez = np.abs(a_ext_s[0] - a_ext_s[1]).max()
    print("LIMIT  at zero-init the two new channels are dead, so E vs Z is still tied")
    print(f"      max|delta| between C/C=C/C and C/C=C\\C = {ez:.3e}")
    print("      -> training these two channels is what buys sensitivity;")
    print("         the surgery only proves it can be done without breaking P1.")


if __name__ == "__main__":
    main()
