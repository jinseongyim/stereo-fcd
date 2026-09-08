"""
Step 37: the deficiency on named medicines, with structures taken from PubChem.

Reviewer 2 asked for "specific examples with actual chemical structures", and the
editor asked for "one or two practical examples where this deficiency really
matters ... through the real chemical structures and models". Sections 1-3 argue
from tokenizer internals and aggregate statistics; this script argues from
medicines whose double-bond configuration is the difference between a drug and a
much weaker or differently acting compound.

No SMILES is written by hand. Every structure is fetched from PubChem by name,
and each pair is validated structurally before it is measured:

  - both members resolve to a PubChem record
  - they share a connectivity (flat) SMILES  -> same constitution
  - their stereo SMILES differ               -> a genuine stereoisomer pair
  - for the E/Z set, RDKit assigns double-bond stereo and the labels differ
  - for the control set, RDKit assigns tetrahedral stereo and no double-bond stereo

Only pairs that pass are measured. This matters because an earlier version of
this script used hand-written SMILES and two of the five pairs were wrong: one
had its CIP labels reversed relative to its name, and one differed at two double
bonds rather than one. Editing a direction character silently changes a
neighbouring bond -- the same failure mode documented in Section S7.

Measurements, per pair, under two SMILES writers:
  canonical    Chem.MolToSmiles(mol)                  -- atom order may change
  order-fixed  Chem.MolToSmiles(mol, canonical=False) -- input atom order kept
reporting whether the one-hot encodings are bit-identical and the maximum
absolute difference of the 512-d ChemNet activations.
"""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

# fcd 1.2.2 calls np.row_stack, removed in numpy 2.x
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from fcd import get_predictions, load_ref_model
from fcd.utils import get_one_hot

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
OUT = HERE / "drug_case_studies.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; stereo-fcd-case-studies/1.0)"}
PROPS = "Title,SMILES,ConnectivitySMILES,InChIKey"
URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/" + PROPS + "/JSON"

SS = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
      Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}

# label, [name aliases for member A], [aliases for member B], why the configuration matters
EZ_PAIRS = [
    ("tamoxifen",
     ["tamoxifen", "(Z)-tamoxifen"],
     ["isotamoxifen", "(E)-tamoxifen"],
     "Z is the marketed anti-oestrogen; the E isomer has a different receptor profile."),
    ("retinoic acid",
     ["tretinoin"],
     ["isotretinoin"],
     "Two separately approved medicines differing in double-bond configuration."),
    ("combretastatin A-4",
     ["combretastatin A-4", "(Z)-combretastatin A-4"],
     ["(E)-combretastatin A-4", "trans-combretastatin A-4"],
     "Z inhibits tubulin polymerisation potently; E is far weaker."),
    ("clomifene",
     ["zuclomiphene", "zuclomifene"],
     ["enclomiphene", "enclomifene"],
     "The two isomers of the marketed mixture differ in oestrogen-receptor action."),
]

CHIRAL_PAIRS = [
    ("omeprazole",
     ["esomeprazole"],
     ["(R)-omeprazole", "dexomeprazole"],
     "Single-enantiomer proton-pump inhibitor versus its antipode."),
    ("citalopram",
     ["escitalopram"],
     ["(R)-citalopram"],
     "The S enantiomer is the marketed antidepressant."),
    ("naproxen",
     ["naproxen"],
     ["(R)-naproxen", "R-naproxen"],
     "The S enantiomer is the marketed anti-inflammatory; the R form is hepatotoxic."),
    ("ofloxacin",
     ["levofloxacin"],
     ["(R)-ofloxacin", "dextrofloxacin"],
     "Levofloxacin is the active S enantiomer of the ofloxacin racemate."),
]


def fetch(aliases):
    """Try each alias; return the first PubChem record that resolves."""
    for a in aliases:
        try:
            q = urllib.request.quote(a)
            req = urllib.request.Request(URL.format(q), headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                p = json.loads(r.read().decode())["PropertyTable"]["Properties"][0]
            p["queried_as"] = a
            time.sleep(0.35)
            return p
        except Exception:
            time.sleep(0.35)
    return None


def cip(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, [], 0
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    bond = [str(b.GetStereo()).replace("BondStereo.STEREO", "")
            for b in mol.GetBonds() if b.GetStereo() in SS]
    atom = sum(1 for a in mol.GetAtoms()
               if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    return mol, bond, atom


def measure(model, mol_a, mol_b, canonical):
    sa = Chem.MolToSmiles(mol_a, canonical=canonical)
    sb = Chem.MolToSmiles(mol_b, canonical=canonical)
    oa, ob = get_one_hot(sa, pad_len=350), get_one_hot(sb, pad_len=350)
    acts = get_predictions(model, [sa, sb], device="cpu")
    return {"smiles_a": sa, "smiles_b": sb, "strings_differ": sa != sb,
            "onehot_identical": bool(np.array_equal(oa, ob)),
            "max_abs_delta": float(np.abs(acts[0] - acts[1]).max())}


def run(model, title, pairs, kind):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)
    rows = []
    for key, alias_a, alias_b, why in pairs:
        pa, pb = fetch(alias_a), fetch(alias_b)
        rec = {"pair": key, "kind": kind, "why": why}
        print(f"\n{key}")
        if pa is None or pb is None:
            miss = alias_a if pa is None else alias_b
            print(f"  SKIP -- PubChem did not resolve {miss}")
            rec["status"] = "unresolved"
            rows.append(rec)
            continue

        mol_a, bond_a, atom_a = cip(pa["SMILES"])
        mol_b, bond_b, atom_b = cip(pb["SMILES"])
        for tag, p, bl, at in (("A", pa, bond_a, atom_a), ("B", pb, bond_b, atom_b)):
            print(f"  {tag}: {p['Title']}  (CID {p['CID']}, queried as {p['queried_as']!r})")
            print(f"     SMILES  {p['SMILES']}")
            print(f"     stereo  double-bond {bl or '(none)'} | tetrahedral centres {at}")

        # structural validation
        same_const = pa["ConnectivitySMILES"] == pb["ConnectivitySMILES"]
        diff_stereo = pa["SMILES"] != pb["SMILES"]
        checks = {"same_constitution": same_const, "stereo_differs": diff_stereo}
        if kind == "ez":
            checks["has_double_bond_stereo"] = bool(bond_a) and bool(bond_b)
            checks["labels_differ"] = bond_a != bond_b
        else:
            checks["has_tetrahedral"] = atom_a > 0 and atom_b > 0
            checks["no_double_bond_stereo"] = not bond_a and not bond_b
        ok = all(checks.values())
        print(f"  validation: {checks}  ->  {'PASS' if ok else 'FAIL'}")
        rec.update({"a": pa, "b": pb, "bond_a": bond_a, "bond_b": bond_b,
                    "atoms_a": atom_a, "atoms_b": atom_b,
                    "checks": checks, "validated": ok})

        if not ok:
            print("  not measured (failed validation)")
            rec["status"] = "failed_validation"
            rows.append(rec)
            continue

        rec["status"] = "measured"
        for canonical, t in ((True, "canonical"), (False, "order_fixed")):
            m = measure(model, mol_a, mol_b, canonical)
            rec[t] = m
            d = m["max_abs_delta"]
            if kind == "ez":
                v = "  <-- EXACTLY ZERO" if d == 0.0 else ""
            else:
                v = "  <-- COLLAPSES (unexpected)" if d == 0.0 else "  <-- separated"
            print(f"    [{t:11s}] one-hot identical={str(m['onehot_identical']):5s} "
                  f"max|d|={d:.3e}{v}")
        rows.append(rec)
    return rows


if __name__ == "__main__":
    print("=" * 84)
    print("Named medicines: does the FCD encoder see the configuration that makes the drug?")
    print("Structures fetched from PubChem; nothing hand-written.")
    print("=" * 84)
    model = load_ref_model()
    ez = run(model, "A. E/Z pairs -- configuration determines the pharmacology", EZ_PAIRS, "ez")
    ch = run(model, "B. Tetrahedral controls -- '@' is in the ChemNet vocabulary",
             CHIRAL_PAIRS, "chiral")

    m_ez = [r for r in ez if r.get("status") == "measured"]
    m_ch = [r for r in ch if r.get("status") == "measured"]
    print("\n" + "=" * 84)
    print("SUMMARY")
    print(f"  E/Z pairs validated and measured : {len(m_ez)}/{len(ez)}")
    for t in ("order_fixed", "canonical"):
        n = sum(1 for r in m_ez if r[t]["max_abs_delta"] == 0.0)
        print(f"    {t:11s} activation exactly 0 : {n}/{len(m_ez)}")
    print(f"  tetrahedral controls measured    : {len(m_ch)}/{len(ch)}")
    n = sum(1 for r in m_ch if r["order_fixed"]["max_abs_delta"] > 0.0)
    print(f"    separated                      : {n}/{len(m_ch)}")

    OUT.write_text(json.dumps({"ez_pairs": ez, "chiral_controls": ch}, indent=2))
    print(f"\nwrote {OUT.name}")
