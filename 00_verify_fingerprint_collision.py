"""
Step 0a: reproduce the claim that standard molecular similarity metrics
cannot distinguish canonical E/Z stereoisomers.

Claim under test: Morgan / MACCS / RDKit-fp Tanimoto give EXACTLY 1.0
for an E/Z pair that RDKit itself canonicalizes to different SMILES.

This is a precondition of the FCD-collapse candidate. If fingerprints
already separate them, the premise is wrong.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

# (name, E-isomer, Z-isomer) -- written with explicit / \ bond stereo
PAIRS = [
    ("2-butene",            r"C/C=C/C",              r"C/C=C\C"),
    ("stilbene",            r"c1ccccc1/C=C/c1ccccc1", r"c1ccccc1/C=C\c1ccccc1"),
    ("2-pentenoic acid",    r"CC/C=C/C(=O)O",         r"CC/C=C\C(=O)O"),
    ("cinnamaldehyde",      r"O=C/C=C/c1ccccc1",      r"O=C/C=C\c1ccccc1"),
    ("oleic-like C8",       r"CCC/C=C/CCC",           r"CCC/C=C\CCC"),
    ("chalcone-ish",        r"O=C(c1ccccc1)/C=C/c1ccccc1", r"O=C(c1ccccc1)/C=C\c1ccccc1"),
]

mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mgen_chiral = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=True
)


def report(name, smi_e, smi_z):
    me, mz = Chem.MolFromSmiles(smi_e), Chem.MolFromSmiles(smi_z)
    assert me is not None and mz is not None, name

    can_e, can_z = Chem.MolToSmiles(me), Chem.MolToSmiles(mz)
    distinct = can_e != can_z

    sims = {
        "Morgan(r2)":        DataStructs.TanimotoSimilarity(mgen.GetFingerprint(me),
                                                            mgen.GetFingerprint(mz)),
        "Morgan+chirality":  DataStructs.TanimotoSimilarity(mgen_chiral.GetFingerprint(me),
                                                            mgen_chiral.GetFingerprint(mz)),
        "MACCS":             DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(me),
                                                            MACCSkeys.GenMACCSKeys(mz)),
        "RDKit-fp":          DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(me),
                                                            Chem.RDKFingerprint(mz)),
    }

    # does RDKit actually perceive the stereo bond?
    stereo_e = [str(b.GetStereo()) for b in me.GetBonds()
                if b.GetStereo() != Chem.BondStereo.STEREONONE]
    stereo_z = [str(b.GetStereo()) for b in mz.GetBonds()
                if b.GetStereo() != Chem.BondStereo.STEREONONE]

    print(f"\n### {name}")
    print(f"  canonical E : {can_e}")
    print(f"  canonical Z : {can_z}")
    print(f"  distinct canonical SMILES : {distinct}")
    print(f"  perceived stereo E/Z      : {stereo_e} / {stereo_z}")
    for k, v in sims.items():
        flag = "  <-- COLLISION" if v == 1.0 else ""
        print(f"  Tanimoto {k:18s} = {v:.6f}{flag}")
    return distinct, sims


if __name__ == "__main__":
    print("RDKit stereo-collision check for standard similarity metrics")
    print("=" * 66)
    rows = [report(*p) for p in PAIRS]

    n = len(rows)
    print("\n" + "=" * 66)
    print("SUMMARY (collision = Tanimoto exactly 1.0 for an E/Z pair)")
    print(f"  pairs with distinct canonical SMILES : {sum(d for d, _ in rows)}/{n}")
    for key in ["Morgan(r2)", "Morgan+chirality", "MACCS", "RDKit-fp"]:
        c = sum(1 for _, s in rows if s[key] == 1.0)
        print(f"  {key:18s} collision rate : {c}/{n}")
