"""Re-extract stereo-bearing SMILES from the already-downloaded ChEMBL 37 chemreps.

Streams to disk (the earlier one-shot write_text produced a 0-byte file).
"""
import gzip
from pathlib import Path

HERE = Path(__file__).parent
GZ = HERE / "chembl_37_chemreps.txt.gz"
OUT = HERE / "chembl_stereo.smi"

n = n_stereo = 0
with gzip.open(GZ, "rt", encoding="utf-8", errors="ignore") as f, \
     open(OUT, "w", encoding="utf-8", newline="\n") as out:
    header = f.readline()
    print("header:", header.strip()[:150], flush=True)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        smi = parts[1]
        n += 1
        if "/" in smi or "\\" in smi:
            n_stereo += 1
            out.write(smi + "\n")
        if n % 500000 == 0:
            out.flush()
            print(f"  scanned {n:,}  stereo {n_stereo:,}", flush=True)

print(f"\ntotal molecules           : {n:,}", flush=True)
print(f"with '/' or '\\' in SMILES : {n_stereo:,}  ({100*n_stereo/n:.3f}%)", flush=True)
print(f"file size: {OUT.stat().st_size/1e6:.1f} MB", flush=True)
