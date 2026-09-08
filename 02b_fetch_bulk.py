"""Step 2 (bulk): download ChEMBL 37 chemreps and extract SMILES carrying E/Z stereo."""
import gzip, urllib.request, time
from pathlib import Path

HERE = Path(__file__).parent
GZ  = HERE / "chembl_37_chemreps.txt.gz"
URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz"

if not GZ.exists():
    print("downloading", URL)
    t0 = time.time()
    def hook(b, bs, total):
        if b % 500 == 0:
            mb = b * bs / 1e6
            print(f"  {mb:8.1f} MB / {total/1e6:8.1f} MB  ({time.time()-t0:5.0f}s)", flush=True)
    urllib.request.urlretrieve(URL, GZ, reporthook=hook)
    print("done in", round(time.time() - t0), "s  size", round(GZ.stat().st_size / 1e6, 1), "MB")
else:
    print("already have", GZ)

n = n_stereo = 0
stereo_lines = []
with gzip.open(GZ, "rt", encoding="utf-8", errors="ignore") as f:
    header = f.readline()
    print("header:", header.strip()[:120])
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        smi = parts[1]
        n += 1
        if "/" in smi or "\\" in smi:
            n_stereo += 1
            stereo_lines.append(smi)

print(f"\ntotal molecules            : {n}")
print(f"with '/' or '\\\\' in SMILES  : {n_stereo}  ({100*n_stereo/n:.3f}%)")
(HERE / "chembl_all_counts.txt").write_text(f"{n}\t{n_stereo}\n", encoding="utf-8")
(HERE / "chembl_stereo.smi").write_text("\n".join(stereo_lines), encoding="utf-8")
print("wrote chembl_stereo.smi")
