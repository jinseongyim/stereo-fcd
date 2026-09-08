"""
Step 20: does MOSES / GuacaMol contain any stereochemistry at all?

The note claims (§3.1) that this defect is unreachable on the two standard
distribution-learning benchmarks because their data carries no stereocentres.
That claim was asserted from an earlier session with no surviving artefact, so
it is re-measured here from the published files.

Counting is a pure string test -- '/' and '\\' for double-bond geometry, '@'
for tetrahedral chirality -- so no RDKit is needed and nothing depends on
parser behaviour.

Sanity check: the line counts must match the published dataset sizes, otherwise
we downloaded the wrong file and the count means nothing.
  MOSES dataset_v1.csv : 1,936,962 molecules
  GuacaMol v1 all      : 1,591,378 molecules
"""
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "moses_guacamol_stereo.json"

SOURCES = {
    "moses": {
        "url": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv",
        "fname": "moses_dataset_v1.csv",
        "expected_n": 1936962,
        "kind": "csv",
    },
    "guacamol_all": {
        "url": "https://ndownloader.figshare.com/files/13612745",
        "fname": "guacamol_v1_all.smiles",
        "expected_n": 1591378,
        "kind": "smiles",
    },
}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r    {total/1e6:.1f} MB", end="", flush=True)
    print(f"\r  saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")


def smiles_iter(path, kind):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if kind == "csv":
            rd = csv.reader(f)
            header = next(rd)
            try:
                col = header.index("SMILES")
            except ValueError:
                col = 0
                print(f"    (no 'SMILES' header; header={header}, using col 0)")
            for row in rd:
                if row and row[col].strip():
                    yield row[col].strip()
        else:
            for line in f:
                s = line.strip().split()[0] if line.strip() else ""
                if s:
                    yield s


def main():
    results = {}
    for name, spec in SOURCES.items():
        print(f"\n=== {name} ===")
        dest = HERE / spec["fname"]
        try:
            fetch(spec["url"], dest)
        except Exception as e:
            print(f"  DOWNLOAD FAILED: {e}")
            results[name] = {"error": str(e)}
            continue

        n = n_slash = n_at = n_any = 0
        examples = []
        for smi in smiles_iter(dest, spec["kind"]):
            n += 1
            has_ez = ("/" in smi) or ("\\" in smi)
            has_at = "@" in smi
            n_slash += has_ez
            n_at += has_at
            if has_ez or has_at:
                n_any += 1
                if len(examples) < 5:
                    examples.append(smi)

        exp = spec["expected_n"]
        ok = (n == exp)
        print(f"  molecules read     : {n:,}   (published {exp:,})  "
              f"{'MATCH' if ok else '*** MISMATCH -- count is not trustworthy ***'}")
        print(f"  containing / or \\  : {n_slash:,}  ({100*n_slash/max(n,1):.4f}%)")
        print(f"  containing @       : {n_at:,}  ({100*n_at/max(n,1):.4f}%)")
        if examples:
            print(f"  examples           : {examples}")

        results[name] = {
            "n": n, "expected_n": exp, "count_matches_published": ok,
            "n_ez": n_slash, "n_at": n_at, "n_any_stereo": n_any,
            "frac_ez": n_slash / max(n, 1), "frac_at": n_at / max(n, 1),
            "examples": examples,
        }

    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")

    print("\n" + "=" * 62)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:14s} FAILED: {r['error']}")
            continue
        verdict = "ZERO stereo" if r["n_any_stereo"] == 0 else \
                  f"{r['n_any_stereo']:,} molecules carry stereo"
        flag = "" if r["count_matches_published"] else "  [COUNT MISMATCH]"
        print(f"  {name:14s} {verdict}{flag}")


if __name__ == "__main__":
    sys.exit(main())
