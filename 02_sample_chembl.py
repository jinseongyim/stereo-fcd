"""
Step 2: sample real drug-like molecules from ChEMBL and measure how often
specified E/Z double-bond stereochemistry actually appears.

Writes chembl_sample.smi (one canonical SMILES per line).
"""
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "chembl_sample.smi"
N_PAGES = 40
PAGE = 1000
TOTAL = 2_921_148
SEED = 0

BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule?limit={lim}&offset={off}&format=json"


def fetch(offset, limit=PAGE, retries=3):
    url = BASE.format(lim=limit, off=offset)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)["molecules"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  offset {offset}: FAILED {type(e).__name__}", file=sys.stderr)
                return []
            time.sleep(2 * (attempt + 1))
    return []


def main():
    rng = random.Random(SEED)
    offsets = rng.sample(range(0, TOTAL - PAGE), N_PAGES)

    seen, out = set(), []
    for i, off in enumerate(offsets, 1):
        mols = fetch(off)
        for m in mols:
            smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
            if smi and smi not in seen:
                seen.add(smi)
                out.append(smi)
        print(f"  page {i:2d}/{N_PAGES} offset={off:<8d} cumulative={len(out)}")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"\nwrote {len(out)} unique SMILES -> {OUT}")


if __name__ == "__main__":
    main()
