"""
Step 9: close the last hole -- do real text-to-molecule models actually EMIT
double-bond stereochemistry?

Everything so far rests on the assumption that models trained on ChEBI-20
(21.6% of test targets carry E/Z) reproduce that stereochemistry in their
outputs. If they emit none, the realistic failure is stereo REMOVAL, which FCD
does charge for -- and the finding dies.

Runs MolT5 caption2smiles on the ChEBI-20 test descriptions and counts.
"""
import json
import os
import sys
import time
from pathlib import Path

# a stale token in the user cache makes the hub return 401; go anonymous
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import torch
from rdkit import Chem, RDLogger
from transformers import AutoTokenizer, T5ForConditionalGeneration

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
TEST = HERE / "chebi20_test.txt"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "laituan245/molt5-large-caption2smiles"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 3300
# keep the footprint small: the GPU is shared with the user's own training job
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MAXNEW = 320
OUT = HERE / f"molt5_outputs_{MODEL.split('/')[-1]}.json"


def read_test(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append({"cid": p[0], "gt": p[1], "desc": p[2]})
            if len(rows) >= limit:
                break
    return rows


def has_ez(s):
    return ("/" in s) or ("\\" in s)


def main():
    rows = read_test(TEST, LIMIT)
    print(f"ChEBI-20 test rows: {len(rows)}")
    gt_ez = sum(1 for r in rows if has_ez(r["gt"]))
    print(f"  ground-truth targets carrying E/Z: {gt_ez} ({100*gt_ez/len(rows):.2f}%)")

    print(f"\nloading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, model_max_length=512, token=False)
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL, token=False, dtype=torch.float16).cuda().eval()
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # Incremental checkpoint. A CUDA OOM -- another job claiming the GPU mid-run --
    # previously discarded 2,000 completed generations. Append as we go and resume.
    part = Path(str(OUT) + ".partial.jsonl")
    preds = []
    if part.exists():
        preds = [json.loads(l)["pred"] for l in
                 part.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  resuming from checkpoint: {len(preds)} generations already done")

    t0 = time.time()
    pf = open(part, "a", encoding="utf-8")
    for i in range(len(preds), len(rows), BATCH):
        batch = rows[i:i + BATCH]
        enc = tok([r["desc"] for r in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=MAXNEW)
        dec = tok.batch_decode(out, skip_special_tokens=True)
        preds.extend(dec)
        for p in dec:
            pf.write(json.dumps({"pred": p}) + "\n")
        pf.flush()
        if (i // BATCH) % 25 == 0:
            done = min(i + BATCH, len(rows))
            print(f"  {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    pf.close()

    # analyse
    n = len(preds)
    n_valid = n_ez = n_ez_valid = n_at = 0
    n_ez_correct = n_ez_gt_pairs = 0
    recs = []
    for r, p in zip(rows, preds):
        m = Chem.MolFromSmiles(p)
        valid = m is not None
        can = Chem.MolToSmiles(m) if valid else None
        pez = has_ez(can) if valid else has_ez(p)
        n_valid += valid
        n_ez += has_ez(p)
        n_at += ("@" in p)
        if valid and pez:
            n_ez_valid += 1
        gm = Chem.MolFromSmiles(r["gt"])
        gcan = Chem.MolToSmiles(gm) if gm else None
        if gcan and has_ez(gcan):
            n_ez_gt_pairs += 1
            if can == gcan:
                n_ez_correct += 1
        recs.append({"cid": r["cid"], "gt": r["gt"], "pred": p,
                     "valid": valid, "pred_has_ez": bool(pez)})

    print("\n" + "=" * 64)
    print(f"MolT5 outputs ({MODEL})")
    print("=" * 64)
    print(f"  generated                        : {n}")
    print(f"  valid SMILES                     : {n_valid} ({100*n_valid/n:.2f}%)")
    print(f"  containing '/' or '\\' (E/Z)      : {n_ez} ({100*n_ez/n:.2f}%)")
    print(f"  containing '@' (tetrahedral)     : {n_at} ({100*n_at/n:.2f}%)")
    print(f"  ground truth E/Z rate            : {100*gt_ez/n:.2f}%")
    print(f"\n  among {n_ez_gt_pairs} targets that carry E/Z:")
    print(f"    exact canonical match          : {n_ez_correct} ({100*n_ez_correct/max(n_ez_gt_pairs,1):.2f}%)")

    OUT.write_text(json.dumps({
        "model": MODEL, "n": n, "n_valid": n_valid,
        "n_pred_ez": n_ez, "n_pred_at": n_at, "n_gt_ez": gt_ez,
        "n_ez_targets": n_ez_gt_pairs, "n_ez_exact_match": n_ez_correct,
        "records": recs,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
