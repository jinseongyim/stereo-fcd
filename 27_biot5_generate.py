"""
Step 27: generate BioT5-base predictions on ChEBI-20 test.

BioT5 outputs SELFIES, decoded to SMILES. Recipe from the model card:
  prefix = "Definition: You are given a molecule description in English. Your
            job is to generate the molecule SELFIES that fits the description.\n\n"
  input  = prefix + "Now complete the following example -\nInput: {desc}\nOutput: "
  decode = tokenizer.decode(..., skip_special_tokens=True).replace(' ', '')
  smiles = selfies.decoder(that)

Output record format matches 09_molt5_generate.py exactly (cid/gt/pred/valid/
pred_has_ez) so that 26_cross_model.py consumes it unchanged.

Purpose: a real second model to test whether FCD ranking and E/Z stereo fidelity
dissociate. BioT5 has much better published FCD than MolT5-large; a dissociation
requires only that its E/Z recall fall below MolT5-large's (0.278).
"""
import json
import sys
from pathlib import Path

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import selfies as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
TEST = HERE / "chebi20_test.txt"
MODEL = "QizhiPei/biot5-base-text2mol"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 3300
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = HERE / "molt5_outputs_biot5-base-text2mol.json"

PREFIX = ("Definition: You are given a molecule description in English. Your job "
          "is to generate the molecule SELFIES that fits the description.\n\n")


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


def to_smiles(selfies_str):
    try:
        smi = sf.decoder(selfies_str)
        return smi if smi else ""
    except Exception:
        return ""


def main():
    rows = read_test(TEST, LIMIT)
    n = len(rows)
    print(f"test rows: {n}")

    print(f"loading {MODEL} ...", flush=True)
    tok = T5Tokenizer.from_pretrained(MODEL, model_max_length=512, token=False)
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL, token=False, dtype=torch.float16).cuda().eval()

    preds = []
    import time
    t0 = time.time()
    for i in range(0, n, BATCH):
        chunk = rows[i:i + BATCH]
        texts = [PREFIX + f"Now complete the following example -\nInput: {r['desc']}\nOutput: "
                 for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, num_beams=1, max_length=512)
        dec = tok.batch_decode(out, skip_special_tokens=True)
        for d in dec:
            preds.append(to_smiles(d.replace(" ", "")))
        if (i // BATCH) % 20 == 0:
            print(f"  {min(i+BATCH, n)}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    n_valid = n_ez = n_at = gt_ez = n_ez_gt = n_ez_correct = 0
    recs = []
    for r, p in zip(rows, preds):
        m = Chem.MolFromSmiles(p) if p else None
        valid = m is not None
        can = Chem.MolToSmiles(m) if valid else None
        pez = has_ez(can) if valid else has_ez(p)
        n_valid += valid
        n_ez += has_ez(p)
        n_at += ("@" in p)
        gm = Chem.MolFromSmiles(r["gt"])
        gcan = Chem.MolToSmiles(gm) if gm else None
        if gcan and has_ez(gcan):
            gt_ez += 1
            n_ez_gt += 1
            if can == gcan:
                n_ez_correct += 1
        recs.append({"cid": r["cid"], "gt": r["gt"], "pred": p,
                     "valid": valid, "pred_has_ez": bool(pez)})

    print("\n" + "=" * 64)
    print(f"BioT5 outputs ({MODEL})")
    print(f"  generated                    : {n}")
    print(f"  valid SMILES                 : {n_valid} ({100*n_valid/n:.2f}%)")
    print(f"  containing '/' or '\\' (E/Z)  : {n_ez} ({100*n_ez/n:.2f}%)")
    print(f"  containing '@'               : {n_at} ({100*n_at/n:.2f}%)")
    print(f"  E/Z-bearing targets          : {n_ez_gt}")
    print(f"    exact canonical match      : {n_ez_correct} "
          f"({100*n_ez_correct/max(n_ez_gt,1):.2f}%)")

    OUT.write_text(json.dumps({
        "model": MODEL, "n": n, "n_valid": n_valid,
        "n_pred_ez": n_ez, "n_pred_at": n_at, "n_gt_ez": gt_ez,
        "n_ez_targets": n_ez_gt, "n_ez_exact_match": n_ez_correct,
        "records": recs,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
