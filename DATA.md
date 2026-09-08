# External data

None of these are redistributed here. They carry their own licences and some are
large enough that a clone would be unusable. Put each in the repository root —
the scripts resolve inputs relative to their own location.

| file the scripts expect | what it is | where it comes from |
|---|---|---|
| `chembl_37_chemreps.txt.gz` | ChEMBL 37 chemical representations (~293 MB) | EBI FTP, `pub/databases/chembl/ChEMBLdb/releases/chembl_37/` |
| `chebi20_train.txt`, `chebi20_validation.txt`, `chebi20_test.txt` | the ChEBI-20 caption↔SMILES split | the MolT5 / Text2Mol release (Edwards et al.) |
| `moses_dataset_v1.csv` | MOSES benchmark set (~84 MB) | `molecularsets/moses` |
| `guacamol_v1_all.smiles` | GuacaMol benchmark set (~77 MB) | `BenevolentAI/guacamol` |
| `opsin.jar` | OPSIN, name→structure (~14 MB) | `dan2097/opsin` releases |

## Derived files the scripts build for you

Run these once and the rest of the pipeline has what it needs:

```bash
python 02_sample_chembl.py        # -> chembl_stereo.smi      (from chembl_37_chemreps.txt.gz)
python 12_build_targets.py        # -> targets.jsonl
python 12b_build_targets_avg.py   # -> targets_avg.jsonl
```

## Third-party source referred to but not copied here

Two GuacaMol source files are quoted in the Supporting Information as evidence
that its data pipeline discards stereochemistry. They are Apache-2.0 and belong
to their authors, so they are cited rather than vendored:

- `guacamol/data/get_data.py` and `guacamol/utils/chemistry.py` in
  `BenevolentAI/guacamol`. The relevant behaviour is the canonicalisation step
  that drops isomeric information before the benchmark sets are written.

Pin a commit when you check it — the line numbers move.
