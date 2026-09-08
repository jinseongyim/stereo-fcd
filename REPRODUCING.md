# Supporting Information archive

Code and results for *Fréchet ChemNet Distance Is Blind to Double-Bond
Stereochemistry*.

Everything here was produced on a single workstation (RTX 4070 Ti SUPER, Windows,
CUDA 12.6). No result requires a GPU except MolT5/BioT5 generation, which can also
run on CPU slowly.

## Requirements

Exact pinned versions (Python 3.12.2) are in `environment.txt`:

```
rdkit 2026.03.3   torch 2.6.0+cu126   fcd 1.2.2
numpy 2.5.1   scipy 1.17.1   selfies 2.1.1   matplotlib 3.10.1
```

Two environment notes, both of which will otherwise cost you an afternoon:

- `fcd` 1.2.2 fails under `numpy >= 2` because `np.row_stack` was removed. Every
  script here installs the shim `np.row_stack = np.vstack` before importing `fcd`.
- `fcd.get_predictions` hangs on Windows with the default `n_jobs`. Pass `n_jobs=0`.

Because canonical SMILES ordering and CIP perception are load-bearing for the
bond-level analysis, the RDKit version is pinned and matters.

## What is not included

The large public inputs are downloaded by the scripts and are omitted here to keep
the archive small:

| input | size | obtained by |
|---|---|---|
| ChEMBL 37 `chemreps` | 293 MB | `02b_fetch_bulk.py` |
| MOSES `dataset_v1.csv` | 85 MB | `20_moses_guacamol_stereo.py` |
| GuacaMol v1 (all) | 77 MB | `20_moses_guacamol_stereo.py` |
| ChEBI-20 splits | 13 MB | `07_chebi20.py` |

Also omitted: `targets.jsonl` / `targets_avg.jsonl` (12.7 MB of conformer-averaged
training targets), regenerable with `12b_build_targets_avg.py`.

## Layout

Scripts and their outputs sit in one flat directory, because each script resolves
paths relative to its own location — splitting code and results apart breaks them.
The layout is flat; the GuacaMol source excerpts quoted in the Supporting
Information are not vendored here (see DATA.md).

## A pitfall worth reading first: inverting E/Z correctly

A stereogenic double bond that carries a CIP label **must** change that label when
its geometry is inverted, so any method that leaves labelled bonds unchanged is a
broken inversion — not evidence of symmetry. Three natural methods under-invert:

- an `RWMol` copy with `SetStereo` returns the original (the copied `BondDir`
  wins in the SMILES writer);
- flipping the `STEREOE`/`STEREOZ` enum in place and calling
  `AssignStereochemistry(..., force=True)` no-ops on bonds with directional
  neighbours on both ends;
- a direction-search heuristic leaves ~12 % of bonds unflipped.

The correct inversion flips the enum and rebuilds directions with
`Chem.SetDoubleBondNeighborDirections`; `36_inversion_methods.py` compares all of
these and shows only this one flips every one of the 292 mapped stereogenic bonds
(0 survivors). Scripts `15b`, `22`, and `35` use it.

## Contents

### Main-text results

| file | what it establishes |
|---|---|
| `c2_full_correct.json` | Full clean inversion of MolT5-large output. FCD 0.386803 → 0.386801 (−0.001 σ); gated Stereo-FCD −11.12 σ; all 384 E/Z-bearing molecules changed. |
| `cip_recount.json` | Per-bond E/Z on the actually inverted molecules (via `35_protocolB_actual.py`): 289/292 = 98.97 % → 3/292 = 1.03 %; exact match 111/380 → 2/380. |
| `flip_vs_strip_repeated.json` | 15-draw flip vs strip vs substitution. Invert every E/Z = 0.010 σ; delete E/Z = 41.8 σ; swap 2 % / 5 % of molecules = 1.56 σ / 7.10 σ. |
| `cross_model.json`, `common_denominator.json` | MolT5-base / MolT5-large / BioT5-base on ChEBI-20; end-to-end and common-support E/Z recall (0.256 vs 0.997 on shared bonds). |
| `fcd_common_set.json` | Common-caption FCD, 3,108 captions: MolT5-large 0.356 vs BioT5-base 0.427, paired bootstrap Δ = +0.071, 95 % CI [−0.003, +0.116]. |
| `lexical_leakage.json` | Order-preserving writer: semantic sensitivity exactly 0, only lexical leakage from atom reordering. |
| `moses_guacamol_stereo.json` | Zero stereochemistry in both benchmark datasets. |
| `drug_case_studies.json` | Four approved E/Z drug pairs and four tetrahedral controls, structures taken from PubChem by name and validated before use (same constitution, differing stereo-SMILES, correct kind of stereocentre). Every E/Z pair: one-hot bit-identical, max\|Δa\| = 0 under both the canonical and the order-preserving writer. Every tetrahedral pair separates (0.41--1.14). Via `37_drug_case_studies.py`. |
| `fcd_usage_evidence.json` | How widely FCD is used, for the claim that the blind spot matters in practice: OpenAlex citation counts (FCD 352, GuacaMol 830, MolT5 133, BioT5 46) and PyPI monthly downloads (`fcd` 4,331, `fcd-torch` 2,598), accessed 2026-09-02. Via `39_fcd_usage_evidence.py`. |
| `figure1.pdf`, `figure1.png`, `figure2.pdf`, `figure2.png`, `toc_graphic.pdf`, `toc_graphic.png` | Figure 1 = the drug pairs, drawn with CIP labels from `rdCIPLabeler` (`38_figure_drugs.py`); Figure 2 = the mechanism (`19_figure1.py`); and the TOC graphic. |

### Supporting results

| file | section |
|---|---|
| `collision_results.json` | S2. Per-pair record for all 18,936 ChEMBL E/Z pairs (5.9 MB). |
| `chebi20_results.json` | S3. Stereo prevalence per split. |
| `strip_penalty_results.json`, `flip_vs_strip_onepool.json` | S4. Strip-vs-flip contrast (single-pool broad picture; superseded by the 15-draw table for reported numbers). |
| `coverage_precision_recall.json`, `bond_matching_audit.json` | S6. Bond-level decomposition and the matcher audit. |
| `inversion_bug_scope.json` | S6/S7. Flip rates of the old vs correct inversion and the FCD check under a real inversion. |
| `stereofcd_eval.json`, `within_pair.json`, `stereo_head.pt` | S6. Gated correction: corruption sweep, unpaired-objective failure, trained head (1.2 MB). |
| `molt5_outputs_molt5-large-caption2smiles.json` | Generated molecules, ChEBI-20 test (MolT5-large). |
| `molt5_outputs_molt5-base-caption2smiles.json`, `molt5_outputs_biot5-base-text2mol.json` | Same, other models. |

### Source evidence

| file | what it shows |
|---|---|
| `guacamol/utils/chemistry.py` (upstream) | `filter_and_canonicalize(..., include_stereocenters=False)` declaration. |
| `guacamol/data/get_data.py` (upstream) | The call site passing `False` in that position. |

Both retrieved 2026-07-20 from the GuacaMol repository at `master`.

## Reproducing the central claim

The shortest path from a clean environment to the headline numbers:

```bash
python 01_verify_chemnet_collapse.py   # vocabulary collapse, @ control
python 09_molt5_generate.py            # generate on ChEBI-20 test
python 15b_c2_full_correct.py          # clean inversion of everything, score FCD
python 35_protocolB_actual.py          # score the same inverted molecules vs truth
python 22_flip_vs_strip_repeated.py    # flip vs strip vs swap, 15 draws
python 19_figure1.py                   # figure
```

`15b` and `35` are the two that matter: the first shows the metric does not move
under a complete, verified inversion; the second shows the molecules did change
(98.97 % → 1.03 % per-bond) on that same inverted set.

## A note on the gated correction

Section S6 documents a metric that does not work. It is included because its
failure is the argument of the paper, not an omission from it: a gated additive
correction on frozen ChemNet embeddings passes synthetic corruption sweeps and
then prefers fully inverted molecules on real generator output (−11.12 σ).
`stereo_head.pt` and `stereofcd_eval.json` are provided so that this can be
checked rather than taken on trust.
