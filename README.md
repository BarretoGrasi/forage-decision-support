# Forage decision support

Reproducible screening and Pareto prioritisation of grass–legume mixtures.

## Scope

This package uses the consolidated Xaraés/MG-5 execution supplied with the manuscript. The 112 original database entries become 111 operational records after identity consolidation. Unresolved mandatory descriptors remain subject to verification. Computational reproduction does not establish agronomic validation.

The newly pasted older recommender and the successive Colab `%run` commands are not the execution entry point. Do not run the older intermediate recommenders after consolidation. Figure data are calculated from the current execution, not from manually entered silhouette lists.

## Installation and execution

Python 3.12.14 is the environment used for the verification described in `docs/verification.md` (see the actual recorded version there).

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

From Google Colab, after uploading and extracting this folder:

```python
%cd /content/forage-decision-support
%pip install -r requirements.txt
!python scripts/run_pipeline.py
```

The runner copies inputs into a fresh output directory, reconciles records, runs the consolidated recommender, and produces tables, audit files, an execution log, figures, and a JSON summary. A nonempty output directory is rejected to prevent mixing different executions. For another run:

```bash
python scripts/run_pipeline.py --output results/run_02
```

## Files

- `data/db_pastagem2_original.csv`: supplied original database (112 entries).
- `data/correcoes_curadoria.csv`: supplied documentary curation controls.
- `scripts/reconciliar_xaraes.py`: consolidation and source-level audit.
- `scripts/recomendador_reconciliado_xaraes_mg5.py`: consolidated algorithm, preserved except for the optional clustering switch.
- `scripts/figures.py`: Pareto projections, computed HAC silhouettes, and Gower matrix comparison.
- `scripts/run_pipeline.py`: single execution entry point.
- `docs/reconciliation_notes.txt`: original consolidation notes, including remaining documentary uncertainties.
- `docs/verification.md`: actual verification results.

## Reference checks

Expected baseline: 126 eligible pairs, ten efficient pairs, 116 dominated pairs; Marandu + Belmonte ranks first under baseline preferences. The figure generator rejects other baseline counts; it does not modify or filter data to force a match. Actual legends are computed from the pair tables.

Outputs under `results/latest/figures/` include:

- `pareto_objective_space.png` and `.pdf`;
- `silhouette_111_records.png` and `.pdf`, plus computed `silhouette_values.csv`;
- `gower_comparison_111_records.png` and `.pdf`, plus `gower_correlations.csv`.

## Optional clustering comparison

The ordinary run includes HAC and its silhouette figure. The expensive PAM, spectral, DBSCAN, and GMM comparator grids are disabled by default. `scikit-learn-extra` is only needed for PAM, not for the recommender or the three figures.

A separate candidate dependency environment is supplied in `requirements-clustering.txt`. It has NOT been verified here; binary compatibility and the full comparison must be checked before publishing its results. In a separate environment:

```bash
python -m pip install -r requirements-clustering.txt
python scripts/run_pipeline.py --full-clustering --output results/full_clustering
```

## Supplementary analyses

The default runner also executes the three additional analyses in `scripts/supplementary.py`:

- exclusion of shared climatic zeros, with numerical reconstruction of the original Gower matrix before comparison;
- min–max objective normalisation over all 126 eligible pairs and paired Monte Carlo comparison using 10,000 Dirichlet(1,1,1) vectors, seed 20260903, unrounded maxima and explicit tie checks;
- individual ecological–productivity weights 50/50, 60/40 and 70/30, with unchanged post-Pareto weights.

Outputs are written to `results/latest/supplementary/`. The four-scenario figure is written to `results/latest/figures/scenario_fronts.png` and `.pdf`. Pareto membership always uses all three objectives, even in two-dimensional projections.

All three analyses reproduced the supplied supplementary results in the tested environment. The full optional alternative-clustering grid remains unverified.

No licence was chosen on behalf of the authors. Confirm permissions for sharing source data and choose a code/data licence before publishing. No GitHub upload or public repository change was performed.
