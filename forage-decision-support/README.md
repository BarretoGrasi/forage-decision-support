# Forage Decision-Support Framework

Reproducible code structure for the study:

**An Interpretable Multi-Criteria Decision-Support Framework for Forage Species Recommendation and Grass–Legume Intercropping under Environmental Constraints**

## Repository structure

- `src/` — reusable implementation of preprocessing, Gower/HAC analysis,
  individual suitability scoring, intercropping scoring, and recommendation.
- `experiments/` — scripts reproducing the baseline, scenario-responsiveness,
  and ecological-productivity sensitivity analyses.
- `diagnostics/` — data-quality diagnostics kept separate from the model.
- `data/` — curated forage dataset (not included in this generated package).
- `results/` — generated tables and figures.
- `docs/` — variable documentation.
- `archive/` — original complete script preserved for traceability.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Copy the curated CSV to:

```text
data/forage_dataset.csv
```

## Reproduce the reference scenario

```bash
python experiments/run_baseline.py
```

Reference conditions:

- pH = 5.0
- climate = Cerrado
- soil texture = clayey
- fertility = medium

## Run the three environmental scenarios

```bash
python experiments/run_scenarios.py
```

## Individual ecological-productivity sensitivity

```bash
python experiments/sensitivity_individual.py
```

## Notes on reproducibility

The original complete script is preserved in
`archive/original_complete_script.py`. The modular files reorganize the same
core workflow into reusable components. Before public release, compare the
outputs of the modular scripts against the original manuscript results using
the final curated dataset.

## Citation

Citation information should be added after publication or when a preprint/DOI
becomes available.
