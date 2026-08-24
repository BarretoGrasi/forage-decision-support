# Variable dictionary

This file should document the standardized variables used by the framework.

## Gower feature space (27 descriptors)

- 20 climatic-adaptation binary descriptors
- 4 soil descriptors:
  - `ph_min`
  - `ph_max`
  - `text_num`
  - `fert_num`
- 2 morphological descriptors:
  - `profundidade_raiz_max`
  - `altura_media`
- 1 productivity descriptor:
  - `prod_norm`

## Recommendation-only / auxiliary descriptors

Examples include agronomic use, functional group, life cycle, and vegetation
stratum. Expand this dictionary with definitions, units, source conventions,
and missing-value rules before public release.
