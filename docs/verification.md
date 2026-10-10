# Verification

Executed the complete default pipeline in Python 3.12.14.

- Eligible pairs: 126
- Efficient pairs: 10
- Dominated pairs: 116
- HAC best k: 2
- Silhouette: 0.5106957102371745
- Pearson: 0.9846571460293833
- Spearman: 0.9802968632025997

Three figure sets and their numeric tables were generated. The full optional clustering comparator grid was not run. Pandas fragmentation warnings were observed; execution completed successfully.

Only two changes were made to the original consolidated scripts: reconciliation accepts an explicit working directory; the expensive clustering comparison is controlled by the runner. Decision rules were preserved.

## Additional analyses verified

- Four-scenario counts: 126/10, 131/15, 41/6 and 10/6 (eligible/efficient).
- Shared-zero exclusion: same ten-member frontier, Jaccard 1.0, Tupi + Belmonte 59.1111.
- Normalisation: same frontier and baseline winner; Marandu + Belmonte sampled frequency 7.24% to 21.58%; same sampled winner 71.73%. All ten win-count rows reproduced the supplied results.
- Ecological–productivity weights: 50/50 has 11 efficient pairs (Jaccard 0.9091), Mombaça + Mineirão 60.1184; 60/40 has 10, Marandu + Belmonte 62.2842; 70/30 has 9 (Jaccard 0.9), Marandu + Belmonte 66.1823.

Additional scripts were reorganised as functions without shared notebook state. Raw record names, including legacy spellings, are preserved to avoid silently changing pair identities.
