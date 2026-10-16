# Supplementary material: optimal transport

Status: preliminary, supplementary to the Phase 1 document. This material belongs to Appendix A of the Phase 1 document. It has no due date.

## 1. Scope

This supplement covers one method: entropic unbalanced optimal transport with an importance-weighted cost, applied to past theme-park and resort openings. The other estimation methods of the earlier analysis are dropped from this supplement and from the Phase 1 appendix. They are the synthetic-control and difference-in-differences effects of each opening, the ambient meta-analysis (A1 to A3), the scaling law A2 and the ambient fallback of the route rule. The single-country replication is dropped as well.

The results are diagnostic. They are not an estimate of the effect of a Disney-type complex in Thailand, and they are not the Thai net benefit.

## 2. Question

What change in international tourism receipts, expressed as a percentage of GDP, do past openings point to for a complex of the proposed size in Thailand?

## 3. Data

- Primary sample: 19 episodes in 19 host economies. The importance model requires at least 20 episodes, so the sample does not meet that requirement.
- Outcome: international tourism receipts as a percentage of GDP. The effect of an opening is the mean gap between the observed outcome and its synthetic counterfactual over the post-opening window. The counterfactual of each episode is taken from the earlier analysis.
- Features of the host economy: nine variables, listed in `results/thailand_transport.json` as log GDP per capita, log population, receipts as a percentage of GDP, log receipts per arrival, arrivals per capita, air passengers per capita, the pre-opening growth of receipts, capital cost as a percentage of GDP and an indicator for an integrated resort.
- Source: World Bank indicator files in `data/raw/wdi` (vintage 2026-07-13). On 16 October 2026 the files were compared cell by cell with the live API: 12,868 compared cells, all equal.
- Cost measure: the dollar costs of the openings are partly derived from local-currency figures. Fifteen of the 27 derived costs have an assumed price year.

## 4. Method

The effects of the 19 episodes are moved to the yearly feature vectors of the target economy. The cost between an episode and a target point is the squared distance of the standardized features, with each feature weighted by its importance weight from the feature stacking model. The entropic regularisation is 0.983. The source side of the problem is relaxed (rho = 1.0), so an episode may send more or less than its own share, and the target side is enforced. The transported effect is the mean of the source effects weighted by the mass that arrives.

Target support in the run of 16 October 2026: effective number of sources 6.75 (minimum 2), largest source share 0.25, no material feature outside the range of the sources. The overlap test gives p = 0.90 over 9,999 permutations.

## 5. Validation

The validation is leave-one-economy-out over 19 folds. In every fold the importance weights are learned again from the training episodes.

| Predictor | Error (RMSE) |
|---|---|
| Equal-weight mean of the episodes | 1.134 |
| Nearest neighbour, one nearest episode | 1.281 |
| Kernel predictor | 1.016 |
| Optimal transport with uniform weights | 1.006 |
| Optimal transport with importance weights | 1.015 |
| Nearest neighbour, three nearest episodes | 0.903 |

The pre-registered comparison has two tests. The importance-weighted transport passes the test against the equal-weight mean, with a ratio of 0.895 against a limit of 0.95. It fails the test against the uniform-weight transport, with a ratio of 1.009 against the same limit. The nearest-neighbour predictor with three neighbours has the lowest error in this table. It is reported here and is not part of the method.

## 6. Importance weights

The importance model is not usable on any of its four criteria in the primary sample: cross-validated R-squared 0.072 (minimum 0.10), permutation p-value 0.436 (maximum 0.10), split-half correlation -0.148 (minimum 0.50), and 19 cases (minimum 20). Across five seeds the model was usable in none of them. The transport nevertheless uses these weights, which is why the uniform-weight transport is reported alongside it.

## 7. Diagnostic result

For the proposed complex of THB 300 billion, which is 1.58 percent of GDP at the 2025 cost year, the transported effect is as follows.

- Point estimate: -0.73 percentage points of GDP.
- Bootstrap band, 5 to 95 percent, over 5,000 resamples of economies: -1.28 to -0.19.
- Predictive 90 percent interval for a new study, from the leave-one-economy-out errors: -2.73 to 1.27. The interval includes zero.
- Transported placebo distribution, 5 to 95 percent, over 20,000 draws: -0.90 to 1.41. The point estimate lies inside this distribution.

The bootstrap band excludes zero. The validation criterion against the uniform-weight transport is not met, however, so the band does not establish an effect. The result is a diagnostic and is reported only as one.

## 8. Limitations

- The sample has 19 episodes, one below the number the importance model requires.
- The importance weights are not usable, and the transport uses them.
- The validation fails the pre-registered test against the uniform-weight transport.
- The intervals of the error ratios are descriptive and are not a test.
- The episodes are past openings in host economies. Their features are not the features of Thailand, which is extrapolated.
- Several cost figures are derived estimates, not reported costs.

## 9. Work for later phases

If the team continues this method, the following work remains.

1. Register, before any further use, whether the transport uses uniform weights or importance weights. The uniform-weight transport validates better, with an error of 1.006 against 1.134 for the mean.
2. Extend the sample to at least 20 episodes, and repeat the validation.
3. Test the stability of the importance model across seeds, and report the verdict for each seed.
4. Check the predictive interval as the number of episodes grows.
5. Decide with the team whether the transport enters the bottom-up appraisal as a cross-check only.

## 10. Reproduction

- Code: `disney-tourism-thailand/src/dtt/transport.py`, `importance.py` and `impact.py`.
- Notebooks, run on the dataset with `DTT_RUN_REAL=1` from the repository root: notebook 03 (feature importance, about 75 minutes) and notebook 04 (optimal transport, about one minute).
- Results: `results/thailand_transport.json`, `results/transport_weights.csv`, `results/loco_table.csv`, `results/loco_ratios.csv`, `results/overlap_test.json` and `results/importance_diagnostics.json`.
- Run: 16 October 2026, with the results in the re-run pull request merged into `main`.
