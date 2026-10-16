# Supplement: record of the historical analysis of theme-park openings and tourism receipts, Thailand

Date of record: 16 October 2026 (UTC+7).

Repository: `econ-disney-seminar` (public), folder `disney-tourism-thailand`.

Code commit `b6bb77f`. Results commit `e206a5b`. Both commits are in the history of `main`.

## 1. Status

The analysis runs on the dataset only. The simulation generators and the simulated run mode are not part of the repository. The clean run of 16 October 2026 completed every step, except the SkylakeX kernel, which cannot run on the computer used for the run. The route rule selects the ambient route, and the primary estimator is A2. The primary effect for the proposal of 300 billion baht has a median of 0.40 percentage points of GDP, with a 5 to 95 percent range from -1.14 to 1.94 points. At the 2025 baseline that is 2.33 billion US dollars a year (range -6.55 to 11.17 billion). The weighted transport gives a point estimate of -0.73 points, with the opposite sign. Both are reported.

## 2. Run provenance

- Run record: the status, summary, manifest, package list, interpreter version, World Bank check and test totals of the run are in `seminar/provenance/run_2026-10-16/`. The run started at 08:12:45 and finished at 10:01:05 on 16 October 2026 (UTC+7).
- Code of the run: the pipeline files of commit `b6bb77f` are identical to the files that ran. The notebooks ran with `DTT_RUN_REAL=1`. The notebook times were 00 11.6 s, 01 23.6 s, 02 36.0 s, 03 4511.4 s, 04 55.9 s and 05 28.0 s, 4711.5 s in total.
- Manifest: `manifest_sha256.csv` was taken at 08:13, before the first step, and lists 160 files. Its entries for `results_20261016\`, `figures_20261016\` and `executed_20261016\` are outputs of an earlier run. They are not part of the final tree.
- Text changes after the manifest was taken in `README.md` (World Bank status, kernel statement, result section and thermal sentence), the first markdown cell of notebooks 00 to 05 (run-time paragraph only), and the heading list in `tests/test_notebooks_static.py`. No code cell changed after the manifest was taken.
- Python 3.12.8 (MSC v.1942, 64-bit) on Windows 11 Home. Package versions from `pip_freeze.txt`: cvxpy 1.9.3, matplotlib 3.11.2, numpy 2.5.4, pandas 3.0.6, POT 0.9.7.post1, pytest 9.1.1, scikit-learn 1.9.1, scipy 1.18.1, statsmodels 0.15.0.

## 3. Results of the run

Route rule (`results/route_decision.json`, rule hash 852b46da9afa092222650d1709049c85d02397fe5f83d3b46eb5e68c9e368b7f):

- Importance model not usable. The primary sample has 19 episodes, below the minimum of 20. The cross-validated R-squared is 0.072 (minimum 0.10), the permutation p-value is 0.436 (maximum 0.10) and the split-half correlation is -0.148 (minimum 0.50).
- Leave-one-economy-out RMSE: weighted transport 1.015, plain mean 1.134, uniform transport 1.006. The ratio of the weighted transport to the uniform transport, 1.009, is above the limit of 0.95, so this criterion fails.
- Target supported: effective number of sources 6.75 (minimum 2), largest source share 0.25, and no material feature outside the range of the sources.
- Route: ambient. Estimator: A2, a scaling law through the origin in the cost share, with LOEO RMSE 1.038 against 1.138 for A1 (ratio 0.913, below the limit of 0.95) and 1.630 for A3. The nearest-neighbour predictor nn3 has the lowest LOEO RMSE of all, 0.903. The rule does not use it.

Scenario complex_total (proposal of THB 300 billion, cost year 2025, 1.58 percent of GDP, US$ 9.12 billion at 32.8828 THB per US dollar, receipts 12.15 percent of GDP on the panel basis):

- Primary effect, percentage points of GDP: median 0.40, 5 to 95 percent -1.14 to 1.94, 25 to 75 percent -0.21 to 1.01.
- Primary effect, US$ billion a year: 2.33 at the 2025 baseline (-6.55 to 11.17) and 2.19 at the 2019 baseline (-6.18 to 10.53).
- Primary effect, percent of the counterfactual receipts: median 9.96, 5 to 95 percent -69.87 to 89.43.
- 90 percent interval for a new study: -2.73 to 1.27 points (half width 2.00, from 19 cases). Minimum detectable effect at alpha 0.05 and power 0.8: 5.56 points (normal approximation 6.21). Detection probability over the effect draws: 0.066.
- Weighted transport, a diagnostic and not the primary route: point estimate -0.73 points. Sensitivity median -0.71 points, 5 to 95 percent -1.28 to -0.19.
- Scenarios of 200 billion and 100 billion baht: primary medians 0.27 and 0.13 points.
- The government claim of about 1 percent of GDP a year (February 2026, no study cited) appears only as a reference scale.

## 4. Verification

| Run | Result |
|---|---|
| pytest, default kernel, dataset check skipped (`pytest_default.log`) | 1795 passed, 1 skipped |
| pytest with `DTT_RUN_REAL=1`, including the dataset check (`pytest_dataset.log`) | 1796 passed |
| pytest with OpenBLAS kernel Haswell (`pytest_Haswell.log`) | 1795 passed, 1 skipped |
| pytest with OpenBLAS kernel Zen (`pytest_Zen.log`) | 1795 passed, 1 skipped |
| pytest with OpenBLAS kernel SkylakeX (`pytest_SkylakeX.log`) | Process stopped with an illegal instruction, exit code -1073741795, after 25.8 seconds |
| Dataset chain, notebooks 00 to 05 in order (`dataset_chain.log`) | Exit 0, 4711.5 seconds |
| Notebook 01, comparison with the reference values | 1179 of 1216 pass; 37 differ |
| World Bank check against the live API (`wdi.log`) | 96 economies, 12,868 compared cells, 12,868 equal, 0 different, 0 present in only one source, 1,532 empty in both |
| Code commit `b6bb77f`, full suite including the appraisal tests | 1877 passed, 1 skipped |

The thermal log `results\thermal_log.csv` has 196 rows. Its temperature column is empty in every row, so no CPU temperature was recorded. Every row shows the action `ok`, and the guard relied on the CPU load.

## 5. Limitations and disclosures

1. Sample size. The primary sample has 19 episodes, one below the 20 that the importance verdict requires. Under the fixed rule this sends the analysis to the ambient route. That is the outcome of the rule. The importance model is not usable on any of its criteria, and under five seeds the verdict was usable in none of them. The variant with all 27 feasible cases also fails.
2. Close choice of estimator. The selection of A2 over A1 is flagged as close, because its RMSE ratio of 0.913 lies within 0.05 of the limit of 0.95.
3. A predictor with lower error is not used. The nearest-neighbour predictor nn3 has an LOEO RMSE of 0.903, below 1.038 for A2 and 1.015 for the weighted transport. The rule does not use it, and it is reported here.
4. Sign disagreement. The primary median is +0.40 points of GDP, and the weighted transport point estimate is -0.73 points. The weighted transport value is the same at the 200 billion and the 300 billion baht scenarios, so the disagreement holds at every cost level in the scenario table.
5. Precision. The predictive interval includes zero. The 90 percent interval for a new study is about 4 points wide, and the minimum detectable effect of 5.56 points is far above the point estimate. The detection probability at the headline is 0.066, which is not the power at the minimum detectable effect.
6. Predictive interval source. The predictive interval is built from the leave-one-economy-out errors of the weighted transport rule (`ot_weighted`), not from the errors of the primary estimator A2.
7. Placebo noise. The absolute null distribution has its placebo noise scaled by a factor of 2.79, equal to the target receipts share of 12.15 percent divided by the source share of 4.36 percent of GDP. The relative null is not rescaled.
8. Scenario widths. Only the medians scale with the cost. The interquartile width of the primary effect is 1.22 points at 300 billion baht, 1.18 at 200 billion and 1.15 at 100 billion.
9. Cost year. At the 2019 cost year the capital cost is 1.78 percent of GDP, about 12 percent above the 1.58 percent at the 2025 cost year.
10. Baseline years. Receipts fall by about 24.5 percent between the 2019 and 2025 baselines, which affects the relative results.
11. Earlier figures. The Stata-era documents give about US$ 1.01 billion a year from a Shanghai-anchored scaling. The pipeline's ambient median is US$ 2.33 billion a year. These are different estimators, and they have not been reconciled.
12. Derived costs. The file `data/processed/cases_investment_fx.csv` holds 27 dollar costs. Twenty are converted from local-currency figures at official annual-average exchange rates, and seven are reported in dollars. All 27 are derived estimates, not reported costs, and 15 have an assumed price year. Costs are known for 28 of the 44 episodes only with the derived dollar costs, and for 12 without them.
13. Proposal status. The THB 300 billion figure is a proposal. No signed agreement, approved budget, named operator or site allocation was found in the sources reviewed.
14. Calibration. The calibration figures in the 9 October 2026 README, such as predictive coverage of 70 to 73 percent at 30 cases, were produced before the run of 16 October 2026. They have not been measured again with the calibration driver, which is not part of this repository.
15. Kernels. SkylakeX cannot run on the computer used for the run. Haswell and Zen agree with the default kernel on every test they run. The check on the dataset is skipped in those runs unless `DTT_RUN_REAL=1` is set, and it passed in the dataset run.
16. World Bank files. The 20 files in `data/raw/wdi/` were retrieved with a web fetch tool during the project, not by a direct download from the World Bank API. The comparison with the live API on 16 October 2026 found no difference in any of the 12,868 compared cells.
17. Cloud computation before 15 October 2026. Before 15 October 2026 some computation ran in the cloud: an early refactoring check of the original real-data equivalence, run once, with only pass or fail visible; structure checks of the catalogue and the derived-cost file by reviewers; and computation in notebook 01 and in two review subagents. No World Bank outcome data were analysed in the cloud. An early cloud scan read text under `data/`, which was then corrected. All later computation ran on the desktop.
18. Simulated world. The repository contains no simulated world, and no simulated result is reported. Some test and fixture names still contain the words simulated or sim. They refer to hand-built fixtures.
19. Tests on hand-built fixtures. Three tests of the earlier simulated world are not in the repository, because their claims cannot hold on the hand-built fixtures: `test_estimated_effects_recover_the_generating_coefficient_of_capex`, `test_exact_fit_in_the_simulated_world_is_reported_in_the_effects_table` and `test_a_window_that_ignores_the_next_episode_picks_up_its_level_shift`. Other tests check the same functions with expected numbers that match the fixtures.
20. Notebook 01. Its stored outputs were cleared. The executed copies from the run are not part of the repository.
21. Data origin. The origin of `data/raw/disney_did_panel.csv` is to be stated in `DATA_NOTICE.md`. The file is part of the public repository.

## 6. Open items

- The origin of `data/raw/disney_did_panel.csv` is to be stated in `DATA_NOTICE.md` (item 21 of section 5).
- No other item in this record is open.

## Annex 1. Source files and SHA-256 values of the final tree

Values are the SHA-256 of the committed content of each file in commit `e206a5b`, folder `disney-tourism-thailand`, as listed by `git ls-tree -r`. Paths are relative to the folder `disney-tourism-thailand`.

| File | SHA-256 |
|---|---|
| `.gitattributes` | `12fb904caa0ce0b078965bcbe9bb44b4635c31b0e4eb9bc6f17db2a970b63b7e` |
| `.gitignore` | `652a83d3587e05bc17d24ba840792bd508393a5929f5824db856968d504548d7` |
| `DATA_NOTICE.md` | `8b59d675e36a7dc49f1c11d5053a13d2f796482d686d9b083ed531caa49b2d6e` |
| `README.md` | `2ad63fa76a5073b7a26bc410c2d3bd9e2e4ec061dbafdc05a8064f716724d4bc` |
| `data/appraisal/assumptions.csv` | `13cec58794aff901572c0a438d3e6767d5fb37cd5e0a34e8d3eb37db497ab52d` |
| `data/appraisal/inputs.csv` | `ae5b8194e35870c92652aaf23fe64d6b35f2c131b7f66164483a9405a4f51bcc` |
| `data/processed/cases_catalogue.csv` | `cd3d24b78aa8f1ca380bdd24e89bfdf9ecd8d687c30a375ce78c69ad6a068b91` |
| `data/processed/cases_investment_fx.csv` | `5567ea28efd9511a7b999ec558923ac78daa8a7738ab5a28c16c35c6658258f7` |
| `data/processed/thailand_baseline.csv` | `2a5b4d28baea34fd6a37ba54a73c8aade4d29ad55868cec2c59f81817d6ab534` |
| `data/processed/thailand_proposal.json` | `4075784fbcf00c96109050f84701b0121364fcc7bde0693b142d437188edc860` |
| `data/raw/disney_did_panel.csv` | `0c9dd53828ad81503079f1b1a63ba5c7e15de5b0e313ccc99a439bce6a9fc46d` |
| `data/raw/wdi/country_metadata.csv` | `319d8c248ba5386c110be64ee2915d7b11a892f5d90ae1b1f9df97f8bab3e1e6` |
| `data/raw/wdi/fetch_log_G1.json` | `3f6a84457b48b8898dadf7d7229ae25e437575e101b078fe3bd68f940ccede4b` |
| `data/raw/wdi/fetch_log_G2.json` | `502336576c945d011c44733b2b66f1ff897ebd4b8c84d2001a6eedc3647d3cba` |
| `data/raw/wdi/fetch_log_G3.json` | `7b66d00c143dcaeccb30a34187097af3d846a30dd02a75b7de4887ed7649f13d` |
| `data/raw/wdi/fetch_log_G4.json` | `7926977509215fb4f11c0b41abcf30f4099890a551f16be89d29f4ee9f67f064` |
| `data/raw/wdi/wdi_G1_IS.AIR.PSGR.csv` | `7b0815d0d813e1ec50a38a876aba1e2a381653eb545c3f5f3a2209efa07e8e3f` |
| `data/raw/wdi/wdi_G1_NY.GDP.MKTP.CD.csv` | `7d435f99b7fe4673a639769e2e060a3cd4f2f699714b2063db50b4adf6c201de` |
| `data/raw/wdi/wdi_G1_SP.POP.TOTL.csv` | `8812e42d470e39b8ca664179d88f20afaf41f1a67d05eca929c1cf0e99d85bbc` |
| `data/raw/wdi/wdi_G1_ST.INT.ARVL.csv` | `09c776e55878f46fada16dd431f1cb066ad863e5a84650d90bbcf07b32440500` |
| `data/raw/wdi/wdi_G1_ST.INT.RCPT.CD.csv` | `24df255f656c78f2ccc413aa45083deabebbcba300334e44c9f0ee628ef747f3` |
| `data/raw/wdi/wdi_G2_IS.AIR.PSGR.csv` | `5c24f1f46b8958382101e5819455f08db27560c60600a1469b4a460a0cf086e2` |
| `data/raw/wdi/wdi_G2_NY.GDP.MKTP.CD.csv` | `543bfe7d3c62727bf8935ad619813b0b1f5283edf5f0c593efcf4a2b4524f6ad` |
| `data/raw/wdi/wdi_G2_SP.POP.TOTL.csv` | `67d02d94fff68ba6e5f646a7484c34c46bad0e20d853d96fa4d040b13f6bd88b` |
| `data/raw/wdi/wdi_G2_ST.INT.ARVL.csv` | `8effd8fbd7ae171b43fca0d464f58ff205f7c8959776ff21ba986b54c70d2dca` |
| `data/raw/wdi/wdi_G2_ST.INT.RCPT.CD.csv` | `5ef445907bfc509bf1d2f7257fc70b3cd6d483e6e554cbea64a9def999f5d06b` |
| `data/raw/wdi/wdi_G3_IS.AIR.PSGR.csv` | `54dca09a19b896dd6c35d9558cfadcf796c5c2a30c239b83a5bffdf3db04e16b` |
| `data/raw/wdi/wdi_G3_NY.GDP.MKTP.CD.csv` | `5e2ea5e9ed1c624601195252c4a86d5e934cc93a13b8efd72766dcd70c5dfe9e` |
| `data/raw/wdi/wdi_G3_SP.POP.TOTL.csv` | `c4e13a5c9c30cd76549666cec51787fa3369b8ff81d953304d5ac6ad69316d9b` |
| `data/raw/wdi/wdi_G3_ST.INT.ARVL.csv` | `74793ff52e334f21e55efec1fd71c1a14a964a423486bd324a26ed8f856e0b5a` |
| `data/raw/wdi/wdi_G3_ST.INT.RCPT.CD.csv` | `486221b593ce4e6ba47fa5c6f315b7e6ff7311709e5cd9589ed806834c791df6` |
| `data/raw/wdi/wdi_G4_IS.AIR.PSGR.csv` | `fc48adec44929bde9f299ec1d229271264c1cf4bf9c69318f4f8e41fea1ee615` |
| `data/raw/wdi/wdi_G4_NY.GDP.MKTP.CD.csv` | `953b1d536bf38a8131a2673fe47500d365f3f39fa2cc9b15f4f234c3722a227b` |
| `data/raw/wdi/wdi_G4_SP.POP.TOTL.csv` | `625122da7d288112e274f0991d1b4cf9f7e6b7823b4b4c28a8e68fcb4ee18ec0` |
| `data/raw/wdi/wdi_G4_ST.INT.ARVL.csv` | `1166dd6909561f5a865cdd15ab77bd5c27df23c5d6f60d54ca8d1bb8d3fdc555` |
| `data/raw/wdi/wdi_G4_ST.INT.RCPT.CD.csv` | `737994c36b7379a31ac98129aec1dff98dadb434d051e2d29fd6baa18dc89d30` |
| `data/reference/replication_reference_values.json` | `7b0c792437d210aa49ffe7cd4733dea27d5cdb722745cfad6fb789dc7f7107a4` |
| `docs/cases_catalogue_notes.md` | `0232abdc88f6d81add7b944df5429c2392d0e998e8cc1b693e2ce714b792e542` |
| `docs/causal_benchmark_search.md` | `0182e2d3d83e10c991cbf39468dbff298b8eb0ebc8eb2ad5bc18c6110b3ed73e` |
| `docs/investment_fx_notes.md` | `8dfd07f22de3be5b45031a4f1c7108dafc14a754a4dea8bf5f0cc2c3ef174cab` |
| `docs/thailand_baseline_notes.md` | `df676ec2caa067232ba8ea7c5761e2650c850b57038e6e16867b30413e9b4d3b` |
| `figures/case_effect_vs_capex.png` | `18c3d03f96d73dabad42538c6560c4e5684b2a4faa1007fc16cf5286e34690a9` |
| `figures/case_effects_forest.png` | `3ebd33db6c78552d46b00b9e5e3bb04bba7ad85038d6ae40dd421b271ffccb02` |
| `figures/case_estimator_comparison.png` | `7b1821013d8ed7abbad792b6455bdef6f1a688f04d6e38c14fc05dcaaa311663` |
| `figures/case_gap_paths.png` | `d8cdb43c23ba8c98d00d72a9389ead773bd38df115c86ead8c001408a88694ce` |
| `figures/impact_routes.png` | `eee97b844be7af26f6d29076417127f81f3740b51768ade2bc88b885b72f0baa` |
| `figures/impact_scaling_law.png` | `11e551dc12862a1cbd013e7ca5d9b7e0ffedd0537092644d90532ae14c51b0f2` |
| `figures/importance_bars.png` | `b0d54c5723ff428b96f65cc76207aa2f4b7e097a4b41ff89424e0d3c14d62c3a` |
| `figures/importance_penalty_path.png` | `b45a544a6a68648b749aef58ab388aa4e435b0ddd7318da41f626b2f5e8211ec` |
| `figures/importance_rank_correlation.png` | `32634f0402cb11e84679d0ec49130567369cffb324415b2a3fb553938410973c` |
| `figures/importance_selection_frequency.png` | `a32481c0826efd68bb593b5c914c74577aa7df51e7ed323045ec3752620fe7a4` |
| `figures/ot_feature_space.png` | `3df482e1d28219b467e548eed4392c92a5f15a876dbfdfb5b3d92c4e23430243` |
| `figures/ot_loco_validation.png` | `e27a34cd6e4ca331a86af25957b3f401ce066eed03a1c8e0b090617b984b3487` |
| `figures/ot_plan_heatmap.png` | `27be0a3021947e20597ba12060756955a4a143b3bb11b6a00a7b1e5ff4cdfd43` |
| `figures/ot_validation_null.png` | `2a51288f763c59bd7674a8f1eafc77a561e0979bcaeaca253ee5161d6e4a8969` |
| `figures/outcome_by_year.png` | `4e0a9fa14996974188c63b31ecdb2b8a6e41c71bb5d2775f1e23bd25aa0fdcd2` |
| `figures/outcome_coverage_heatmap.png` | `21a574a06df293667ae6e6070f9e6f4e4f104b3c24f7a7b55a489770cdb92fb6` |
| `figures/real_event_study_hk.png` | `e2876940b14bf42a96e60ea39447d3596e2d4a08eee38afa5ca596c398ca0957` |
| `figures/real_event_study_sh.png` | `fdd8018fac891dac92f1d9909b9dc3ed5f82b53945ba4dc067df6110047afbdb` |
| `figures/real_placebo_hk.png` | `70a292dd42898cf9b772d02588afa1b2e5b39ad6ec9cad09984e048431b2b28c` |
| `figures/real_scm_hk.png` | `233a6476c5258d26f72c9d24262595c73d1e649823bc0492a5299ee8ed001375` |
| `figures/real_scm_sh.png` | `1e031e330b638e12b80813fe99e975fe51b9cbe7fe1f4c69e270491b2120b0fc` |
| `figures/ref_event_study_hk.png` | `128847ceff92f764124034c023c25f1e415ce47962ceafc63cd710ac0d20f0ec` |
| `figures/ref_event_study_sh.png` | `146a93b1898e49aa79f97c82cd3629ecb1cbd0fcbc30447e52ed7813387e8e37` |
| `figures/ref_placebo_hk.png` | `bc8e13a2ffb66c7e0f2fa646376abd614822ab67823dfd2b5a6096c769decbfc` |
| `notebooks/00_build_global_panel.ipynb` | `fc9600af621d9dbf89cba1ba866c6b5cacdb414ca47c603b4d44b471fbb03f88` |
| `notebooks/01_replication_and_validation.ipynb` | `f1a92772cad40c5e1ac8a1e799f900b35707753d0cdb77bb4dde53aaae7781a7` |
| `notebooks/02_case_effects.ipynb` | `93d5483a62038f60a3f7c9bec3e47114fdcf77df43e4543522588a6a34285b1d` |
| `notebooks/03_feature_importance.ipynb` | `3d492e6ffcf14139e19fcef85276bf14e89ee3d3934dd272372407c631ea5d6a` |
| `notebooks/04_optimal_transport.ipynb` | `2dd8b6e22b19ee3ec791e32326debc8da9fc1b78e4141be688f3051ae7fc265f` |
| `notebooks/05_thailand_impact.ipynb` | `d4dae773e2a9dfddee40ab63018259ddf5eef61af22ff2a0cfa329694f8d2bb6` |
| `requirements.txt` | `d10ee69f9aaf3dcff762b6e1df489d51aafb4d7738d3bd0d50ae49f0281449bd` |
| `results/ambient_loeo.csv` | `d6ce302f2bc71e10f68d0c3c33d0d1bba7edad5a80555d849ff3f7c2b5803059` |
| `results/case_effects.csv` | `489a4d07127eb78e1ff9ec75c27dfbdffa3aadbfa001b15f26084b7bb2ad70bd` |
| `results/case_effects_robustness.csv` | `0788caee2f9bb4d7758426bc70a26d62c3b46d14de6324b4e8a80c06eaefb067` |
| `results/case_features.csv` | `c94fa5cff06e1532891925cfe47328397e749dc5a429110e3b7f37185d8e51ec` |
| `results/case_features_imputed.csv` | `92facc85bb953c1fb2b95b78597965dc30625e8404bffa1b94e8c3fadfc49472` |
| `results/case_investment_fits.csv` | `1ee29370210040d92b2fe6a45562c792047cebee0e2eb319f15d425e74b3755e` |
| `results/case_placebos.csv` | `c75081aa79277183db4b6fab16395771077639219b56a52397fe565121e94207` |
| `results/donor_eligibility.csv` | `4554e9fd6a6f01c48656a061bf55983d6f7488e5c4a8c2d2069be0459ed443ff` |
| `results/episodes.csv` | `d942ef2fe5bb5d126fc41528da030b4c57a75f735b2eff646ebec98c93e135ef` |
| `results/feasibility.csv` | `112db87394a14e4cb313305cf5a074adb8325ec2837e56e44d1f5bc7f644589c` |
| `results/feature_importance.csv` | `543d91b84f4310f93d84818e66a60563a909a4a15727eb170eb8c4a0ef4a876c` |
| `results/figure_inventory_00_build_global_panel.csv` | `ec1e929e81bc50f4b466382b055b1f963c4ebd6f1acd91d42a5c5ef6983b2f68` |
| `results/figure_inventory_02_case_effects.csv` | `cc0a1dad3815a3f6cbba23b5c8b45b9497c89710ae062d5f9ecfafd95b20d989` |
| `results/figure_inventory_03_feature_importance.csv` | `e28ee274ecc1335bcf3f96c168d671f186af1750d6885466710001418a88aa26` |
| `results/figure_inventory_04_optimal_transport.csv` | `cad79a17db63475735eff21812604ebdcb5ef6cbf078a335e2047d9adad122b2` |
| `results/figure_inventory_05_thailand_impact.csv` | `6df029faf773a53122cc6c66e7c7000a2b04f5e9a6b8124ba090e11ff63579fe` |
| `results/importance_diagnostics.json` | `8e3ae05daf733c319f4d724c1d983fd2b5f0865ac5e66d4cd91ce205f99a1331` |
| `results/imputation_report.csv` | `1dd00c253bd0c68957522e3fe9856df80cc3f57010890586672650a9b330d2b6` |
| `results/loco_fold_weights.csv` | `a9ae0b87e6adb3f9da64bd01a934a7be22713a02d58c816dfe11ab6662bd0aad` |
| `results/loco_ratios.csv` | `d5c84486144eaf5d38efd1ff2b7a50ed61a15a457aa7c46ce5bfe342aa5689ba` |
| `results/loco_ratios_rel.csv` | `380a975ebb17c362ab8c0516f2529086ff25c5343f4f58de45e5c73d1dea21f4` |
| `results/loco_summary.csv` | `e930aa2454a7174cb36c676bd1b43929c7ce4fe52f97e5d50f33b0df11492502` |
| `results/loco_summary_rel.csv` | `57f46821f2180b6c5bb666600d1f43a5603de4c74edc33986e46ceb3c958fb5f` |
| `results/loco_table.csv` | `decbbf86725d6e099e250ef3cb331784c8b17f648fb8ccb95c21236bad2a4d29` |
| `results/loco_table_rel.csv` | `974991860e46c040e6d82562267f738e5f8e88ef59547525eaec78800f21f1d0` |
| `results/overlap_test.json` | `f97abdbd845ae6640c9a616200608bf0a3ff3a4e30290260efbcf4430e2fcf3a` |
| `results/panel_coverage.csv` | `37ded335eabf606430d148ec35e049c842b260efe878e758c043e9cddf9cf345` |
| `results/reference_features.csv` | `08656183bebce926ccf5742c84b480f57996250f73434063bad849da01d6dd2d` |
| `results/route_decision.json` | `38e86fdf4a7d1a55380482a56d6f2e07f7e8be8cc42e8585c5e4f1a698ad59cc` |
| `results/stamps/00_build_global_panel.json` | `1ea794a5de1c7b14b143124b16635dabf1cb773b4436bcec1a063addd1adaf53` |
| `results/stamps/02_case_effects.json` | `c8995d389fbdc6857e4b54af8694822a0bb9beb14cfc921305fe66628d775b3a` |
| `results/stamps/03_feature_importance.json` | `25fcecd6f8a53f7bfec9ea796c879df25fbf27a1f0015ab12d9f9fe5f41e3584` |
| `results/stamps/04_optimal_transport.json` | `75bbcbf3d18b1e72eb04e7bbb51c216a42a2e2a23d17e93df386315e6f8dac66` |
| `results/stamps/05_thailand_impact.json` | `ca3767a20d6d6b1e6f58b382e07aed6304be78330bbd0323c98e891b7fc577c2` |
| `results/thailand_detection.csv` | `1a48f6607d2fa027a0d53e502c82ddb9fece5123bc4cf4baffbd0408eb42af71` |
| `results/thailand_impact.csv` | `33d4e59038f44a4f0bf7f5cf933a787cdf7330f76f6e4f24fdc2e5a0ea316e00` |
| `results/thailand_scenario_draws.csv` | `63af966fccbbd179ac90a0173699f0b43b8cc89e6e9b8e657af139fbb4ada686` |
| `results/thailand_scenario_null_draws.csv` | `2a9fe6263944c82826f1be91e4774251d16b6a85e1710c3baae68c3a16b6f7b0` |
| `results/thailand_scenario_usage.csv` | `7b5a7a6a6a4a1e0d7c0aab18fb9313b9cbcf656935431acc7416eb8d1ba579c4` |
| `results/thailand_scenarios.csv` | `efc2bc397545ff1ca11c23b4c75889c8ff610fd27aec3dac3c3eb21e91540bf3` |
| `results/thailand_summary.json` | `40d36f9a3e771f5e1b8399de24dc04aa65893184eed233ea0dee8efc6be84b4d` |
| `results/thailand_target_cloud.csv` | `12d551bf826bf0f6ed90c2a5f35b542dfa3da7dd3980a2071de39f1ee46bbc7e` |
| `results/thailand_transport.json` | `9c5ffa0d7ba09252dcca0b201ce23f6eaadcec568d727dfbb4bfee6ff77e70f8` |
| `results/thailand_transport_draws.csv` | `1f26c87cda697821cf8f3284b5bd8229ee11724e1133911107d814aa9139b013` |
| `results/transport_plan.csv` | `9fe4c69ea3424e14623b3cb744157c742d341a62a633d1064783a10b72ef112b` |
| `results/transport_sensitivity.csv` | `12b3fd8b2c583cdccdfbdbd2c860fb2bfb7634d3eeb916c2e9255a41a26e4276` |
| `results/transport_usage.csv` | `5dd02ca670fc49960c1c56f8ee968b7426c4c3fe2dc65686abfbf185c7d34a86` |
| `results/transport_weights.csv` | `9092d93e833844924ca0d864af5aab0b1e4cace90f4406a8eeabeaa92e051f78` |
| `scripts/verify_wdi_files.py` | `06ba7bf129623320659241d6a26324391dd90fbb78a02ad59fb6db932b6665d5` |
| `src/dtt/__init__.py` | `a546f33411d27cb0ca1fde1d39482e392cc7eea5d2d5fccbe5a8805296477f89` |
| `src/dtt/appraisal.py` | `9603e845f9ca43d229c6771d95cc205acd603aabcecef3718ee5d13ce2deac2f` |
| `src/dtt/cases.py` | `381e11984e5ff0ebe78e6adb648e447ccbaf87aa416691bc30d47bbe623e3b7e` |
| `src/dtt/did.py` | `57422bd2f941be629f1479b18e6a486c16acde0e5d775a8b7a3c7cde0aea6be4` |
| `src/dtt/effects.py` | `98822093671488fce84bf30253bd8285f04d3bb46b83b0469d4024cff2e8978b` |
| `src/dtt/impact.py` | `624f02a1686a6e0c384e403cb00b4afbfc51f07b2f03ae7f38225aba9aa9d5dc` |
| `src/dtt/importance.py` | `ca8daf3182898bc0001b908a4571940eca881e52f2de62ef3da5d109e64bab66` |
| `src/dtt/meta.py` | `2b5b58cd9ce5771553c4be2bbeaa2a387b6294b9ee70af7347efac727310be03` |
| `src/dtt/panel.py` | `703e0f381838e74c6d7d1541fa1598c702460a2a0b048050666e3aa051d39ae1` |
| `src/dtt/replication.py` | `fd1de9febf171e9bada1de739cf735543e89049297154b6cb93963e232e5d7c0` |
| `src/dtt/scm.py` | `d05224f19b58b1af8e5f6b023e00e536b49ca42ce4ed0bda04e75dc23eec562c` |
| `src/dtt/stata_compat.py` | `6da80499e41bf73036d3d15cd9d5d63571e37fcf7932d75fd130455b4026c0a3` |
| `src/dtt/thermal.py` | `620c08df01ea49da30b7ae6e0cd9eb08f68b457a6ba4e3e73b6fa5a0ac692220` |
| `src/dtt/transport.py` | `8ae373124faff0ad4fe6d1ead4454e735437908afd3f1d57908280a752054231` |
| `src/dtt/workflow.py` | `edd063674f6c2ddab45faebf3051a952c720248448eb567465ca26731f83543b` |
| `tests/conftest.py` | `d9f9b3ccdacfe4ccda6f2962f0895322838a414e21382dc3817bb31882cdb912` |
| `tests/handmade_panels.py` | `896e03eea13cf43c5ea30517b0fb0df780fd2bd736cec1998d1f86001e04e247` |
| `tests/handmade_world.py` | `d6019eaae092979d69fe1a386fb809966ca99b6b21317d6210a24df74eb7154b` |
| `tests/test_appraisal.py` | `ae4a5ba8d01dc37ba9e3be307acd4b77c0ee3323e8aad97d6b5dd0480467c700` |
| `tests/test_cases.py` | `2866b7ea4638394ccf746c0c05c0ced56f84b22c1962ed6d632d440a77effdf6` |
| `tests/test_did.py` | `f6a521e4540297471a07a4b0fdfc173154365fb6b813f14fbde61cda7254c780` |
| `tests/test_effects.py` | `51e07f359ef6022a63a1943d02c8c99c80ed7df96aa35beac0668a197e820bbf` |
| `tests/test_impact.py` | `f364ec3ea28d223538c2d2b883c74f7ff7a098ef086e66d34d27e5e0c69658cf` |
| `tests/test_importance.py` | `588713a0ab978188c68cf65ce415954b117c74566fb0346f69298181a66888f1` |
| `tests/test_meta.py` | `8aa967d38709cdcd792ee7d165f272e2cc416a0da6f5b7f078a312bbf3d890d3` |
| `tests/test_notebooks_static.py` | `16e68f5c256b4899557176e41fb496dc33ee934367acb6e624ff025ade9abc8e` |
| `tests/test_panel.py` | `932402cb360cde2c685b250759597f7f1f532535db6556287e29a6670243e465` |
| `tests/test_replication.py` | `e6b98fa4b74a11b64231c8686ad78b84fe08ed0d60b99f83793cbf386200bda2` |
| `tests/test_scm.py` | `f3e1a669417889eb085b2c07d14931a27fb2d27d1fb2ec8fb38fac0aecfed603` |
| `tests/test_se_conventions.py` | `b25a7e9c9b958034195b3d3f087b07044e8b9938275e606fdbf10f90135c71bb` |
| `tests/test_stata_compat.py` | `cde2b4cbd5db3650992e30745cb0b814944f7d4bc41fa59e1c2f7999791063d9` |
| `tests/test_thermal.py` | `676a38647bf634c81b4987f829a71fa8508dead9b02bd248933f0aa5912a7a73` |
| `tests/test_transport.py` | `3057f6015bffc45e4168eb8771c7aed32b6708dcf82c1c014f5aca13186569a0` |
| `tests/test_verify_wdi_files.py` | `d3104d0c7ad7ec44d1e1ab5472ffc7a6419901a20486616cbb84da52a3ef5739` |
| `tests/test_workflow.py` | `63a2fc5505b68accb79061ce10c0c79defa94a3a864eebf91422a3bfb2de9a72` |

