# Theme-park openings and international tourism receipts: an effect estimate for Thailand

## Purpose

This repository estimates how much the opening of a large theme park or resort changes the international tourism receipts of the host economy, and it applies the estimate to a theme park that has been proposed for Thailand. The estimate rests on past openings. For every opening that the World Bank data can follow, the analysis compares the host economy with a weighted combination of economies that had no opening, and it then asks which past openings resemble Thailand. The result is a range for the annual change in Thailand's international tourism receipts, expressed in percentage points of GDP, as a percentage of the receipts that would have occurred without the park, and in US dollars and baht. The analysis also reports whether a change of that size could be told apart from noise.

The outcome is international tourism receipts as a percentage of GDP. The effect of an opening is the mean gap between the observed outcome and its synthetic counterfactual over the years after the opening. The window is at most five years long, ends before 2020 and ends before the next episode in the same economy begins. The package also reproduces an earlier single-country analysis of Disneyland openings and checks it against reference values stored in data/reference.

## Design

1. **Cases and episodes.** The cases catalogue lists openings of theme parks, multi-park resorts, integrated resorts, destination resorts and mega attractions, with the opening date, the cost and a grade for the evidence. Openings in one economy that follow each other within five years are merged into one treatment episode; openings before 2000 stay separate and are not estimated. Feasibility rules require at least five pre-opening years with data, at least three post-opening years, a donor pool of at least fifteen economies and no other opening of the same economy inside the post-opening window. The primary sample consists of the feasible episodes whose synthetic control fits the pre-opening years well, which means that the pre-opening root mean squared prediction error is at most twice the median of the placebo pre-opening errors, and whose effect is identified. An episode with an exact pre-opening fit and fewer pre-opening years than donors has many weight vectors that fit equally well, so its effect has no unique value; such an episode is set aside, and the smallest and largest effect over the exact-fit weights are reported in the columns `att_pp_min` and `att_pp_max`.
2. **Synthetic control with placebo inference.** The weights of the donor economies are fitted on the pre-opening years of the outcome. Every donor is then treated in turn as if it had the opening, and the standard deviation of these placebo effects is the standard error of the effect of the episode. The ratio-based placebo p-value is not used because the pre-opening fit is often exact; the rank of the mean gap among the placebo gaps is reported instead.
3. **Feature stacking with lambda-averaged importance.** To learn which features of the host economy modify the size of the effect, a stacked ensemble of an elastic net, a ridge regression, a shallow random forest and a shallow gradient boosting model is fitted with the economy as the grouping unit of the cross-validation; the folds are random assignments of economies, repeated and averaged, and the weights of the stack are fitted inside each training fold. The importance of a feature is the increase of the held-out error when the feature is permuted, averaged over the elastic-net penalty path with weights that decrease with the cross-validated error. Features whose absolute correlation exceeds 0.8 form a cluster that is also permuted as a whole. A group bootstrap, a label permutation test and a split-half stability check accompany the estimate, and a usability verdict states whether the importance vector may be used.
4. **Unbalanced optimal transport with an importance-weighted cost.** The effects of the source episodes are moved to the yearly feature vectors of the target economy by entropic optimal transport. The cost between an episode and a target point is the squared distance of the standardised features, each feature weighted by its importance. The source side of the problem is relaxed, so an episode may send more or less than its own share, and the target side is enforced. The transported effect is the mean of the source effects weighted by the mass that arrives. Leave-one-economy-out validation compares the rule with the plain mean of all episodes, with nearest-neighbour and kernel predictors and with transport that uses uniform weights; in every fold the importance weights and the entropic regularisation are learned again from the training economies only. The predictive distribution for Thailand adds the leave-one-economy-out prediction errors of the rule to the transported effect, a bootstrap over economies describes the sampling variability of the transported effect, and the placebo effects give a transported null distribution. The intervals of the error ratios are descriptive and are not a test.
5. **Route rule fixed in advance.** The primary route is the importance-weighted transport route if and only if the importance diagnostics report a usable model, the leave-one-economy-out RMSE of the weighted transport is at most 0.95 times that of the plain mean and of the uniform transport, the primary sample has at least 20 episodes (below about 20 the split-half criterion of the importance model has no power), and the Thailand target lies inside the support of the cases (an effective number of at least two sources behind the transported effect and no heavily weighted feature outside the range of the cases). Otherwise the primary route is the ambient route. The decision is computed and printed before any estimate for Thailand is shown, and the other route is always reported next to the primary one.
6. **Ambient fallback.** The ambient route pools the case effects with a Bayesian normal hierarchical model (A1, with a weakly informative prior for the mean: centre 0 and scale five times the larger of the standard deviation of the effects and the median standard error), fits a scaling law through the origin in the cost as a share of GDP (A2), or fits a meta-regression on the cost share and the receipts share with a ridge penalty (A3). The estimator is selected by leave-one-economy-out RMSE: A2 or A3 replaces A1 only if its RMSE is at least 5 percent below that of A1.
7. **Scaling and detection.** The effect is scaled to US dollars and baht with the GDP and the receipts of the baseline years 2019 and 2024. The minimum detectable effect and the probability of detection come from the transported placebo distribution (the empirical null); the value from the normal approximation is reported next to it. The claims that officials have made about the park appear only as a reference scale; no study stands behind them.

## Repository layout

- `README.md`: this file.
- `requirements.txt`: Python packages with minimum versions.
- `.gitignore`: files that are not kept under version control.
- `src/dtt/`: the Python package, with one module for each step of the analysis.
  - `panel.py`: builds the country-year panel from the raw World Bank files.
  - `cases.py`: catalogue, treatment episodes, feasibility rules, donor pools and economy features.
  - `scm.py`, `effects.py`: synthetic control, placebo inference and the effect of every case.
  - `importance.py`: stacked ensemble and lambda-averaged permutation importance.
  - `transport.py`: unbalanced optimal transport, validation, bootstrap and overlap test.
  - `meta.py`, `impact.py`: ambient meta-analysis, scaling to US dollars and baht, detection and the route rule.
  - `did.py`, `replication.py`, `simulate.py` (synthetic test panels) and the estimation primitives they use: the single-country analysis and its equivalence check.
  - `simulate_global.py`: synthetic test worlds with known effects for the unit tests.
  - `thermal.py`, `workflow.py`: temperature guard and the run context shared by the notebooks.
- `notebooks/`: the analysis, to be run in this order.
  - `00_build_global_panel.ipynb`: builds the panel and records which economies can serve as donors.
  - `01_replication_and_validation.ipynb`: reproduces the single-country analysis and checks it against the reference values.
  - `02_case_effects.ipynb`: episodes, feasibility, the effect of every case, robustness variants and economy features.
  - `03_feature_importance.ipynb`: importance of the features and the weights of the transport cost.
  - `04_optimal_transport.ipynb`: validation of the transport rule, the transported effect for Thailand and its uncertainty.
  - `05_thailand_impact.ipynb`: route decision, ambient routes, scaling to US dollars and baht, detection and the reference comparison.
- `tests/`: the pytest suite, which runs on synthetic test data and stored reference values, with one check of the single-country analysis on `data/raw/disney_did_panel.csv`. `test_notebooks_static.py` checks that the shipped notebooks have no stored outputs and that no file of the repository holds an em dash or a token-like string.
- `data/raw/`: the World Bank indicator files in `wdi/` and the panel of the single-country analysis.
- `data/processed/`: the cases catalogue, the derived dollar costs, the Thailand baseline and the proposal file.
- `data/reference/`: the numbers that the single-country analysis must reproduce.
- `docs/`: notes on the catalogue, the derived costs, the Thailand baseline and proposal status, and a search for published causal estimates.
- `results/`: tables and JSON files written by the notebooks.
- `figures/`: figures written by the notebooks.

## Data and provenance

- **World Bank indicators.** The panel uses five indicators from the World Development Indicators in the vintage of 2026-07-13: international tourism receipts (ST.INT.RCPT.CD), international tourism arrivals (ST.INT.ARVL), GDP in current US dollars (NY.GDP.MKTP.CD), population (SP.POP.TOTL) and air passengers carried (IS.AIR.PSGR). They cover 96 economies in four country groups, which gives twenty files in `data/raw/wdi/`, next to `country_metadata.csv` and the fetch logs of each group.
- **Cases catalogue.** `data/processed/cases_catalogue.csv` lists 69 cases in 32 economies. Each case carries an evidence grade. Grade A needs an operator, government or annual report page that confirms the opening month and, where a dollar cost is given, the cost. Grade B needs two news or press release pages. Grade C rests on one secondary page or an encyclopedia. The catalogue has 20 rows of grade A, 28 of grade B and 21 of grade C. It flags 54 rows as feasible on the panel by their opening year; merging the openings from 2000 onward gives 44 episodes, of which 31 meet the year rules, and notebook 02 applies the donor and fit rules to them on the panel. `docs/cases_catalogue_notes.md` lists the rows where sources conflict and the fields that are blank.
- **Derived dollar costs.** `data/processed/cases_investment_fx.csv` holds 27 costs derived from reported local-currency figures and official annual-average exchange rates. They are derived estimates and not reported costs. `docs/investment_fx_notes.md` describes how they were built.
- **Thailand baseline and proposal.** `data/processed/thailand_baseline.csv` holds the adopted series for 2015 to 2025 (receipts, arrivals, GDP, exchange rate and receipts as a share of GDP) in long format, together with alternative definitions and cross-checks. `docs/thailand_baseline_notes.md` gives the sources, the reasons for the adopted series and the status of theme-park proposals as of 2026-10-08. `data/processed/thailand_proposal.json` records the three cost scenarios of the proposal, the status text (a proposal only, with no signed agreement, approved budget, named operator or site allocation found in the sources reviewed) and the claims that officials have made, which cite no study.
- **Single-country analysis.** `data/raw/disney_did_panel.csv` is the panel of nine economies used by that analysis, and `data/reference/replication_reference_values.json` holds the reference numbers with the number of decimals printed.

## Running the analysis on a desktop

### Installation

Create a virtual environment, activate it and install the requirements. The pinned packages need Python 3.12 or newer; the analysis was developed with Python 3.13. The requirements do not include a notebook front end: to open the notebooks interactively, install JupyterLab as well (`python -m pip install jupyterlab`); executing them from the command line needs only the listed packages.

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Tests

The test suite checks every module against synthetic test data. One test runs the single-country analysis on the panel file `data/raw/disney_did_panel.csv` and compares the results with the stored reference values. The whole suite takes about 6 minutes.

```
python -m pytest -q
```

### Notebooks

The notebooks run on the dataset in the repository: the World Bank files in `data/raw/wdi`, the cases catalogue and the other files in `data/processed`. Execute the notebooks in numerical order, because each one reads the files that the earlier ones wrote. The first cell of each notebook states its measured run time. A full run takes about an hour, almost all of it in notebook 03.

PowerShell:

```powershell
python -m ipykernel install --prefix $env:VIRTUAL_ENV --name dtt-venv
foreach ($nb in (Get-ChildItem notebooks\0*.ipynb | Sort-Object Name)) {
    python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=dtt-venv --output-dir executed $nb.FullName
    if ($LASTEXITCODE -ne 0) { break }
}
```

bash:

```bash
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name dtt-venv
for nb in notebooks/0*.ipynb; do
    python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=dtt-venv --output-dir executed "$nb" || break
done
```

The first command registers a Jupyter kernel inside the virtual environment, so that the notebooks run with the installed packages and not with another interpreter; the run context stops with a message when an installed package is older than the minimum in `requirements.txt`. The notebooks in `notebooks/` are shipped without stored outputs; the executed copies are written to `executed/`, which git ignores. Tables and JSON files land in `results/`, figures in `figures/` and the assembled panel in `data/processed/global_panel.csv`. The environment variables `DTT_RESULTS_DIR` and `DTT_FIGURE_DIR` redirect the tables and the figures to other directories.

## Outputs of the notebooks

Tables are CSV files and summaries are JSON files in `results/`; figures are PNG files in `figures/`. Each notebook also writes `figure_inventory_<notebook>.csv`, which lists the panels, axis labels and legend entries of its figures.

- Notebook 00: `data/processed/global_panel.csv`, `panel_coverage.csv` (with the number of holes and of negative values of each economy) and `donor_eligibility.csv`.
- Notebook 02: `episodes.csv`, `feasibility.csv`, `case_effects.csv` (effects, placebo standard errors, rank p-values, the exact-fit flag with the identified range, and the flags of the primary and sensitivity samples), `case_placebos.csv`, `case_effects_robustness.csv`, `reference_features.csv`, `case_features.csv`, `case_features_imputed.csv` and `imputation_report.csv`.
- Notebook 03: `feature_importance.csv` (importance, importance along the penalty path, selection frequency, bootstrap band, cluster and cluster importance), `importance_diagnostics.json` and `transport_weights.csv`.
- Notebook 04: `loco_table.csv`, `loco_summary.csv` and `loco_ratios.csv` (leave-one-economy-out validation for the effect in percentage points of GDP), the same three files with the suffix `_rel` for the relative effect, `loco_fold_weights.csv` (the weights learned in each fold), `transport_plan.csv`, `transport_usage.csv`, `transport_sensitivity.csv`, `thailand_transport.json`, `thailand_transport_draws.csv`, `thailand_null_draws.csv` and `overlap_test.json`.
- Notebook 05: `route_decision.json`, `thailand_impact.csv`, `thailand_summary.json` and `ambient_loeo.csv`.

## Thermal guard

Long computations call a guard from `src/dtt/thermal.py` that pauses them while the machine is too hot or, when no temperature is available, too busy. The guard reads the hottest CPU temperature from the first source that answers: the sensors that `psutil` reports under the names coretemp, k10temp, cpu_thermal and acpitz; the Linux files `/sys/class/thermal/thermal_zone*/temp` of the zones whose type names a processor, package or core sensor; on Windows the ACPI thermal zone (`MSAcpi_ThermalZoneTemperature`), read through PowerShell; and on Windows the CPU temperature sensors of LibreHardwareMonitor or OpenHardwareMonitor, when one of them is running. A source that fails is not asked again for 60 seconds. The guard also reads the CPU load and the memory use. Readings outside 10 to 150 degrees Celsius are ignored.

When the temperature is above 85 degrees Celsius the guard waits in steps of 5 seconds until a reading at or below 78 degrees is obtained; a step without a reading does not end the wait. When no temperature sensor answers, the guard waits in steps of 5 seconds while the CPU load of other programs is above 92 percent (the load of the analysis itself and of its child processes is subtracted) and otherwise lets the computation continue. A wait for the CPU load alone ends after 300 seconds: the guard logs the action `cpu_timeout`, prints one line, and does not wait for the CPU load again in that run. The guard raises an error only when a wait that began with a temperature above the limit exceeds 30 minutes. The guard prints one line when a wait starts and one when it ends. Every check appends a row with the time, the temperature, the CPU load, the memory use, the action and the waiting time to `results/thermal_log.csv`. A notebook calls the guard at most once every 10 seconds.

The guard is on by default, and `DTT_GUARD=0` switches it off. The variable accepts only 0 and 1. The guard never changes a result. Many desktop computers expose no temperature to ordinary programs; the guard then relies on the CPU load, and the first lines of each notebook print whether a temperature is available.

## Reproducibility

Every random draw uses a numpy generator seeded from a named seed in `src/dtt/workflow.py`: effects 0, importance 0, importance bootstrap 1, transport 0, transport bootstrap 2, overlap test 3, and meta-analysis 0. The settings cell of each notebook prints the seeds and the bootstrap sizes in use. No computation uses parallel workers. Every comparison of two computed numbers in the importance and transport steps uses an explicit tolerance, so that rounding noise does not change a verdict, a rank or a count; continuous results can still differ in the last digits between machines that use different numerical libraries.

## Limitations

- The estimates are conditional on a park of the stated cost opening. They are mean annual effects on international tourism receipts over the first five years after the opening, and they are not the total economic impact of a park.
- The case effects come from country-level synthetic controls with few pre-opening years. The pandemic years are excluded from every estimation window.
- The proposal is not a commitment. The cost in US dollars of the proposal and of the earlier cases is derived from a cost in local currency with an exchange rate of one year, so every dollar cost is an estimate.
- Cases in one economy are not independent. The validation leaves out whole economies, the bootstrap resamples whole economies, and the number of economies, not of episodes, governs the precision.
- The number of episodes is small relative to the number of features. The importance weights are therefore uncertain, and the route rule can select the ambient route when the importance model is not usable.
- The feature weights used for the transport to Thailand are learned on all primary episodes. In the leave-one-economy-out validation they are learned again from the training economies of each fold, so the validation does not favour the weighted rule through its weights. With few economies the learned weights are noisy, and a fold in which no weights can be learned falls back to uniform weights; the notebook counts such folds.
- An episode with an exact pre-opening fit and more donors than pre-opening years has no unique effect. It is left out of the primary sample and appears in the sample of all feasible episodes with its identified range.
- The predictive interval for Thailand carries the error of transporting an effect to an economy that was not among the cases, but it rests on a small number of economies. The intervals of the error ratios between methods are descriptive.
- When the cost of the proposal as a share of GDP lies outside the range of the cases, the scaling law and the meta-regression extrapolate.
- The claims of officials are shown as a reference scale only. They are not tested by the analysis.
