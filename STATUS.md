# Project status

Last updated: 2026-10-09. This page tells the seminar group where each project stands, what is still open and where you can help. Both projects are work in progress and every number below is preliminary.

## 1. Disney and tourism receipts: lessons for Thailand

Folder: [disney-tourism-thailand](disney-tourism-thailand). It builds on the first seminar topic, "Estimating the Impact of Disneyland on Tourism Revenue: Evidence from Hong Kong and Shanghai and Lessons for Thailand".

**Question.** How much does the opening of a theme park or destination resort change a host economy's international tourism receipts (percent of GDP), and what does that imply for the proposed THB 300 billion park complex in Thailand?

**Method in short.** Each opening worldwide gets its own synthetic-control estimate on a World Bank panel of 96 economies (1995 to 2024; 2020 to 2022 are left out), with placebo tests for inference. The case effects are then moved to Thailand by one of two routes. A rule fixed in advance chooses the route: importance-weighted optimal transport if its checks pass, otherwise an "ambient" route (a pooled hierarchical model, a scaling law in the cost share of GDP, or a meta-regression). The README in the project folder has the details.

**Where it stands**
- The code package, six Jupyter notebooks (no Stata needed any more) and the tests exist. The module test files passed when they were last run.
- A first end-to-end run on the dataset is finished. Tables are in `disney-tourism-thailand/results`, figures in `disney-tourism-thailand/figures` (start with `impact_routes.png`).

**First result of the run on the dataset (preliminary)**
- Sample: 44 openings (episodes), 27 feasible, 19 in the primary sample, in 19 economies. The importance route needs at least 20, so the rule picked the ambient route. The importance model also showed no usable signal (cross-validated R2 0.056, permutation p-value 0.94).
- Selected estimator: the scaling law in the cost share. For the THB 300 billion complex (about 1.6 percent of GDP) it gives a median of about +US$2.2 billion a year in tourism receipts (+0.4 percent of GDP, +3.4 percent of receipts). The 90 percent interval runs from about -US$6.2 to +US$10.5 billion, so it includes zero.
- The other routes disagree in sign: pooled mean about -0.7, meta-regression about -4.9 and transport about -5.7 US$ billion a year. Thailand's receipts share (12.1 percent of GDP) is above every opening in the sample, so the meta-regression and the transport route extrapolate.
- Reading: the data point to a positive effect that grows with the cost of the project, but they cannot separate it from zero. An after-the-fact check on Thailand's own data would detect an effect of this size only about 7 percent of the time. The officials' figure of 1 percent of GDP a year is a reference line in the figure, not a result.

**Open items**
- Final verification: a full test run on the final version, the same results on different CPU math libraries, a clean install on Python 3.12 and 3.13, and an independent review of the results.
- Data notice: attribution for the World Bank data (CC BY 4.0), the origin of `data/raw/disney_did_panel.csv`, and the choice of a code licence.
- Check of the 20 World Bank files against the official API (a script for this is still to be written).
- Review the 37 reference values of the original Stata analysis that notebook 01 does not reproduce (1179 of 1216 match): synthetic-control weights, balance and placebo ratios that differ by 0.3 to 1.6 percent, and one bootstrap confidence-interval endpoint. Read off the notebook output where Hong Kong ranks in the original table.
- Only 53 of the 96 economies have a complete outcome series for 1995 to 2019, so the donor pool is smaller than first planned.
- Dollar costs of the openings are derived estimates, not reported figures.

**How to run.** See `disney-tourism-thailand/README.md`. A full run on the dataset takes one to two hours on a laptop (notebook 03 took 64 and 99 minutes in two runs), almost all of it in notebook 03. Two runs produced identical tables.

## 2. Investor horizon of Disney's shareholders (SEC Form 13F)

Folder: [13F_investor_horizon](13F_investor_horizon).

**Question.** Are the institutions that hold Disney stock long-horizon investors at the time Disney announces large park investments? The test looks at two announcements: the Hong Kong Disneyland expansion (22 November 2016) and the USD 60 billion Parks and Experiences plan (19 September 2023).

**Method in short.** From the SEC's quarterly Form 13F data sets the script rebuilds each manager's share holdings at each quarter-end, computes the manager's quarterly portfolio churn (the measure of Gaspar, Massa and Matos, 2005), averages it over the four quarters before the announcement, sorts managers into short, medium and long horizon terciles, and measures how much of Disney's stock each group holds. Disney is also ranked against the 500 largest 13F stock holdings.

**Where it stands**
- The pipeline runs for both announcements and the results are written to `13F_investor_horizon/results` (last run on 2026-10-05). The error logs are empty.
- Two robustness settings are produced next to the main one: a two-quarter churn window and a capped alternative churn formula.
- Main setting, 2016 and 2023:

| Measure | 2016 | 2023 |
| --- | --- | --- |
| Share of Disney's shares held by 13F filers | 57.8 % | 61.3 % |
| Held by long-horizon managers (share of shares outstanding) | 39.8 % | 42.0 % |
| Held by short-horizon managers | 3.0 % | 4.9 % |
| Disney's holder-weighted average churn | 0.154 | 0.122 |
| Median churn of all managers | 0.249 | 0.187 |
| Share of the 500 largest holdings whose holders churn less than Disney's | 9 % | 14 % |

- Reading: Disney's holders turn their portfolios over less than the typical manager, and long-horizon managers hold the largest block. These are descriptive numbers; nothing is tested against a counterfactual yet.

**Open items**
- No README in depth, no tests and no independent check of the churn numbers yet. The folder has two script versions; `run_13f_v2.py` is the later one and wrote the current results.
- The link between the horizon evidence and the question about capex announcements is not yet worked out (for example comparison firms, event-window analysis, and the literature).
- Large intermediate files used for cloud computing are not stored in this repository (they are regenerated by the script).

**How to run.** See `13F_investor_horizon/README.md`. The SEC requires every automated download to identify itself, so create a one-line file `sec_user_agent.txt` with your name and e-mail address first; git ignores that file.

## How you can help

- Read the project README, run the project once and tell the group where it breaks.
- Check a claim: pick one number on this page and trace it back to the table or script that produces it.
- Take an open item above and claim it in a GitHub Issue (template "Task"), so that two people do not work on the same one.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md): work on a branch, open a pull request, and do not hand-edit generated results.
