# 13F investor horizon: Disney

This folder tests whether the institutional holders of Disney (DIS) are long-horizon investors at the time of two large park-investment announcements.

- 22 November 2016: Hong Kong Disneyland expansion. Focal quarter-end: 30 September 2016.
- 19 September 2023: USD 60 billion Parks and Experiences plan. Focal quarter-end: 30 June 2023.

## Method

For each announcement `run_13f_v2.py`:

1. downloads the SEC's quarterly Form 13F data sets for the five quarter-ends up to the focal quarter, plus one later file so that late amendments are picked up;
2. rebuilds each manager's share holdings per quarter-end (option and principal-amount rows are excluded; a restatement amendment replaces the original report, a new-holdings amendment is added to it);
3. computes each manager's quarterly portfolio churn rate (Gaspar, Massa and Matos, 2005) for the four quarters up to the focal quarter and averages it;
4. sorts the managers into churn terciles (short, medium and long horizon) and measures how much of Disney's stock each group holds;
5. ranks Disney against the 500 largest 13F stock holdings on the same measures.

Three settings are produced: the main one (churn at a common price, four quarters), a robustness setting with a two-quarter window, and one with an alternative churn formula capped for extreme values.

## Files

| File | What it is |
| --- | --- |
| `run_13f_v2.py` | The current script (the later of two versions). |
| `run_13f.py` | The first version, kept for reference. |
| `run_full.bat`, `rerun_analysis.bat`, `rerun_analysis_v2.bat` | Windows launchers for the scripts. |
| `results/dis_horizon_summary.csv` | One row per announcement and setting: ownership, tercile shares, churn cut-offs, Disney's rank among the 500 largest holdings. |
| `results/summary_2016.json`, `results/summary_2023.json` | The same quantities with more detail. |
| `results/manager_churn_*.csv` | Churn of every manager, for each churn formula. |
| `results/dis_holders_*.csv`, `results/peers_*.csv` | Disney's holders by horizon group, and the comparison with the other large holdings. |
| `results/run_log.txt`, `results/errors_*.txt`, `results/inspect_report.txt` | Run log, error output (empty in the last runs) and the layout of one SEC file. |

## How to run

1. Install the packages: `python -m pip install pandas numpy`.
2. The SEC requires every automated download to identify itself. Create a one-line file `sec_user_agent.txt` in this folder with your name and e-mail address. Git ignores it.
3. In PowerShell, from this folder: `python run_13f_v2.py inspect` (one file, prints the layout and a Disney sample) or `python run_13f_v2.py full` (the whole pipeline, about 3 to 5 minutes once the downloads are cached). Downloads go to the temporary folder; results go to `results/`.

## Notes

- Large intermediate panels used for cloud computing (`results/cloud_transfer`) are not stored in the repository; the script writes them again.
- The 13F value field is reported in dollars from filings dated 3 January 2023 onward and in thousands of dollars before; the script handles the switch.
- The status of the project, including what is still open, is in the `STATUS.md` at the top of the repository.
