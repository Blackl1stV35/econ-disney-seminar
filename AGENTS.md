# Instructions for coding agents

This file is read by coding agents (Claude Code, Codex and similar) that work in this repository on behalf of a seminar member. Humans should read CONTRIBUTING.md; the rules are the same.

## Repository map

- `disney-tourism-thailand/`: Python package `dtt` in `src/dtt`, Jupyter notebooks in `notebooks/`, tests in `tests/`, data in `data/`, tables in `results/`, figures in `figures/`. Its own README explains the method and the commands.
- `13F_investor_horizon/`: two scripts and their results for the Disney 13F investor-horizon test. No tests yet.
- `STATUS.md`: where each project stands and what is open. Read it first. Update it when you change the state of a project.

## Set up and test (Disney project)

Python 3.12 or newer. From inside `disney-tourism-thailand/`:

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # bash: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q --ignore=tests/test_pipeline_sim.py     # about 5 minutes
python -m pytest -q                                          # adds the simulated pipeline, roughly 15 more minutes
```

Run the fast command before every pull request. Run the full command when you change a notebook, `src/dtt/workflow.py` or a table contract. Use simulated data (`DTT_WORLD=sim`) for experiments. A real-data run needs `DTT_RUN_REAL=1`, takes about an hour (almost all in notebook 03) and is not part of a normal change.

## Rules

1. Never commit a secret or personal data: tokens, API keys, passwords, e-mail addresses, the SEC contact file `sec_user_agent.txt`, or user names in file paths. A test scans the Disney project for token-like strings.
2. Work on a branch named `<owner>/<topic>` and open a pull request. Never push to `main`, never force-push, never rewrite history, never merge your own pull request.
3. Edit one project folder per pull request, and claim the task in a GitHub Issue before you start, so that two people or agents do not work on the same thing.
4. Do not hand-edit anything in `results/` or `figures/`; they are produced by the notebooks. Change them only in a pull request whose single purpose is a re-run, written by the person who ran it, with the commit used and the run time in the description. Do not mix code changes and result updates.
5. Do not edit raw data (`data/raw`, `data/reference`) or delete data files.
6. Style in the Disney project (tests enforce part of it): no em dash (U+2014) in any text file, complete plain sentences, NumPy-style docstrings, notebooks stored without outputs, figures in grayscale with axis labels, units and a legend, and no internal notes, history wording or TODO text in shipped text. Use the same plain style in the 13F project.
7. Keep claims honest: write what the code did and what the numbers say. Results are preliminary until the open items in `STATUS.md` are closed.
8. Add or update tests with every change to code. Do not weaken a test to make it pass.
9. Ask the human before you do anything that cannot be undone or that leaves the repository: changing GitHub settings, pushing to another repository, deleting branches, publishing data, or running something that costs money.
10. End commit messages with the attribution line that your tool prescribes.
