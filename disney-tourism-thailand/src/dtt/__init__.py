"""Theme-park effects on tourism receipts and their transport to Thailand.

The package contains a Python implementation of a difference-in-differences and
synthetic-control analysis of Disneyland openings, a global extension of that
analysis to theme-park and resort openings in a World Bank country panel, and the
tools that move the estimated effects to the Thailand economy.

Estimation
----------
stata_compat
    Regression and panel estimators with the standard-error and degrees-of-freedom
    conventions of the original analysis (pure NumPy).
did
    Two-way fixed-effects difference-in-differences, pre-trend tests, wild cluster
    bootstrap-t and event studies.
scm
    Synthetic control with nested and direct weight solvers, placebo-in-space
    inference and ridge augmentation.
replication
    Panel loading, the full single-country analysis and the equivalence check
    against the reference values.
simulate
    Simulated single-country panels for validating the estimators.

Global extension
----------------
panel
    Country-year panel built from the raw World Bank files.
cases
    Cases catalogue, treatment episodes, feasibility rules, donor pools and
    economy features.
effects
    Synthetic-control effect of every feasible case with placebo-in-space
    inference.
simulate_global
    Simulated worlds with known case effects for end-to-end checks.

Transport to the target economy
-------------------------------
importance
    Feature stacking and permutation importance averaged over the penalty path.
transport
    Unbalanced optimal transport with an importance-weighted cost, leave-one-group-out
    validation and bootstrap intervals.
meta
    Random-effects and Bayesian hierarchical meta-analysis, meta-regression and a
    scaling law, used when the importance route is not supported.
impact
    Thailand baseline, absolute and relative scaling of an effect, minimum
    detectable effect and the rule that selects the primary route.

Running
-------
thermal
    Host temperature probes and a guard that pauses long computations.
workflow
    Run context shared by the notebooks (paths, mode, seeds, guard, saving).
"""

__version__ = "0.2.0"
