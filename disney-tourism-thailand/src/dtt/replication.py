"""Replication driver and equivalence check against the reference values.

``run_all`` re-runs every analysis of the original study on a panel with the
structure of ``data/raw/disney_did_panel.csv`` and returns the results under
slash-separated keys.  ``data/reference/replication_reference_values.json``
holds the numbers printed by an independent run of the original analysis under
the same keys, each with the number of decimals with which it was printed.
The first component of a key (``log1`` or ``log2``) names the run that printed
the number.  ``check_equivalence`` compares the two sets of numbers with
explicit tolerances and returns a PASS/FAIL table.

Tolerance rules (``TOLERANCES``)
--------------------------------
count
    integers (sample sizes, clusters, degrees of freedom, ranks): exact.
deterministic
    estimates, standard errors, test statistics, p-values and confidence
    limits of regression output: half a unit in the last printed digit of the
    reference value, plus a relative ``1e-9``.
degenerate coefficient rows
    A coefficient whose reference standard error is below ``SE_NOISE_RELATIVE``
    times ``max(1, |coefficient|)`` has a variance that is zero in exact
    arithmetic.  Its standard error passes when the Python value is at most that
    noise level, its t statistic is not compared, and the tolerance of its
    confidence limits is widened by ten times the noise level.  The computed
    value is rounding error, of order ``1e-8``, and differs between CPU
    kernels of the same package versions.
panel_stat
    ``sigma_u``, ``sigma_e``, ``rho``, ``corr(u_i, Xb)`` and the between and
    overall R-squared: as deterministic.  They follow the documented
    definitions of the ``xtreg, fe`` output and are checked only on the panel
    file.
mc_p, mc_ci
    wild-bootstrap p-values and confidence limits printed by ``boottest`` come
    from 9,999 random Webb draws.  The Python value is the exact (enumerated)
    result.  A p-value passes if it lies within ``n_sigma = 4`` binomial
    standard errors of the exact value; a confidence limit passes if it lies in
    the interval obtained by inverting the exact test at levels
    ``0.05 -/+ 4 sqrt(0.05 * 0.95 / 9999)``; both add half a unit of print
    precision.
scm_rmspe
    absolute ``1e-5``.  The comparison is one-sided in the direction in which an
    exact optimiser can differ: Python may be lower than the reference value,
    not higher.
scm_weight
    absolute ``6e-4`` (weights are printed to three decimals).
scm_balance, scm_gap
    absolute ``4e-3`` in percentage points of GDP, which bounds the effect of a
    weight difference of ``6e-4`` per donor.
scm_ratio
    relative ``2e-3`` for pre- and post-RMSPE and their ratio in the placebo
    table.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from . import did, scm
from .simulate import UNIT_NAMES
from .stata_compat import RegResult, XtregResult, factor_dummies

__all__ = [
    "REPO_ROOT",
    "DEFAULT_PANEL",
    "DEFAULT_TARGETS",
    "COLUMNS",
    "TOLERANCES",
    "SE_NOISE_RELATIVE",
    "SCM_SPEC",
    "load_panel",
    "validate_panel",
    "wide_block",
    "Results",
    "run_all",
    "load_targets",
    "target_values",
    "printed_decimals",
    "print_halfwidth",
    "classify_key",
    "check_equivalence",
    "summarize_equivalence",
    "format_check_table",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
#: CSV file with the panel of the original study (177 rows)
DEFAULT_PANEL = REPO_ROOT / "data" / "raw" / "disney_did_panel.csv"
#: JSON file with the reference values used as equivalence targets
DEFAULT_TARGETS = REPO_ROOT / "data" / "reference" / "replication_reference_values.json"

COLUMNS = ["unit_id", "year", "receipts_pct_gdp", "treated", "post", "treated_post", "study_case", "rel_year"]
Y = "receipts_pct_gdp"
TREATMENT_YEAR = {"Hong Kong": 2005, "Shanghai": 2016}

#: synthetic-control designs of the original analysis:
#: tag -> (study_case, unit ids of the block, first year, last year, first treated year, treated unit id)
SCM_SPEC = {
    "hk": ("Hong Kong", [2, 3, 4, 7, 8, 9], 1999, 2018, 2005, 2),
    "sh": ("Shanghai", [1, 5, 6], 2008, 2019, 2016, 5),
}

#: A standard error below this multiple of ``max(1, |coefficient|)`` is numerical noise.  The cluster-robust
#: variance of a coefficient that is absorbed by its own cluster is zero in exact arithmetic; the computed value
#: is the rounding error of a sum of squares, so its square root is of order ``1e-8`` and depends on the
#: floating-point kernel of the machine.
SE_NOISE_RELATIVE = 1.0e-6

TOLERANCES = {
    "count": {"abs": 0.0},
    "deterministic": {"rel": 1e-9, "floor": 1e-12},
    "panel_stat": {"rel": 1e-9, "floor": 1e-12},
    "mc_p": {"n_sigma": 4.0},
    "mc_ci": {"n_sigma": 4.0},
    "scm_rmspe": {"abs": 1e-5},
    "scm_weight": {"abs": 6e-4},
    "scm_balance": {"abs": 4e-3},
    "scm_gap": {"abs": 4e-3},
    "scm_ratio": {"rel": 2e-3},
}


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------
def load_panel(path: str | Path = DEFAULT_PANEL, validate: bool = True) -> pd.DataFrame:
    """Read the panel CSV into the standard column set and types.

    Parameters
    ----------
    path : str or Path
        CSV file with at least the columns of :data:`COLUMNS`; further columns
        are ignored.
    validate : bool
        True raises ``ValueError`` when the structure differs from the
        documented one (see :func:`validate_panel`).

    Returns
    -------
    DataFrame
        The columns of :data:`COLUMNS`, sorted by ``unit_id`` and ``year``.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"panel file lacks columns: {missing}")
    df = raw[COLUMNS].copy()
    for c in ("unit_id", "year", "treated", "post", "treated_post", "rel_year"):
        df[c] = df[c].astype(int)
    df["receipts_pct_gdp"] = df["receipts_pct_gdp"].astype(float)
    df["study_case"] = df["study_case"].astype(str).str.strip()
    df = df.sort_values(["unit_id", "year"]).reset_index(drop=True)
    if validate:
        problems = validate_panel(df)
        if problems:
            raise ValueError("panel structure differs from the documented structure:\n  " + "\n  ".join(problems))
    return df


def validate_panel(df: pd.DataFrame) -> list[str]:
    """List the structural problems of a panel.

    Parameters
    ----------
    df : DataFrame
        Panel with the columns of :data:`COLUMNS`.

    Returns
    -------
    list of str
        One message per departure from the documented structure (177 rows,
        nine units, the Hong Kong and Shanghai study blocks, the treatment
        variables); empty when the panel matches.
    """
    issues: list[str] = []
    if len(df) != 177:
        issues.append(f"expected 177 rows, found {len(df)}")
    if sorted(df["unit_id"].unique().tolist()) != list(range(1, 10)):
        issues.append("unit_id values are not 1..9")
    if df.duplicated(["unit_id", "year"]).any():
        issues.append("duplicate unit-year rows")
    hk = df[df.study_case == "Hong Kong"]
    sh = df[df.study_case == "Shanghai"]
    if sorted(hk.unit_id.unique().tolist()) != [2, 3, 4, 7, 8, 9] or len(hk) != 129:
        issues.append(f"Hong Kong block should have units 2,3,4,7,8,9 and 129 rows (found {len(hk)})")
    if sorted(sh.unit_id.unique().tolist()) != [1, 5, 6] or len(sh) != 48:
        issues.append(f"Shanghai block should have units 1,5,6 and 48 rows (found {len(sh)})")
    if len(sh[sh.year >= 2008]) != 36:
        issues.append("Shanghai window 2008-2019 should have 36 rows")
    for case, t0 in TREATMENT_YEAR.items():
        blk = df[df.study_case == case]
        if not (blk.rel_year == blk.year - t0).all():
            issues.append(f"rel_year != year - {t0} in the {case} block")
        if not (blk.post == (blk.year >= t0).astype(int)).all():
            issues.append(f"post != 1[year >= {t0}] in the {case} block")
    if not (df.treated_post == df.treated * df.post).all():
        issues.append("treated_post != treated * post")
    if sorted(df.loc[df.treated == 1, "unit_id"].unique().tolist()) != [2, 5]:
        issues.append("treated units should be 2 and 5")
    if df[Y].isna().any():
        issues.append("missing receipts_pct_gdp")
    return issues


# ----------------------------------------------------------------------------
# Results container
# ----------------------------------------------------------------------------
@dataclass
class Results:
    """Output of :func:`run_all`.

    Attributes
    ----------
    values : dict
        Slash-separated key -> number.
    meta : dict
        Key -> tolerance hints (class and Monte Carlo acceptance ranges) that
        override the key-based classification in :func:`check_equivalence`.
    objects : dict
        Fitted objects (regressions, bootstrap results, synthetic controls,
        placebo tables) for tables and figures.
    """

    values: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    objects: dict = field(default_factory=dict)


def _clean(x: float | None) -> float:
    """Convert a number or None (missing) to a float, with NaN for missing."""
    if x is None:
        return np.nan
    return float(x)


def _record_coefs(res: Results, prefix: str, tab: pd.DataFrame, rename: Callable[[str], str] | None = None) -> None:
    """Store the coefficient table of a regression under ``prefix/coef/<term>/<field>``."""
    for term, row in tab.iterrows():
        name = rename(term) if rename else term
        for f in ("coef", "se", "t", "p", "ci_lo", "ci_hi"):
            res.values[f"{prefix}/coef/{name}/{f}"] = _clean(row[f])


def _record_xtreg(res: Results, prefix: str, fit: XtregResult, rename: Callable[[str], str] | None = None) -> None:
    """Store the numbers printed by ``xtreg, fe`` for one fit."""
    v = res.values
    v[f"{prefix}/n_obs"] = fit.n
    v[f"{prefix}/n_groups"] = fit.n_groups
    v[f"{prefix}/n_clusters"] = fit.n_clusters
    v[f"{prefix}/obs_per_group_min"] = fit.obs_per_group[0]
    v[f"{prefix}/obs_per_group_avg"] = fit.obs_per_group[1]
    v[f"{prefix}/obs_per_group_max"] = fit.obs_per_group[2]
    v[f"{prefix}/F_df2"] = fit.F_df2
    for k in ("r2_within", "r2_between", "r2_overall", "corr_u_xb", "sigma_u", "sigma_e", "rho"):
        v[f"{prefix}/{k}"] = _clean(getattr(fit, k))
    for name, val in fit.sigma_u_variants.items():
        v[f"{prefix}/sigma_u_variant/{name}"] = _clean(val)
    _record_coefs(res, prefix, fit.table(), rename)


def _record_regress(res: Results, prefix: str, fit: RegResult) -> None:
    """Store the numbers printed by ``regress, vce(cluster)`` for one fit."""
    v = res.values
    v[f"{prefix}/n_obs"] = fit.n
    v[f"{prefix}/n_clusters"] = fit.n_clusters
    v[f"{prefix}/F_df2"] = fit.n_clusters - 1
    v[f"{prefix}/r2"] = fit.r2
    v[f"{prefix}/root_mse"] = fit.root_mse
    _record_coefs(res, prefix, fit.table())


def _record_boot(res: Results, prefix: str, br: did.WildBootstrapResult, reps_ref: int) -> None:
    """Store the numbers printed by ``boottest`` together with their Monte Carlo tolerance hints."""
    v = res.values
    v[f"{prefix}/df"] = br.df
    v[f"{prefix}/t"] = br.t_stat
    v[f"{prefix}/p"] = br.p_value
    v[f"{prefix}/ci_lo"] = br.ci[0]
    v[f"{prefix}/ci_hi"] = br.ci[1]
    res.meta[f"{prefix}/p"] = {"kind": "mc_p", "reps": reps_ref}
    if br.ci_band is not None:
        res.meta[f"{prefix}/ci_lo"] = {"kind": "mc_ci", "accept": br.ci_band[0]}
        res.meta[f"{prefix}/ci_hi"] = {"kind": "mc_ci", "accept": br.ci_band[1]}


def wide_block(df: pd.DataFrame, case: str, units: list[int], start: int, end: int) -> pd.DataFrame:
    """Outcome matrix (years by units) of one study block over a balanced window.

    Parameters
    ----------
    df : DataFrame
        Panel with the columns of :data:`COLUMNS`.
    case : str
        Study case (``"Hong Kong"`` or ``"Shanghai"``).
    units : list of int
        Unit ids of the block, in the column order of the result.
    start, end : int
        First and last year of the window.

    Returns
    -------
    DataFrame
        Index ``year``, columns ``units``.

    Raises
    ------
    ValueError
        If a unit lacks a year of the window.
    """
    blk = df[(df.study_case == case) & df.year.between(start, end) & df.unit_id.isin(units)]
    W = blk.pivot(index="year", columns="unit_id", values=Y)[units]
    if W.isna().any().any():
        raise ValueError(f"{case} window {start}-{end} is not balanced")
    return W


_wide_block = wide_block


def _record_scm(res: Results, prefix_scm: str, prefix_log2: str, case_tag: str, W: pd.DataFrame, treated: int, trp: int, label_by_id: bool) -> scm.SCMResult:
    """Fit the synthetic control of one case and store the items of both runs under the two key prefixes."""
    units = [treated] + [u for u in W.columns if u != treated]
    Wm = W[units]
    years = Wm.index.to_numpy()
    pre = years < trp
    y1 = Wm[treated].to_numpy()
    Y0 = Wm[units[1:]].to_numpy()
    fit = scm.synth(y1, Y0, pre, method="both", n_starts=6, seed=0)
    v = res.values
    for pfx in (prefix_scm, prefix_log2):
        v[f"{pfx}/rmspe"] = fit.rmspe_pre
    for j, u in enumerate(units[1:]):
        v[f"{prefix_scm}/weight/{UNIT_NAMES[u]}"] = fit.w[j]
        v[f"{prefix_log2}/weight/{u}"] = fit.w[j]
    for pfx in (prefix_scm, prefix_log2):
        for i, yr in enumerate(years[pre]):
            name = f"{Y}({yr})"
            v[f"{pfx}/balance/{name}/treated"] = y1[pre][i]
            v[f"{pfx}/balance/{name}/synthetic"] = float((Y0[pre] @ fit.w)[i])
    res.objects[f"scm/{case_tag}"] = {"fit": fit, "years": years, "units": units, "pre": pre, "W": Wm}
    return fit


# ----------------------------------------------------------------------------
# The full analysis
# ----------------------------------------------------------------------------
def run_all(df: pd.DataFrame, boot_reps: int = 9999, seed: int = 42, exhaustive: bool = True) -> Results:
    """Run every analysis of the original study on a panel.

    The analyses are the three difference-in-differences regressions with the
    report-formula standard error, the two pre-trend tests with normal
    inference, the four fixed-effects regressions with wild cluster bootstrap-t
    (null imposed, Webb weights), the two pre-trend regressions with t(G-1)
    inference and wild bootstrap, the two event studies, the two synthetic
    controls and the placebo tables of both runs.

    Parameters
    ----------
    df : DataFrame
        Panel with the columns of :data:`COLUMNS`.
    boot_reps : int
        Number of draws of the Monte Carlo wild bootstrap kept in
        ``objects["boot_mc"]``.
    seed : int
        Seed of the Monte Carlo wild bootstrap.
    exhaustive : bool
        True computes the bootstrap values compared with the reference values
        by enumerating every weight vector; False draws ``boot_reps`` vectors.

    Returns
    -------
    Results
        ``values`` holds the numbers under the keys of the reference file,
        ``meta`` the tolerance hints of the bootstrap values and ``objects``
        the fitted objects.
    """
    res = Results()
    v = res.values
    d = df.reset_index(drop=True)
    hk = d[d.study_case == "Hong Kong"]
    sh = d[d.study_case == "Shanghai"]
    core3 = hk[hk.unit_id.isin([2, 3, 7, 9])]
    sh_bal = sh[sh.year >= 2008]

    v["log1/panel/year_min"] = int(d.year.min())
    v["log1/panel/year_max"] = int(d.year.max())

    # --- log1/friend_did: DiD with the report-formula standard error -------------
    res.objects["friend_did"] = {}
    for tag, sub in (("1a", hk), ("1b", core3), ("1c", sh)):
        r = did.friend_did(sub, label=tag)
        res.objects["friend_did"][tag] = r
        p = f"log1/friend_did/{tag}"
        v[f"{p}/n_obs"], v[f"{p}/n_clusters"], v[f"{p}/df"] = r.n, r.n_clusters, r.df
        v[f"{p}/coef"], v[f"{p}/se"], v[f"{p}/t"], v[f"{p}/p"] = r.coef, r.se, r.t, r.p
        v[f"{p}/ci_lo"], v[f"{p}/ci_hi"] = r.ci_lo, r.ci_hi

    # --- log1/friend_pre: pre-trend with normal inference ------------------------
    res.objects["friend_pre"] = {}
    for tag, sub, origin in (("hk", hk[hk.post == 0], 1998), ("sh", sh[sh.post == 0], 2001)):
        sub = sub.assign(t=sub.year - origin)
        r = did.pretrend_test(sub, t_col="t")
        res.objects["friend_pre"][tag] = r
        p = f"log1/friend_pre/{tag}"
        v[f"{p}/n_obs"], v[f"{p}/n_clusters"] = r.n, r.n_clusters
        v[f"{p}/coef"], v[f"{p}/se"], v[f"{p}/z"], v[f"{p}/p"] = r.coef, r.se, r.z, r.p_z
        v[f"{p}/ci_lo"], v[f"{p}/ci_hi"] = r.ci_z

    # --- log2/xtreg_twfe: xtreg fe with cluster SE and wild bootstrap -------------
    res.objects["xtreg_twfe"], res.objects["boot_exact"], res.objects["boot_mc"] = {}, {}, {}
    for tag, sub in (("1a", hk), ("1b", core3), ("1c", sh), ("1d", sh_bal)):
        fit = did.xtreg_twfe(sub)
        res.objects["xtreg_twfe"][tag] = fit
        pfx = f"log2/xtreg_twfe/{tag}"
        _record_xtreg(res, pfx, fit)
        if tag == "1d":
            continue
        sub = sub.reset_index(drop=True)
        X = pd.concat(
            [sub[["treated_post"]].astype(float), factor_dummies(sub.year.to_numpy(), "year")], axis=1
        )
        common = dict(weights="webb", ssc_k=None)
        br = did.boottest_xtreg_fe(
            sub[Y].to_numpy(), X, sub.unit_id.to_numpy(), "treated_post",
            reps=boot_reps, seed=seed, exhaustive=exhaustive, band=(9999, 4.0),
        )
        res.objects["boot_exact"][tag] = br
        _record_boot(res, f"{pfx}/boottest", br, reps_ref=9999)
        res.objects["boot_mc"][tag] = did.boottest_xtreg_fe(
            sub[Y].to_numpy(), X, sub.unit_id.to_numpy(), "treated_post", reps=boot_reps, seed=seed, exhaustive=False
        )

    # --- log2/pretrend: pre-trend regressions with t(G-1) and wild bootstrap -------
    res.objects["pretrend"] = {}
    for tag, sub in (("hk", hk[hk.post == 0]), ("sh", sh[sh.post == 0])):
        sub = sub.assign(t=sub.rel_year).reset_index(drop=True)
        pr = did.pretrend_test(sub, t_col="t")
        res.objects["pretrend"][tag] = pr
        pfx = f"log2/pretrend/{tag}"
        _record_regress(res, pfx, pr.fit)
        br = did.boottest_regress(
            pr.y, pr.X, pr.cluster, "c.t#c.treated", reps=boot_reps, seed=seed, exhaustive=exhaustive, band=(9999, 4.0)
        )
        res.objects["boot_exact"][f"pre_{tag}"] = br
        _record_boot(res, f"{pfx}/boottest", br, reps_ref=9999)

    # --- log2/event_study: event studies --------------------------------------------
    res.objects["event_study"] = {}
    es = did.event_study(hk, lo=-6, hi=10, base=-1)
    res.objects["event_study"]["hk"] = es
    _record_xtreg(res, "log2/event_study/hk", es.fit, rename=lambda t: f"ev_s.{int(t.split('.')[1]) + 6}" if t.startswith("ev.") else t)
    es = did.event_study(sh_bal, lo=None, hi=None, base=-1)
    res.objects["event_study"]["sh"] = es
    _record_xtreg(res, "log2/event_study/sh", es.fit, rename=lambda t: f"ev_sh_s.{int(t.split('.')[1]) + 8}" if t.startswith("ev.") else t)

    # --- Synthetic control (both runs) and placebo tables -----------------------------
    res.objects["scm"] = {}
    for tag, (case, units, start, end, trp, treated) in SCM_SPEC.items():
        W = wide_block(d, case, units, start, end)
        fit = _record_scm(res, f"log1/scm/{tag}", f"log2/synth/{tag}", tag, W, treated, trp, True)
        years = res.objects[f"scm/{tag}"]["years"]
        pre = res.objects[f"scm/{tag}"]["pre"]
        cols = res.objects[f"scm/{tag}"]["units"]
        Wm = res.objects[f"scm/{tag}"]["W"]
        v[f"log1/scm/{tag}/window_observations_deleted"] = len(d) - len(d[(d.study_case == case) & d.year.between(start, end)])
        v[f"log1/scm/{tag}/mean_post_gap/value"] = fit.mean_post_gap
        v[f"log1/scm/{tag}/mean_post_gap/from"] = trp
        v[f"log1/scm/{tag}/mean_post_gap/to"] = end
        pl1 = scm.placebo_in_space(
            Wm.to_numpy(), 0, pre, labels=[UNIT_NAMES[u] for u in cols], exclude_treated=False
        )
        res.objects[f"placebo_log1/{tag}"] = pl1
        for _, r in pl1.table.iterrows():
            pfx = f"log1/scm/{tag}/placebo_meangap/{r['unit']}"
            v[f"{pfx}/mean_gap"] = r["mean_post_gap"]
            v[f"{pfx}/rank"] = int(r["rank_abs_mean_gap"])
        if tag == "hk":
            pl2 = scm.placebo_in_space(Wm.to_numpy(), 0, pre, labels=cols, exclude_treated=True)
            res.objects["placebo_log2/hk"] = pl2
            for _, r in pl2.table.iterrows():
                pfx = f"log2/placebo_ratio/{int(r['unit'])}"
                v[f"{pfx}/pre_rmspe"] = r["pre_rmspe"]
                v[f"{pfx}/post_rmspe"] = r["post_rmspe"]
                v[f"{pfx}/ratio"] = r["ratio"]
                v[f"{pfx}/mean_gap"] = r["mean_post_gap"]
                v[f"{pfx}/rank"] = int(r["rank_ratio"])

    # --- Sample facts printed by the original runs ---------------------------------------
    ev = np.where(d.treated == 1, d.rel_year, -1)
    v["log2/data_facts/ev_binned_real_changes"] = int((np.clip(ev, -6, 10) != ev).sum())
    v["log2/data_facts/ev_sh_missing_values"] = int(((d.study_case != "Shanghai") | (d.year < 2008)).sum())
    v["log2/data_facts/window_observations_deleted_hk"] = v["log1/scm/hk/window_observations_deleted"]
    v["log2/data_facts/window_observations_deleted_sh"] = v["log1/scm/sh/window_observations_deleted"]
    return res


# ----------------------------------------------------------------------------
# Equivalence check
# ----------------------------------------------------------------------------
def load_targets(path: str | Path | Mapping | None = None) -> dict:
    """Load the reference values used as equivalence targets.

    Parameters
    ----------
    path : str, Path, mapping or None
        Path of a reference file, an already loaded mapping (returned as a
        shallow copy), or None for :data:`DEFAULT_TARGETS`.

    Returns
    -------
    dict
        ``meta`` (provenance sentence), ``values`` (key -> ``{"value": number,
        "dp": decimals printed}``) and ``tables`` (event-study coefficients of
        both study cases, the Hong Kong placebo tables; lists of row
        dictionaries).
    """
    if isinstance(path, Mapping):
        return dict(path)
    return json.loads(Path(path or DEFAULT_TARGETS).read_text(encoding="utf-8"))


def target_values(targets: str | Path | Mapping | None = None) -> dict[str, float]:
    """Reference numbers as a flat mapping.

    Parameters
    ----------
    targets : str, Path, mapping or None
        Argument of :func:`load_targets`.

    Returns
    -------
    dict
        Key -> reference number.
    """
    return {key: entry["value"] for key, entry in load_targets(targets)["values"].items()}


_NUMBER_RE = re.compile(r"^[-+]?(?P<int>\d*)(?:\.(?P<frac>\d*))?(?:[eE](?P<exp>[-+]?\d+))?$")


def printed_decimals(text: str) -> int:
    """Number of decimals with which a number was printed.

    Parameters
    ----------
    text : str
        Printed number in fixed or exponent notation, for example
        ``"5.910137"``, ``".1824719"``, ``"101"`` or ``"9.09e-15"``.

    Returns
    -------
    int
        Decimals of the mantissa minus the exponent (``"9.09e-15"`` gives 17,
        ``"7.6e+14"`` gives -13); 0 for a number printed without a decimal
        point.

    Raises
    ------
    ValueError
        If ``text`` is not a printed number.
    """
    m = _NUMBER_RE.match(text.strip())
    if m is None or not (m.group("int") or m.group("frac")):
        raise ValueError(f"not a printed number: {text!r}")
    if m.group("frac") is None:
        return 0
    return len(m.group("frac")) - int(m.group("exp") or 0)


def print_halfwidth(dp: int) -> float:
    """Half a unit in the last printed digit of a number.

    Parameters
    ----------
    dp : int
        Number of decimals with which the number was printed (see
        :func:`printed_decimals`).  0 marks an integer, which carries no
        rounding error.

    Returns
    -------
    float
        ``0.5 * 10 ** (-dp)``, or 0 when ``dp`` is 0.
    """
    return 0.0 if dp == 0 else 0.5 * 10.0 ** (-dp)


_KIND_RULES = [
    ("skip", re.compile(r"/boottest/(reps|seed|ci_level)$|/F_df1$")),
    ("count", re.compile(r"/(n_obs|n_clusters|n_groups|obs_per_group_min|obs_per_group_max|F_df2|df|rank|from|to|window_observations_deleted|year_min|year_max)$|/data_facts/")),
    ("mc_p", re.compile(r"/boottest/p$")),
    ("mc_ci", re.compile(r"/boottest/ci_(lo|hi)$")),
    ("scm_rmspe", re.compile(r"/(scm|synth)/[a-z]+/rmspe$")),
    ("scm_weight", re.compile(r"/weight/")),
    ("scm_balance", re.compile(r"/balance/")),
    ("scm_gap", re.compile(r"/mean_post_gap/value$|/placebo_meangap/.*/mean_gap$|/placebo_ratio/\d+/mean_gap$")),
    ("scm_ratio", re.compile(r"/placebo_ratio/\d+/(pre_rmspe|post_rmspe|ratio)$")),
    ("panel_stat", re.compile(r"/(sigma_u|sigma_e|rho|corr_u_xb|r2_between|r2_overall)$")),
]


def classify_key(key: str) -> str:
    """Tolerance class of a result key.

    Parameters
    ----------
    key : str
        Slash-separated key of :attr:`Results.values`.

    Returns
    -------
    str
        One of the classes of :data:`TOLERANCES`, or ``"skip"`` for run
        settings (draws, seed, confidence level, numerator degrees of freedom)
        that are not compared.
    """
    for kind, rx in _KIND_RULES:
        if rx.search(key):
            return kind
    return "deterministic"


def check_equivalence(
    results: Results | Mapping,
    targets: Mapping | str | Path | None = None,
    tolerances: Mapping | None = None,
) -> pd.DataFrame:
    """Compare Python results with the reference values.

    Parameters
    ----------
    results : Results or mapping key -> float
        Output of :func:`run_all`.
    targets : mapping, path or None
        Reference values in the layout of the reference file (a ``values``
        mapping from key to ``{"value", "dp"}``) or the path of such a file;
        the default is :data:`DEFAULT_TARGETS`.
    tolerances : mapping, optional
        Overrides for entries of :data:`TOLERANCES`.

    Returns
    -------
    DataFrame
        One row per reference value with columns ``key, kind, reference,
        python, abs_diff, tol, status, note``.  ``status`` is PASS, FAIL or
        MISSING (the Python result lacks the key).  Keys classified as
        ``skip`` are omitted.
    """
    tg = load_targets(targets)
    tol = {k: dict(v) for k, v in TOLERANCES.items()}
    for k, v in (tolerances or {}).items():
        tol.setdefault(k, {}).update(v)
    values = results.values if isinstance(results, Results) else dict(results)
    meta = results.meta if isinstance(results, Results) else {}
    entries = tg["values"]
    flat = {key: entry["value"] for key, entry in entries.items()}
    rows = []
    for key, sval in flat.items():
        kind = meta.get(key, {}).get("kind") or classify_key(key)
        if kind == "skip":
            continue
        half = print_halfwidth(entries[key].get("dp", 0))
        pval = values.get(key, None)
        row = dict(key=key, kind=kind, reference=sval, python=pval, abs_diff=np.nan, tol=np.nan, status="MISSING", note="")
        if pval is None or (isinstance(pval, float) and np.isnan(pval) and kind != "deterministic"):
            rows.append(row)
            continue
        pval = float(pval)
        diff = pval - sval
        row["abs_diff"] = abs(diff)
        if kind == "count":
            row["tol"] = 0.0
            ok = pval == sval
        elif kind in ("deterministic", "panel_stat"):
            t = tol[kind]
            row["tol"] = half + t["rel"] * abs(sval) + t["floor"]
            ok = abs(diff) <= row["tol"]
            base = key.rsplit("/", 1)[0]
            noise = SE_NOISE_RELATIVE * max(1.0, abs(flat.get(base + "/coef", 0.0)))
            degenerate = "/coef/" in key and flat.get(base + "/se", 1.0) < noise
            if degenerate and key.endswith("/se"):
                # The cluster-robust variance of this coefficient is zero by construction
                # (the treated cluster has its own intercept); only noise-level values occur.
                row["tol"] = noise
                ok = abs(pval) <= noise
                row["note"] = "degenerate row: standard error is numerically zero"
            elif degenerate and key.endswith("/t"):
                ok, row["note"] = True, "degenerate row: t not compared"
            elif degenerate and key.endswith(("/ci_lo", "/ci_hi")):
                row["tol"] += 10.0 * noise
                ok = abs(diff) <= row["tol"]
                row["note"] = "degenerate row: limit widened by the noise level of the standard error"
            elif np.isnan(pval):
                ok = False
            if not ok and key.endswith("/sigma_u"):
                base = key[: -len("/sigma_u")]
                hits = [
                    n
                    for n in ("sd_groups", "sd_obs", "adj_arith", "adj_harm")
                    if abs(values.get(f"{base}/sigma_u_variant/{n}", np.inf) - sval) <= row["tol"]
                ]
                if hits:
                    row["note"] = "primary definition differs; matching alternative definition(s): " + ", ".join(hits)
        elif kind == "mc_p":
            reps = meta.get(key, {}).get("reps", 9999)
            sigma = np.sqrt(max(pval * (1 - pval), 1.0 / reps) / reps)
            row["tol"] = tol["mc_p"]["n_sigma"] * sigma + half
            ok = abs(diff) <= row["tol"]
            row["note"] = f"reference p is a Monte Carlo estimate ({reps} draws); Python p is exact"
        elif kind == "mc_ci":
            lo, hi = meta[key]["accept"]
            row["tol"] = 0.5 * (hi - lo) + half
            ok = (lo - half) <= sval <= (hi + half)
            row["note"] = f"accepted range [{lo:.4g}, {hi:.4g}] +/- print precision"
        elif kind == "scm_rmspe":
            row["tol"] = tol["scm_rmspe"]["abs"] + half
            ok = diff <= row["tol"]
            if diff < -row["tol"]:
                row["note"] = "Python optimum is lower than the reference value"
        elif kind in ("scm_weight", "scm_balance", "scm_gap"):
            row["tol"] = tol[kind]["abs"] + half
            ok = abs(diff) <= row["tol"]
        elif kind == "scm_ratio":
            row["tol"] = tol["scm_ratio"]["rel"] * abs(sval) + half
            ok = abs(diff) <= row["tol"]
        else:  # pragma: no cover
            raise ValueError(kind)
        row["status"] = "PASS" if ok else "FAIL"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_equivalence(table: pd.DataFrame) -> pd.DataFrame:
    """Count PASS, FAIL and MISSING rows by tolerance class.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`check_equivalence`.

    Returns
    -------
    DataFrame
        One row per tolerance class plus a row ``ALL``; columns ``PASS``,
        ``FAIL``, ``MISSING`` and ``total``.
    """
    s = table.pivot_table(index="kind", columns="status", values="key", aggfunc="count", fill_value=0)
    for c in ("PASS", "FAIL", "MISSING"):
        if c not in s:
            s[c] = 0
    s = s[["PASS", "FAIL", "MISSING"]]
    s["total"] = s.sum(axis=1)
    s.loc["ALL"] = s.sum(axis=0)
    return s.astype(int)


_HEADLINE = re.compile(
    r"log1/friend_did/\w+/(?:coef|se|t|p|df)$"
    r"|log1/friend_pre/\w+/(?:coef|se|z|p)$"
    r"|log2/xtreg_twfe/\w+/coef/treated_post/(?:coef|se|p)$"
    r"|log2/(?:xtreg_twfe|pretrend)/\w+/boottest/(?:t|p|ci_lo|ci_hi)$"
    r"|log2/pretrend/\w+/coef/c\.t#c\.treated/(?:coef|se|p)$"
    r"|/rmspe$|/weight/|log2/placebo_ratio/\d+/(?:ratio|rank)$|/placebo_meangap/.*/rank$"
)


def format_check_table(table: pd.DataFrame, headline_only: bool = True, float_fmt: str = "{:.7g}") -> str:
    """Format the equivalence table as printable text.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`check_equivalence`.
    headline_only : bool
        True shows the headline quantities and every non-PASS row; False shows
        every row.
    float_fmt : str
        Format string of the numeric columns.

    Returns
    -------
    str
        The summary by tolerance class followed by the selected rows.
    """
    summary = summarize_equivalence(table)
    show = table[table.key.str.contains(_HEADLINE)] if headline_only else table
    bad = table[table.status != "PASS"]
    out = ["Equivalence summary by tolerance class", summary.to_string(), ""]
    shown = pd.concat([show, bad]).drop_duplicates("key")
    cols = ["key", "kind", "reference", "python", "abs_diff", "tol", "status"]
    f = shown[cols].copy()
    for c in ("reference", "python", "abs_diff", "tol"):
        f[c] = f[c].map(lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else float_fmt.format(x))
    out.append(f"Rows shown: {len(f)} of {len(table)} (headline quantities and every non-PASS row)")
    out.append(f.to_string(index=False))
    return "\n".join(out)
