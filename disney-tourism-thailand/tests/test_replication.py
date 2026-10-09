"""Tests for dtt.replication: panel loading, the analysis driver, the equivalence check and the reference values.

The analysis is run on a SIMULATED panel.  The equivalence check is exercised
with pseudo-targets built from the Python results themselves (every row must
pass) and with perturbed copies (the perturbed rows must fail), and against the
reference values for the quantities that depend only on the sample structure.
The reference file is checked against numbers read by hand from the original
output and against the arithmetic relations that tie its entries together.  The
test that needs the real panel runs only when DTT_RUN_REAL=1 is set and the
panel file exists.
"""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dtt import replication as rp
from dtt import simulate

PROVENANCE = "Numbers printed by an independent run of the original analysis, used as equivalence targets."


@pytest.fixture(scope="module")
def sim():
    df, truth = simulate.simulate_panel(seed=11, trend_spread=0.0)
    return df


@pytest.fixture(scope="module")
def results(sim):
    return rp.run_all(sim)


def printed(v):
    """Print a number with seven significant digits."""
    if float(v).is_integer() and abs(v) < 1e9:
        return str(int(v))
    return format(float(v), ".7g")


def self_targets(results, reference):
    """Pseudo-targets: the Python results rounded to print precision, under the keys of the reference file."""
    values = {}
    for key in reference["values"]:
        v = results.values.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            text = printed(v)
            values[key] = {"value": float(text), "dp": rp.printed_decimals(text)}
    return {"values": values}


def with_entry(targets, key, value, dp=None):
    """Copy of pseudo-targets with the value of one key replaced (printed decimals kept unless given)."""
    values = dict(targets["values"])
    values[key] = {"value": value, "dp": values[key]["dp"] if dp is None else dp}
    return {"values": values}


def value(reference, key):
    return reference["values"][key]["value"]


def coef_rows(reference, section, key):
    """Coefficient table of one regression of the second run: term -> {field: number}."""
    prefix = f"log2/{section}/{key}/coef/"
    rows = {}
    for k, entry in reference["values"].items():
        if k.startswith(prefix):
            term, name = k[len(prefix):].rsplit("/", 1)
            rows.setdefault(term, {})[name] = entry["value"]
    return rows


REGRESSIONS = [("xtreg_twfe", tag) for tag in ("1a", "1b", "1c", "1d")] + [("pretrend", "hk"), ("pretrend", "sh"), ("event_study", "hk"), ("event_study", "sh")]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,dp,expected",
    [
        ("5.910137", 6, 5e-7),
        (".1824719", 7, 5e-8),
        ("0.0147", 4, 5e-5),
        ("101", 0, 0.0),
        ("0", 0, 0.0),
        ("-3.080338", 6, 5e-7),
        ("1.5e-15", 16, 5e-17),
        ("9.09e-15", 17, 5e-18),
        ("7.6e+14", -13, 5e12),
        ("32.5417", 4, 5e-5),
        ("21.5", 1, 0.05),
    ],
)
def test_printed_decimals_and_print_halfwidth(text, dp, expected):
    assert rp.printed_decimals(text) == dp
    assert rp.print_halfwidth(dp) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("text", ["abc", "", ".", "1.2.3", "e5", "--1"])
def test_printed_decimals_rejects_text_that_is_not_a_number(text):
    with pytest.raises(ValueError, match="not a printed number"):
        rp.printed_decimals(text)


@pytest.mark.parametrize(
    "key,kind",
    [
        ("log1/friend_did/1a/coef", "deterministic"),
        ("log1/friend_did/1a/df", "count"),
        ("log1/friend_did/1a/n_obs", "count"),
        ("log2/xtreg_twfe/1a/boottest/p", "mc_p"),
        ("log2/xtreg_twfe/1a/boottest/ci_lo", "mc_ci"),
        ("log2/xtreg_twfe/1a/boottest/reps", "skip"),
        ("log2/xtreg_twfe/1a/boottest/seed", "skip"),
        ("log2/xtreg_twfe/1a/F_df1", "skip"),
        ("log2/xtreg_twfe/1a/sigma_u", "panel_stat"),
        ("log2/xtreg_twfe/1a/r2_within", "deterministic"),
        ("log2/xtreg_twfe/1a/r2_between", "panel_stat"),
        ("log1/scm/hk/rmspe", "scm_rmspe"),
        ("log2/synth/sh/rmspe", "scm_rmspe"),
        ("log1/scm/hk/weight/Singapore", "scm_weight"),
        ("log1/scm/hk/balance/receipts_pct_gdp(2004)/treated", "scm_balance"),
        ("log1/scm/hk/mean_post_gap/value", "scm_gap"),
        ("log1/scm/hk/placebo_meangap/Singapore/mean_gap", "scm_gap"),
        ("log1/scm/hk/placebo_meangap/Singapore/rank", "count"),
        ("log2/placebo_ratio/3/ratio", "scm_ratio"),
        ("log2/placebo_ratio/3/rank", "count"),
        ("log2/data_facts/ev_binned_real_changes", "count"),
    ],
)
def test_classify_key(key, kind):
    assert rp.classify_key(key) == kind


def test_every_target_key_is_classified_and_tolerances_are_explicit(reference):
    kinds = {rp.classify_key(k) for k in reference["values"]}
    assert kinds <= set(rp.TOLERANCES) | {"skip"}
    for name, spec in rp.TOLERANCES.items():
        assert spec, name


# ----------------------------------------------------------------------------
# The reference file
# ----------------------------------------------------------------------------
def _string_leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _string_leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _string_leaves(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def test_reference_file_holds_numbers_only(reference):
    assert set(reference) == {"meta", "values", "tables"}
    assert reference["meta"] == {"provenance": PROVENANCE}
    assert list(_string_leaves(reference)) == [("/meta/provenance", PROVENANCE)]
    assert len(reference["values"]) == 1216
    for key, entry in reference["values"].items():
        assert set(entry) == {"value", "dp"}, key
        assert _is_number(entry["value"]) and np.isfinite(entry["value"]), key
        assert isinstance(entry["dp"], int) and not isinstance(entry["dp"], bool), key
        assert entry["dp"] != 0 or float(entry["value"]).is_integer() or rp.classify_key(key) == "scm_weight", key
    text = rp.DEFAULT_TARGETS.read_text(encoding="utf-8")
    assert "\u2014" not in text and "stata" not in text.lower()


def test_reference_file_has_no_run_settings(reference):
    assert all(rp.classify_key(k) != "skip" for k in reference["values"])


def test_reference_decimals_describe_the_printed_precision(reference):
    spot = {
        "log1/friend_did/1a/coef": 6,
        "log1/friend_did/1a/t": 2,
        "log1/friend_did/1a/p": 4,
        "log1/friend_did/1a/n_obs": 0,
        "log1/scm/hk/weight/Ireland": 0,
        "log1/scm/hk/weight/New Zealand": 1,
        "log1/scm/sh/weight/Beijing": 3,
        "log2/xtreg_twfe/1a/coef/treated_post/se": 7,
        "log2/xtreg_twfe/1a/boottest/ci_lo": 3,
        "log2/pretrend/hk/coef/_cons/se": 17,
        "log2/pretrend/hk/coef/_cons/t": -13,
    }
    for key, dp in spot.items():
        assert reference["values"][key]["dp"] == dp, key


def test_reference_tables_equal_the_values(reference):
    tables = reference["tables"]
    fields = ("coef", "se", "t", "p", "ci_lo", "ci_hi")
    for case, offset, prefix in (("hk", 6, "ev_s"), ("sh", 8, "ev_sh_s")):
        rows = tables["event_study"][case]
        assert [r["rel"] for r in rows] == sorted(r["rel"] for r in rows)
        assert all(_is_number(r[f]) for r in rows for f in fields)
        for r in rows:
            assert r["rel"] != -1
            for f in fields:
                assert r[f] == value(reference, f"log2/event_study/{case}/coef/{prefix}.{r['rel'] + offset}/{f}")
        n_terms = sum(1 for t in coef_rows(reference, "event_study", case) if t.startswith(prefix + "."))
        assert len(rows) == n_terms
    assert [r["rel"] for r in tables["event_study"]["hk"]] == [r for r in range(-6, 11) if r != -1]
    assert [r["rel"] for r in tables["event_study"]["sh"]] == [r for r in range(-8, 4) if r != -1]
    for r in tables["placebo_ratio"]["hk"]:
        for f in ("pre_rmspe", "post_rmspe", "ratio", "mean_gap", "rank"):
            assert r[f] == value(reference, f"log2/placebo_ratio/{r['unit_id']}/{f}")
    for r in tables["placebo_mean_gap"]["hk"]:
        name = simulate.UNIT_NAMES[r["unit_id"]]
        for f in ("mean_gap", "rank"):
            assert r[f] == value(reference, f"log1/scm/hk/placebo_meangap/{name}/{f}")
    assert len(tables["placebo_ratio"]["hk"]) == len(tables["placebo_mean_gap"]["hk"]) == 6


def test_load_targets_accepts_a_path_a_mapping_and_the_default(reference, tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    assert rp.load_targets(path) == reference
    assert rp.load_targets(str(path)) == reference
    assert rp.load_targets(reference) == reference
    assert rp.load_targets(None) == reference
    numbers = rp.target_values(reference)
    assert len(numbers) == 1216 and numbers["log1/friend_did/1a/n_obs"] == 129
    assert rp.target_values(path) == numbers


SPOT = [
    # friend_did (report-formula SE)
    ("log1/friend_did/1a/coef", 5.910137),
    ("log1/friend_did/1a/se", 0.171312),
    ("log1/friend_did/1a/df", 101),
    ("log1/friend_did/1a/t", 34.50),
    ("log1/friend_did/1a/n_obs", 129),
    ("log1/friend_did/1a/n_clusters", 6),
    ("log1/friend_did/1a/ci_lo", 5.570300),
    ("log1/friend_did/1a/ci_hi", 6.249974),
    ("log1/friend_did/1b/coef", 5.869956),
    ("log1/friend_did/1b/se", 0.214694),
    ("log1/friend_did/1b/n_obs", 86),
    ("log1/friend_did/1b/df", 60),
    ("log1/friend_did/1c/coef", 0.925246),
    ("log1/friend_did/1c/se", 0.276902),
    ("log1/friend_did/1c/p", 0.0025),
    ("log1/friend_did/1c/df", 26),
    # friend_pre
    ("log1/friend_pre/hk/coef", 0.472077),
    ("log1/friend_pre/hk/z", 4.73),
    ("log1/friend_pre/hk/n_obs", 41),
    ("log1/friend_pre/sh/coef", 0.231015),
    ("log1/friend_pre/sh/se", 0.066807),
    ("log1/friend_pre/sh/p", 0.0005),
    # synthetic control
    ("log1/scm/hk/rmspe", 1.086396),
    ("log1/scm/hk/weight/New Zealand", 0.9),
    ("log1/scm/hk/weight/Singapore", 0.1),
    ("log1/scm/hk/weight/Ireland", 0.0),
    ("log1/scm/hk/balance/receipts_pct_gdp(2004)/treated", 7.02),
    ("log1/scm/hk/balance/receipts_pct_gdp(2004)/synthetic", 4.8794),
    ("log1/scm/hk/mean_post_gap/value", 6.612),
    ("log1/scm/sh/rmspe", 0.1639926),
    ("log1/scm/sh/weight/Beijing", 0.583),
    ("log1/scm/sh/weight/Shenzhen", 0.417),
    ("log1/scm/sh/balance/receipts_pct_gdp(2008)/synthetic", 2.726735),
    ("log1/scm/sh/mean_post_gap/value", 0.222),
    ("log1/scm/hk/placebo_meangap/Hong Kong/mean_gap", 6.61205),
    ("log1/scm/hk/placebo_meangap/Hong Kong/rank", 1),
    ("log1/scm/hk/placebo_meangap/New Zealand/mean_gap", -3.080338),
    ("log1/scm/hk/placebo_meangap/Ireland/rank", 6),
    ("log1/scm/sh/placebo_meangap/Beijing/mean_gap", -0.324608),
    ("log1/scm/sh/placebo_meangap/Shanghai/rank", 2),
    # xtreg fe tables
    ("log2/xtreg_twfe/1a/coef/treated_post/coef", 5.910137),
    ("log2/xtreg_twfe/1a/coef/treated_post/se", 0.1824719),
    ("log2/xtreg_twfe/1a/coef/treated_post/t", 32.39),
    ("log2/xtreg_twfe/1a/coef/treated_post/ci_lo", 5.441078),
    ("log2/xtreg_twfe/1a/coef/year.2004/coef", 0.2703813),
    ("log2/xtreg_twfe/1a/coef/_cons/coef", 3.617916),
    ("log2/xtreg_twfe/1a/r2_within", 0.6666),
    ("log2/xtreg_twfe/1a/r2_between", 0.7374),
    ("log2/xtreg_twfe/1a/sigma_u", 1.6173527),
    ("log2/xtreg_twfe/1a/sigma_e", 1.0093722),
    ("log2/xtreg_twfe/1a/rho", 0.71968996),
    ("log2/xtreg_twfe/1a/n_clusters", 6),
    ("log2/xtreg_twfe/1b/coef/treated_post/coef", 5.869956),
    ("log2/xtreg_twfe/1b/coef/treated_post/se", 0.2405224),
    ("log2/xtreg_twfe/1c/coef/treated_post/coef", 0.9252456),
    ("log2/xtreg_twfe/1c/coef/treated_post/se", 0.3233765),
    ("log2/xtreg_twfe/1c/coef/treated_post/p", 0.104),
    ("log2/xtreg_twfe/1d/coef/treated_post/coef", 0.302625),
    ("log2/xtreg_twfe/1d/coef/treated_post/se", 0.0942791),
    ("log2/xtreg_twfe/1d/n_obs", 36),
    ("log2/xtreg_twfe/1d/r2_within", 0.9129),
    # boottest
    ("log2/xtreg_twfe/1a/boottest/t", 32.5417),
    ("log2/xtreg_twfe/1a/boottest/p", 0.0147),
    ("log2/xtreg_twfe/1a/boottest/ci_lo", 3.395),
    ("log2/xtreg_twfe/1a/boottest/ci_hi", 8.518),
    ("log2/xtreg_twfe/1b/boottest/p", 0.0688),
    ("log2/xtreg_twfe/1b/boottest/ci_lo", -2.918),
    ("log2/xtreg_twfe/1b/boottest/ci_hi", 13.38),
    ("log2/xtreg_twfe/1c/boottest/ci_lo", -25.71),
    ("log2/xtreg_twfe/1c/boottest/ci_hi", 20.99),
    ("log2/xtreg_twfe/1c/boottest/p", 0.2256),
    # pre-trend regressions
    ("log2/pretrend/hk/coef/c.t#c.treated/coef", 0.4720772),
    ("log2/pretrend/hk/coef/c.t#c.treated/se", 0.0997015),
    ("log2/pretrend/hk/coef/c.t#c.treated/p", 0.005),
    ("log2/pretrend/hk/coef/unit_id.8/coef", -5.759595),
    ("log2/pretrend/hk/coef/_cons/se", 9.09e-15),
    ("log2/pretrend/hk/r2", 0.9256),
    ("log2/pretrend/hk/root_mse", 0.43729),
    ("log2/pretrend/hk/boottest/p", 0.2978),
    ("log2/pretrend/hk/boottest/ci_lo", -1.07),
    ("log2/pretrend/hk/boottest/ci_hi", 1.759),
    ("log2/pretrend/sh/coef/c.t#c.treated/coef", 0.2310153),
    ("log2/pretrend/sh/coef/c.t#c.treated/p", 0.074),
    ("log2/pretrend/sh/boottest/t", 3.4579),
    ("log2/pretrend/sh/boottest/p", 0.1267),
    ("log2/pretrend/sh/boottest/ci_lo", -2.671),
    # event studies
    ("log2/event_study/hk/coef/ev_s.0/coef", -2.899406),
    ("log2/event_study/hk/coef/ev_s.15/coef", 8.3942),
    ("log2/event_study/hk/coef/ev_s.16/coef", 4.049079),
    ("log2/event_study/hk/coef/ev_s.16/se", 0.1964296),
    ("log2/event_study/hk/coef/year.2019/coef", -0.5045849),
    ("log2/event_study/hk/r2_within", 0.9154),
    ("log2/event_study/sh/coef/ev_sh_s.11/coef", 0.3875),
    ("log2/event_study/sh/coef/ev_sh_s.11/se", 0.1769141),
    ("log2/event_study/sh/coef/ev_sh_s.5/p", 0.14),
    ("log2/event_study/sh/r2_within", 0.9282),
    # synth and placebo ratio table
    ("log2/synth/hk/rmspe", 1.086396),
    ("log2/synth/hk/weight/4", 0.9),
    ("log2/synth/hk/weight/7", 0.1),
    ("log2/synth/sh/rmspe", 0.1639926),
    ("log2/synth/sh/weight/1", 0.583),
    ("log2/synth/sh/weight/6", 0.417),
    ("log2/placebo_ratio/3/ratio", 15.7467),
    ("log2/placebo_ratio/3/rank", 1),
    ("log2/placebo_ratio/2/pre_rmspe", 1.086392),
    ("log2/placebo_ratio/2/post_rmspe", 7.25291),
    ("log2/placebo_ratio/2/ratio", 6.676143),
    ("log2/placebo_ratio/2/rank", 3),
    ("log2/placebo_ratio/7/rank", 4),
    ("log2/placebo_ratio/8/ratio", 0.9978765),
    # sample facts
    ("log2/data_facts/ev_binned_real_changes", 14),
    ("log2/data_facts/ev_sh_missing_values", 141),
    ("log2/data_facts/window_observations_deleted_hk", 57),
]


@pytest.mark.parametrize("key,expected", SPOT)
def test_reference_spot_values(reference, key, expected):
    assert key in reference["values"], f"missing key {key}"
    assert value(reference, key) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_at_least_a_hundred_spot_values():
    assert len(SPOT) >= 100


def test_t_equals_coef_over_se_for_every_coefficient_row(reference):
    n = 0
    for section, key in REGRESSIONS:
        for term, r in coef_rows(reference, section, key).items():
            if r["se"] < 1e-10:
                continue
            assert abs(r["t"] - r["coef"] / r["se"]) <= 0.005 + 1e-9, (section, key, term)
            n += 1
    assert n > 150


def test_confidence_limits_use_t_with_g_minus_1_df(reference):
    """Half-width of every printed interval equals t_{0.975, G-1} times the SE."""
    n = 0
    for section, key in REGRESSIONS:
        crit = stats.t.isf(0.025, value(reference, f"log2/{section}/{key}/n_clusters") - 1)
        for term, r in coef_rows(reference, section, key).items():
            if r["se"] < 1e-10:
                continue
            half = 0.5 * (r["ci_hi"] - r["ci_lo"])
            mid = 0.5 * (r["ci_hi"] + r["ci_lo"])
            # Fewer decimals are printed for larger magnitudes (7 significant digits),
            # so both bounds scale with the printed value rather than being absolute.
            assert half == pytest.approx(crit * r["se"], rel=2e-6, abs=2e-6), (section, key, term)
            assert mid == pytest.approx(r["coef"], rel=2e-6, abs=1e-5), (section, key, term)
            n += 1
    assert n > 150


def test_p_values_use_t_with_g_minus_1_df(reference):
    for section, key in REGRESSIONS:
        df = value(reference, f"log2/{section}/{key}/n_clusters") - 1
        for term, r in coef_rows(reference, section, key).items():
            if r["se"] < 1e-10:
                continue
            p = 2 * stats.t.sf(abs(r["coef"] / r["se"]), df)
            # the printed p has three decimals and t is recomputed from unrounded inputs
            assert abs(p - r["p"]) <= 0.0005 + 2e-4, (section, key, term, p, r["p"])


def test_rho_follows_from_sigma_u_and_sigma_e(reference):
    for section, key in REGRESSIONS:
        if section == "pretrend":
            continue
        base = f"log2/{section}/{key}"
        su, se = value(reference, f"{base}/sigma_u"), value(reference, f"{base}/sigma_e")
        assert value(reference, f"{base}/rho") == pytest.approx(su**2 / (su**2 + se**2), abs=2e-8), (section, key)


def test_friend_did_arithmetic(reference):
    for tag in ("1a", "1b", "1c"):
        v = {f: value(reference, f"log1/friend_did/{tag}/{f}") for f in ("coef", "se", "t", "p", "df", "ci_lo", "ci_hi")}
        assert v["t"] == pytest.approx(v["coef"] / v["se"], abs=0.005)
        assert v["p"] == pytest.approx(2 * stats.t.sf(abs(v["coef"] / v["se"]), v["df"]), abs=6e-5)
        crit = stats.t.isf(0.025, v["df"])
        assert v["ci_lo"] == pytest.approx(v["coef"] - crit * v["se"], abs=2e-6)
        assert v["ci_hi"] == pytest.approx(v["coef"] + crit * v["se"], abs=2e-6)


def test_friend_pre_uses_normal_inference(reference):
    crit = stats.norm.isf(0.025)
    for case in ("hk", "sh"):
        v = {f: value(reference, f"log1/friend_pre/{case}/{f}") for f in ("coef", "se", "z", "p", "ci_lo", "ci_hi")}
        assert v["z"] == pytest.approx(v["coef"] / v["se"], abs=0.005)
        assert v["p"] == pytest.approx(2 * stats.norm.sf(abs(v["coef"] / v["se"])), abs=6e-5)
        assert v["ci_lo"] == pytest.approx(v["coef"] - crit * v["se"], abs=2e-6)


def test_pretrend_estimates_agree_between_the_two_runs(reference):
    """The first run (normal inference) and the second (t inference) estimate the same interaction coefficient."""
    for case in ("hk", "sh"):
        row = coef_rows(reference, "pretrend", case)["c.t#c.treated"]
        assert value(reference, f"log1/friend_pre/{case}/coef") == pytest.approx(row["coef"], abs=2e-6)
        assert value(reference, f"log1/friend_pre/{case}/se") == pytest.approx(row["se"], abs=2e-6)


def test_did_estimates_agree_between_the_two_runs(reference):
    for tag in ("1a", "1b", "1c"):
        row = coef_rows(reference, "xtreg_twfe", tag)["treated_post"]
        assert value(reference, f"log1/friend_did/{tag}/coef") == pytest.approx(row["coef"], abs=2e-6)


def test_synth_balance_reproduces_the_pre_rmspe_of_the_placebo_table(reference):
    """RMSPE computed from the printed Hong Kong balance table agrees with the placebo-loop value."""
    prefix = "log1/scm/hk/balance/"
    names = [k[len(prefix):-len("/treated")] for k in reference["values"] if k.startswith(prefix) and k.endswith("/treated")]
    assert len(names) == 6
    gaps = [value(reference, f"{prefix}{n}/treated") - value(reference, f"{prefix}{n}/synthetic") for n in names]
    assert np.sqrt(np.mean(np.square(gaps))) == pytest.approx(value(reference, "log2/placebo_ratio/2/pre_rmspe"), abs=1e-4)


def test_placebo_ratio_column_is_post_over_pre(reference):
    for r in reference["tables"]["placebo_ratio"]["hk"]:
        assert r["ratio"] == pytest.approx(r["post_rmspe"] / r["pre_rmspe"], rel=5e-6)


def test_placebo_ranks_are_sorted_by_ratio_descending(reference):
    rows = reference["tables"]["placebo_ratio"]["hk"]
    ratios = [r["ratio"] for r in rows]
    assert ratios == sorted(ratios, reverse=True)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


def test_hong_kong_placebo_rank_as_printed(reference):
    """Ratio table: Hong Kong ranks 3 of 6; mean-gap table: Hong Kong ranks 1."""
    ratio_rank = {r["unit_id"]: r["rank"] for r in reference["tables"]["placebo_ratio"]["hk"]}
    gap_rank = {r["unit_id"]: r["rank"] for r in reference["tables"]["placebo_mean_gap"]["hk"]}
    assert ratio_rank[2] == 3 and ratio_rank[7] == 4 and gap_rank[2] == 1
    assert sorted(ratio_rank) == sorted(gap_rank) == [2, 3, 4, 7, 8, 9]


# ----------------------------------------------------------------------------
# Panel loading and validation
# ----------------------------------------------------------------------------
def real_layout(df):
    """Simulated panel in the column order of the real file, with its two extra columns."""
    out = df.copy()
    out["unit"] = out.unit_id.map(simulate.UNIT_NAMES)
    out["treatment_year"] = out.study_case.map(rp.TREATMENT_YEAR)
    cols = ["study_case", "unit", "year", "treated", "post", "treated_post", "rel_year", "treatment_year", "receipts_pct_gdp", "unit_id"]
    return out[cols]


def test_validate_panel_accepts_the_simulated_panel_and_flags_damage(sim):
    assert rp.validate_panel(sim) == []
    assert any("177" in m for m in rp.validate_panel(sim.iloc[:-1]))
    bad = sim.copy()
    bad.loc[bad.index[0], "treated_post"] = 1
    assert any("treated_post" in m for m in rp.validate_panel(bad))
    bad = sim.copy()
    bad.loc[bad.index[3], "receipts_pct_gdp"] = np.nan
    assert any("missing" in m for m in rp.validate_panel(bad))
    bad = sim.copy()
    bad["rel_year"] = bad["rel_year"] + 1
    assert any("rel_year" in m for m in rp.validate_panel(bad))


def damaged(df, kind):
    """Copy of a panel with one structural departure of the named kind."""
    out = df.copy()
    if kind == "duplicate":
        return pd.concat([out, out.iloc[[0]]], ignore_index=True)
    if kind == "unit_ids":
        out.loc[out.unit_id == 9, "unit_id"] = 10
    elif kind == "hong_kong_block":
        out.loc[out.unit_id == 9, "study_case"] = "Shanghai"
    elif kind == "post":
        out.loc[out.index[(out.study_case == "Hong Kong") & (out.year == 2004)][0], "post"] = 1
    elif kind == "treated_units":
        out.loc[out.unit_id == 3, "treated"] = 1
    elif kind == "shanghai_window":
        out = out.drop(index=out.index[(out.study_case == "Shanghai") & (out.year == 2010)][0])
    return out


@pytest.mark.parametrize(
    "kind,fragment",
    [
        ("duplicate", "duplicate unit-year rows"),
        ("unit_ids", "unit_id values are not 1..9"),
        ("hong_kong_block", "Hong Kong block should have units 2,3,4,7,8,9 and 129 rows"),
        ("post", "post != 1[year >= 2005] in the Hong Kong block"),
        ("treated_units", "treated units should be 2 and 5"),
        ("shanghai_window", "Shanghai window 2008-2019 should have 36 rows"),
    ],
)
def test_validate_panel_names_each_departure_from_the_structure(sim, kind, fragment):
    messages = rp.validate_panel(damaged(sim, kind))
    assert any(fragment in m for m in messages), messages


def test_wide_block_returns_years_by_units_and_rejects_unbalanced_windows(sim):
    case, units, start, end, _, _ = rp.SCM_SPEC["hk"]
    wide = rp.wide_block(sim, case, units, start, end)
    assert list(wide.columns) == units and list(wide.index) == list(range(start, end + 1))
    expected = sim[(sim.unit_id == 7) & (sim.year == 2004)].receipts_pct_gdp.iloc[0]
    assert wide.loc[2004, 7] == expected
    with pytest.raises(ValueError, match="not balanced"):
        rp.wide_block(sim, case, units, 1998, 2019)


def test_load_panel_round_trip_through_a_csv_file(sim, tmp_path):
    expected = sim.sort_values(["unit_id", "year"]).reset_index(drop=True)
    path = tmp_path / "panel.csv"
    sim.to_csv(path, index=False)
    back = rp.load_panel(path)
    assert list(back.columns) == rp.COLUMNS
    pd.testing.assert_frame_equal(back, expected, check_dtype=False, check_exact=True)
    assert back.unit_id.dtype.kind == "i" and back.receipts_pct_gdp.dtype.kind == "f"
    assert rp.validate_panel(back) == []


def test_load_panel_reads_the_layout_of_the_real_file(sim, tmp_path):
    """Extra columns, another column order, unsorted rows and a byte-order mark are handled."""
    expected = sim.sort_values(["unit_id", "year"]).reset_index(drop=True)
    shuffled = real_layout(sim).sample(frac=1.0, random_state=0)
    path = tmp_path / "layout.csv"
    shuffled.to_csv(path, index=False, encoding="utf-8-sig")
    back = rp.load_panel(path)
    assert list(back.columns) == rp.COLUMNS
    pd.testing.assert_frame_equal(back, expected, check_dtype=False, check_exact=True)


def test_load_panel_rejects_missing_columns_and_wrong_structure(sim, tmp_path):
    p1 = tmp_path / "a.csv"
    sim.drop(columns=["rel_year"]).to_csv(p1, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        rp.load_panel(p1)
    p2 = tmp_path / "b.csv"
    sim.iloc[:-5].to_csv(p2, index=False)
    with pytest.raises(ValueError, match="structure"):
        rp.load_panel(p2)
    assert len(rp.load_panel(p2, validate=False)) == len(sim) - 5


# ----------------------------------------------------------------------------
# run_all
# ----------------------------------------------------------------------------
def test_run_all_covers_every_non_skipped_target_key(results, reference):
    keys = [k for k in reference["values"] if rp.classify_key(k) != "skip"]
    missing = [k for k in keys if k not in results.values]
    assert missing == []
    assert len(keys) == 1216


def test_run_all_reproduces_the_sample_structure_of_the_reference_values(results, reference):
    """Sample sizes, cluster counts and degrees of freedom depend only on the panel structure."""
    fl = rp.target_values(reference)
    v = results.values
    structural = [k for k in fl if rp.classify_key(k) == "count" and not k.endswith("/rank")]
    assert len(structural) >= 70
    bad = [(k, fl[k], v[k]) for k in structural if v[k] != fl[k]]
    assert bad == []


def test_run_all_recovers_the_known_effects(results):
    v = results.values
    assert abs(v["log1/friend_did/1a/coef"] - 5.0) < 0.5
    assert abs(v["log1/friend_did/1c/coef"] - 1.0) < 0.5
    assert abs(v["log1/scm/hk/mean_post_gap/value"] - 5.0) < 0.5
    assert abs(v["log1/scm/sh/mean_post_gap/value"] - 1.0) < 0.5


def test_run_all_internal_consistency(results):
    v = results.values
    # xtreg and friend_did estimate the same coefficient with the relation between their standard errors
    for tag in ("1a", "1b", "1c"):
        n, g, df = v[f"log1/friend_did/{tag}/n_obs"], v[f"log1/friend_did/{tag}/n_clusters"], v[f"log1/friend_did/{tag}/df"]
        k = n - df
        ratio = np.sqrt(g / (g - 1) * (n - 1) / (n - k + g - 1) * (n - k) / n)
        assert v[f"log2/xtreg_twfe/{tag}/coef/treated_post/se"] == pytest.approx(v[f"log1/friend_did/{tag}/se"] * ratio, rel=1e-9)
        assert v[f"log2/xtreg_twfe/{tag}/coef/treated_post/coef"] == pytest.approx(v[f"log1/friend_did/{tag}/coef"], rel=1e-9)
    # the same synthetic control appears under both prefixes
    assert v["log1/scm/hk/rmspe"] == v["log2/synth/hk/rmspe"]
    # bootstrap p-values lie in [0, 1] and confidence sets contain the estimate
    for tag in ("1a", "1b", "1c"):
        p = f"log2/xtreg_twfe/{tag}/boottest"
        assert 0 <= v[f"{p}/p"] <= 1
        assert v[f"{p}/ci_lo"] < v[f"log2/xtreg_twfe/{tag}/coef/treated_post/coef"] < v[f"{p}/ci_hi"]
    # the exhaustive and the Monte Carlo bootstrap agree within Monte Carlo error
    for tag, br in results.objects["boot_mc"].items():
        ex = results.objects["boot_exact"][tag]
        se = np.sqrt(max(ex.p_value * (1 - ex.p_value), 1e-4) / 9999)
        assert abs(br.p_value - ex.p_value) < 5 * se


def test_run_all_is_deterministic(sim, results):
    again = rp.run_all(sim)
    for k, a in results.values.items():
        b = again.values[k]
        assert (a == b) or (np.isnan(a) and np.isnan(b)), k


# ----------------------------------------------------------------------------
# check_equivalence
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pseudo(results, reference):
    return self_targets(results, reference)


def test_self_consistency_every_row_passes(results, pseudo):
    table = rp.check_equivalence(results, pseudo)
    assert len(table) > 1100
    bad = table[table.status != "PASS"]
    assert bad.empty, bad[["key", "kind", "reference", "python", "tol"]].head(20).to_string()
    assert set(table.columns) == {"key", "kind", "reference", "python", "abs_diff", "tol", "status", "note"}


@pytest.mark.parametrize(
    "key",
    [
        "log1/friend_did/1a/coef",
        "log1/friend_did/1a/se",
        "log2/xtreg_twfe/1c/coef/treated_post/ci_hi",
        "log2/xtreg_twfe/1a/sigma_u",
        "log2/xtreg_twfe/1a/r2_within",
        "log1/scm/sh/weight/Beijing",
        "log1/scm/hk/balance/receipts_pct_gdp(2004)/synthetic",
        "log1/scm/hk/mean_post_gap/value",
        "log2/placebo_ratio/3/ratio",
    ],
)
def test_small_perturbations_make_the_row_fail(results, pseudo, key):
    vals = dict(results.values)
    vals[key] = vals[key] * (1 + 3e-3) + 3e-3
    table = rp.check_equivalence(rp.Results(values=vals, meta=results.meta), pseudo).set_index("key")
    assert table.loc[key, "status"] == "FAIL"
    assert (table.drop(index=key).status == "PASS").all()


def test_count_class_is_exact_and_missing_keys_are_reported(results, pseudo):
    vals = dict(results.values)
    vals["log1/friend_did/1a/n_obs"] += 1
    del vals["log1/friend_did/1b/se"]
    table = rp.check_equivalence(rp.Results(values=vals, meta=results.meta), pseudo).set_index("key")
    assert table.loc["log1/friend_did/1a/n_obs", "status"] == "FAIL"
    assert table.loc["log1/friend_did/1b/se", "status"] == "MISSING"
    assert (table.status == "FAIL").sum() == 1 and (table.status == "MISSING").sum() == 1


def test_monte_carlo_classes_use_binomial_and_band_tolerances(results, pseudo):
    key_p = "log2/xtreg_twfe/1a/boottest/p"
    exact = results.values[key_p]
    sigma = np.sqrt(max(exact * (1 - exact), 1 / 9999) / 9999)
    for shift, expected in ((3.0 * sigma, "PASS"), (6.0 * sigma, "FAIL")):
        tab = rp.check_equivalence(results, with_entry(pseudo, key_p, exact + shift)).set_index("key")
        assert tab.loc[key_p, "status"] == expected
    key_ci = "log2/xtreg_twfe/1a/boottest/ci_lo"
    (lo, hi) = results.meta[key_ci]["accept"]
    assert lo <= results.values[key_ci] <= hi
    for target, expected in ((0.5 * (lo + results.values[key_ci]), "PASS"), (lo - 0.5 * (hi - lo) - 1e-3, "FAIL")):
        tab = rp.check_equivalence(results, with_entry(pseudo, key_ci, target)).set_index("key")
        assert tab.loc[key_ci, "status"] == expected


def test_rmspe_comparison_is_one_sided(results, pseudo):
    key = "log1/scm/hk/rmspe"
    for delta, expected in ((+2e-3, "PASS"), (-2e-3, "FAIL")):
        # the reference value is larger than the Python value by delta: Python found a lower optimum
        tab = rp.check_equivalence(results, with_entry(pseudo, key, results.values[key] + delta)).set_index("key")
        assert tab.loc[key, "status"] == expected
    tab = rp.check_equivalence(results, with_entry(pseudo, key, results.values[key] + 2e-3)).set_index("key")
    assert "lower" in tab.loc[key, "note"]


def test_tolerance_overrides_are_applied(results, pseudo):
    key = "log1/scm/sh/weight/Beijing"
    shifted = with_entry(pseudo, key, results.values[key] + 2e-3)
    loose = rp.check_equivalence(results, shifted, tolerances={"scm_weight": {"abs": 5e-3}}).set_index("key")
    strict = rp.check_equivalence(results, shifted).set_index("key")
    assert loose.loc[key, "status"] == "PASS" and strict.loc[key, "status"] == "FAIL"
    assert rp.TOLERANCES["scm_weight"]["abs"] == 6e-4  # module defaults are not mutated


def test_tolerances_carry_the_documented_values():
    assert rp.TOLERANCES == {
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


def test_deterministic_rows_pass_within_half_a_printed_unit_and_fail_beyond(results, pseudo):
    key = "log1/friend_did/1a/coef"
    base = pseudo["values"][key]["value"]
    half = rp.print_halfwidth(pseudo["values"][key]["dp"])
    assert half == pytest.approx(5e-7)
    for factor, expected in ((0.9, "PASS"), (1.1, "FAIL")):
        vals = dict(results.values)
        vals[key] = base + factor * half
        tab = rp.check_equivalence(rp.Results(values=vals, meta=results.meta), pseudo).set_index("key")
        assert tab.loc[key, "status"] == expected
        assert tab.loc[key, "tol"] == pytest.approx(half + 1e-9 * abs(base), rel=1e-3)


def _degenerate_targets(coef, se_ref, extra=None):
    """Reference values of one coefficient row (coefficient, standard error, t and a confidence limit)."""
    base = "log2/xtreg_twfe/1c/coef/year.2003"
    entries = {
        f"{base}/coef": {"value": coef, "dp": 7},
        f"{base}/se": {"value": se_ref, "dp": 7},
        f"{base}/t": {"value": coef / se_ref if se_ref else 0.0, "dp": 0},
        f"{base}/ci_hi": {"value": coef + 2.0 * se_ref, "dp": 7},
    }
    entries.update(extra or {})
    return {"values": entries}, base


@pytest.mark.parametrize("computed_se", [0.0, 6.0e-9, 7.0e-9, 9.0e-7])
@pytest.mark.parametrize("reference_se", [0.0, 2.0e-8])
def test_degenerate_rows_do_not_depend_on_the_noise_level_of_the_kernel(computed_se, reference_se):
    coef = 0.0312
    targets, base = _degenerate_targets(coef, reference_se)
    values = {
        f"{base}/coef": coef,
        f"{base}/se": computed_se,
        f"{base}/t": coef / computed_se if computed_se else float("inf"),
        f"{base}/ci_hi": coef + 2.0 * computed_se,
    }
    table = rp.check_equivalence(values, targets).set_index("key")
    assert (table.status == "PASS").all(), table[["reference", "python", "tol", "status", "note"]].to_string()
    assert "degenerate" in table.loc[f"{base}/t", "note"]


def test_a_real_standard_error_against_a_zero_reference_still_fails():
    coef = 0.0312
    targets, base = _degenerate_targets(coef, 0.0)
    values = {f"{base}/coef": coef, f"{base}/se": 1.0e-3, f"{base}/t": 31.2, f"{base}/ci_hi": coef}
    table = rp.check_equivalence(values, targets).set_index("key")
    assert table.loc[f"{base}/se", "status"] == "FAIL"


def test_the_noise_level_scales_with_the_coefficient():
    big = 5000.0
    targets, base = _degenerate_targets(big, 3.0e-4)
    values = {f"{base}/coef": big, f"{base}/se": 4.0e-3, f"{base}/t": 1.0e6, f"{base}/ci_hi": big}
    table = rp.check_equivalence(values, targets).set_index("key")
    assert table.loc[f"{base}/se", "status"] == "PASS"
    assert table.loc[f"{base}/se", "tol"] == pytest.approx(rp.SE_NOISE_RELATIVE * big)
    small = 0.02
    targets, base = _degenerate_targets(small, 3.0e-4)
    values = {f"{base}/coef": small, f"{base}/se": 3.0e-4, f"{base}/t": small / 3.0e-4, f"{base}/ci_hi": small + 6.0e-4}
    table = rp.check_equivalence(values, targets).set_index("key")
    assert (table.status == "PASS").all() and table["note"].eq("").all()


def test_sigma_u_note_names_matching_alternative_definitions(results, pseudo):
    key = "log2/xtreg_twfe/1a/sigma_u"
    alt = results.values[f"{key}_variant/adj_arith"]
    assert alt != results.values[key]
    tab = rp.check_equivalence(results, with_entry(pseudo, key, alt, dp=rp.printed_decimals(printed(alt)))).set_index("key")
    assert tab.loc[key, "status"] == "FAIL" and "adj_arith" in tab.loc[key, "note"]


def test_missing_printed_decimals_are_treated_as_exact(results, pseudo):
    key = "log1/scm/sh/weight/Beijing"
    exact_only = {"values": {**pseudo["values"], key: {"value": results.values[key] + 6.2e-4}}}
    tab = rp.check_equivalence(results, exact_only).set_index("key")
    assert tab.loc[key, "status"] == "FAIL"
    assert tab.loc[key, "tol"] == pytest.approx(rp.TOLERANCES["scm_weight"]["abs"])


def test_summary_and_formatted_table(results, pseudo):
    table = rp.check_equivalence(results, pseudo)
    s = rp.summarize_equivalence(table)
    assert s.loc["ALL", "PASS"] == len(table) and s.loc["ALL", "FAIL"] == 0 and s.loc["ALL", "MISSING"] == 0
    text = rp.format_check_table(table)
    assert "Equivalence summary by tolerance class" in text and "PASS" in text and "reference" in text
    vals = dict(results.values)
    vals["log1/friend_did/1a/coef"] += 1.0
    bad = rp.check_equivalence(rp.Results(values=vals, meta=results.meta), pseudo)
    assert "FAIL" in rp.format_check_table(bad)


def test_equivalence_with_the_reference_values_on_simulated_data_fails_loudly(results, reference):
    """Numbers from a simulated panel must not pass as the real results."""
    table = rp.check_equivalence(results, reference)
    det = table[table.kind == "deterministic"]
    assert (det.status == "FAIL").mean() > 0.9
    assert (table[table.kind == "mc_p"].status == "FAIL").all()


def test_equivalence_accepts_the_path_of_a_reference_file(results, reference, tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    a = rp.check_equivalence(results, path)
    b = rp.check_equivalence(results, reference)
    pd.testing.assert_frame_equal(a, b)


# ----------------------------------------------------------------------------
# Real panel (runs only on request)
# ----------------------------------------------------------------------------
def test_real_run_requires_the_switch_and_the_panel_file(sim, tmp_path):
    path = tmp_path / "panel.csv"
    sim.to_csv(path, index=False)
    assert rp.REAL_RUN_VARIABLE == "DTT_RUN_REAL"
    assert rp.real_run_enabled(path, env={"DTT_RUN_REAL": "1"})
    assert not rp.real_run_enabled(path, env={})
    assert not rp.real_run_enabled(path, env={"DTT_RUN_REAL": "0"})
    assert not rp.real_run_enabled(path, env={"DTT_RUN_REAL": "true"})
    assert not rp.real_run_enabled(tmp_path / "absent.csv", env={"DTT_RUN_REAL": "1"})
    assert rp.DEFAULT_PANEL == rp.REPO_ROOT / "data" / "raw" / "disney_did_panel.csv"


@pytest.mark.skipif(not rp.real_run_enabled(), reason="real-data check not run (set DTT_RUN_REAL=1 to run on the desktop)")
def test_real_panel_matches_the_reference_values(reference):
    df = rp.load_panel()
    res = rp.run_all(df)
    table = rp.check_equivalence(res, reference)
    for kind in ("count", "deterministic"):
        sub = table[table.kind == kind]
        assert (sub.status == "PASS").all(), sub[sub.status != "PASS"].head(20).to_string()


# ----------------------------------------------------------------------------
# Repository content
# ----------------------------------------------------------------------------
def test_repository_has_no_dependence_on_stata_files(root):
    banned_suffixes = {".dta", ".do", ".ado", ".log", ".smcl"}
    ignored = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    files = [p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & ignored)]
    assert [str(p.relative_to(root)) for p in files if p.suffix.lower() in banned_suffixes] == []
    reader = "read_" + "stata"
    writer = "to_" + "stata"
    sources = [p for p in (root / "src").rglob("*.py")]
    assert sources
    assert [p.name for p in sources if reader in p.read_text(encoding="utf-8") or writer in p.read_text(encoding="utf-8")] == []
