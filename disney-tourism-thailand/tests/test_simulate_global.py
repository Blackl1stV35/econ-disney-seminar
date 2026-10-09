"""Tests for dtt.simulate_global: file formats, schema, truth and feasibility of the synthetic test world.

All data are SIMULATED by the generator under test.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dtt.cases import CASE_COLUMNS, build_episodes, case_features, check_feasibility, load_cases  # noqa: E402
from dtt.panel import GROUPS, INDICATORS, build_global_panel, wdi_filename  # noqa: E402
from dtt.simulate_global import ANALYSIS_LAST_YEAR, SimWorld, simulate_global_world  # noqa: E402

N_ECON = 60
FIRST, LAST = 1995, 2024
N_YEARS = LAST - FIRST + 1


@pytest.fixture(scope="module")
def plain_world(tmp_path_factory) -> SimWorld:
    return simulate_global_world(tmp_path_factory.mktemp("plain"), seed=0)


@pytest.fixture(scope="module")
def cluster_world(tmp_path_factory) -> SimWorld:
    return simulate_global_world(tmp_path_factory.mktemp("cluster"), seed=0, cluster_economies=True)


@pytest.fixture(scope="module", params=["plain", "clustered"])
def world(request, plain_world, cluster_world) -> SimWorld:
    return plain_world if request.param == "plain" else cluster_world


@pytest.fixture(scope="module")
def panel(world) -> pd.DataFrame:
    return build_global_panel(world.raw_dir)


@pytest.fixture(scope="module")
def cases(world) -> pd.DataFrame:
    return load_cases(world.cases_path)


@pytest.fixture(scope="module")
def cluster_panel(cluster_world) -> pd.DataFrame:
    return build_global_panel(cluster_world.raw_dir)


@pytest.fixture(scope="module")
def cluster_cases(cluster_world) -> pd.DataFrame:
    return load_cases(cluster_world.cases_path)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def test_signature_defaults_and_return_type(world):
    import inspect

    params = inspect.signature(simulate_global_world).parameters
    assert list(params) == ["directory", "n_econ", "first_year", "last_year", "n_cases", "seed", "cluster_economies"]
    assert [params[k].default for k in list(params)[1:]] == [60, 1995, 2024, 16, 0, False]
    assert isinstance(world, SimWorld)
    assert world.n_econ == N_ECON and world.first_year == FIRST and world.last_year == LAST


HISTORY_PATTERN = re.compile(
    r"\b(as before|before this (?:option|argument|change)|previously|formerly|no longer|used to|was added|were added|"
    r"has been added|have been added|existed|originally|earlier versions?|older versions?|old behaviou?r|new behaviou?r)\b",
    re.IGNORECASE,
)


def test_documentation_states_behaviour_without_narrating_history():
    import inspect

    import dtt.simulate_global as module

    assert HISTORY_PATTERN.search("The output for False is the same as before this option existed.")
    assert HISTORY_PATTERN.search("This argument was added later.") and not HISTORY_PATTERN.search("The year before the opening.")
    hits = [line.strip() for line in inspect.getsource(module).splitlines() if HISTORY_PATTERN.search(line)]
    assert hits == []


def test_world_records_whether_economies_are_shared(plain_world, cluster_world):
    assert plain_world.cluster_economies is False and cluster_world.cluster_economies is True


def test_twenty_indicator_files_in_raw_format(world):
    root = world.raw_dir
    expected = [wdi_filename(g, code) for g in GROUPS for code in INDICATORS]
    assert len(expected) == 20
    for name in expected:
        assert (root / name).is_file(), name
        assert name in world.paths
    assert (root / "country_metadata.csv").is_file()
    assert world.cases_path.is_file()

    groups_codes: dict[str, set[str]] = {}
    for g in GROUPS:
        sets = []
        for code in INDICATORS:
            header, rows = read_rows(root / wdi_filename(g, code))
            assert header == ["iso3", "year", "value"]
            codes = sorted({r["iso3"] for r in rows})
            assert len(rows) == len(codes) * N_YEARS
            for c in codes:
                years = [int(r["year"]) for r in rows if r["iso3"] == c]
                assert years == list(range(FIRST, LAST + 1))
            for r in rows:
                assert r["value"] == "" or re.fullmatch(r"\d+", r["value"]), r
            sets.append(set(codes))
        assert all(s == sets[0] for s in sets)
        groups_codes[g] = sets[0]
    assert all(len(groups_codes[g]) == N_ECON // 4 for g in GROUPS)
    union = set().union(*groups_codes.values())
    assert sum(len(s) for s in groups_codes.values()) == len(union) == N_ECON
    assert union == {f"SA{i:02d}" for i in range(N_ECON)}


def test_missing_values_are_empty_cells_in_the_noisy_indicators(world):
    root = world.raw_dir
    n_empty = {}
    for code, variable in INDICATORS.items():
        n_empty[variable] = sum(
            sum(1 for r in read_rows(root / wdi_filename(g, code))[1] if r["value"] == "") for g in GROUPS
        )
    assert n_empty["receipts_usd"] > 0
    assert n_empty["arrivals"] > 0
    assert n_empty["air_pax"] > 0
    assert n_empty["gdp_usd"] == 0
    assert n_empty["pop"] == 0


def test_metadata_covers_every_economy(world):
    meta = pd.read_csv(world.paths["metadata"])
    assert list(meta.columns) == ["iso3", "name", "region", "income_level", "capital", "longitude", "latitude"]
    assert len(meta) == N_ECON and meta["iso3"].is_unique
    assert meta["iso3"].tolist() == [f"SA{i:02d}" for i in range(N_ECON)]
    assert meta["region"].nunique() > 1 and meta["income_level"].nunique() > 1


def test_paths_dictionary_lists_every_file(world):
    assert world.paths["raw_dir"] == world.raw_dir
    assert world.paths["cases"] == world.cases_path
    assert world.paths["metadata"].name == "country_metadata.csv"
    written = {p.name for p in world.raw_dir.iterdir()}
    assert written == {k for k in world.paths if k.endswith(".csv")}
    assert len(written) == 22


def test_same_seed_gives_identical_files_and_other_seed_differs(tmp_path):
    a = simulate_global_world(tmp_path / "a", seed=3)
    b = simulate_global_world(tmp_path / "b", seed=3)
    c = simulate_global_world(tmp_path / "c", seed=4)
    names = sorted(p.name for p in a.raw_dir.iterdir())
    assert names == sorted(p.name for p in b.raw_dir.iterdir())
    assert all((a.raw_dir / n).read_bytes() == (b.raw_dir / n).read_bytes() for n in names)
    assert any((a.raw_dir / n).read_bytes() != (c.raw_dir / n).read_bytes() for n in names)
    pd.testing.assert_frame_equal(a.true_effects, b.true_effects)
    assert not a.true_effects.equals(c.true_effects)


# ----------------------------------------------------------------------------
# Cases catalogue
# ----------------------------------------------------------------------------
def test_catalogue_has_the_21_column_schema(world, cases):
    header, rows = read_rows(world.cases_path)
    assert header == list(CASE_COLUMNS) and len(header) == 21
    assert len(rows) == 16 == len(cases)
    assert cases["case_id"].is_unique
    assert cases["opening_year"].dtype == np.int64
    assert cases["investment_usd_bn_nominal"].notna().all() and (cases["investment_usd_bn_nominal"] > 0).all()
    assert set(cases["iso3"]) <= set(pd.read_csv(world.paths["metadata"])["iso3"])
    assert cases["category"].nunique() > 1
    assert (cases["opening_year"].diff().dropna() >= 0).all()


def test_true_effects_table_layout(world):
    assert list(world.true_effects.columns) == ["case_id", "true_att_pp", "true_att_rel_pct"]
    assert len(world.true_effects) == 16
    assert world.true_effects["case_id"].is_unique
    assert world.true_effects[["true_att_pp", "true_att_rel_pct"]].notna().all().all()
    assert list(world.drivers["case_id"]) == list(world.true_effects["case_id"])


def test_true_effect_follows_the_generating_formula(world):
    b = world.coefficients
    assert set(b) >= {"b0", "b1", "b2"}
    assert b["b1"] != 0.0 and b["b2"] == 0.0
    d = world.drivers.merge(world.true_effects, on="case_id")
    expected = b["b0"] + b["b1"] * d["capex_pct_gdp"] + b["b2"] * d["receipts_pct_gdp_pre"] + d["effect_noise"]
    np.testing.assert_allclose(d["true_att_pp"], expected, atol=1e-12)
    assert d["capex_pct_gdp"].between(0.05, 4.0).all()
    assert d["capex_pct_gdp"].std() > 0.3
    assert d["receipts_pct_gdp_pre"].std() > 0.3


def test_drivers_equal_the_features_computed_by_dtt_cases(world, panel, cases):
    feats = case_features(cases, panel).set_index("case_id")
    d = world.drivers.set_index("case_id")
    np.testing.assert_allclose(feats["capex_pct_gdp"], d["capex_pct_gdp"], rtol=1e-9)
    np.testing.assert_allclose(feats["receipts_pct_gdp"], d["receipts_pct_gdp_pre"], atol=1e-6)
    np.testing.assert_allclose(cases.set_index("case_id")["investment_usd_bn_nominal"], d["investment_usd_bn_nominal"])


def test_effect_is_added_from_the_opening_year_on(world, panel):
    merged = panel.merge(world.counterfactual, on=["iso3", "year"], validate="one_to_one")
    assert len(merged) == len(panel)
    merged["diff"] = merged["receipts_pct_gdp"] - merged["receipts_pct_gdp_cf"]
    truth = world.drivers.merge(world.true_effects, on="case_id")
    case_econ = set(truth["iso3"])
    others = merged.loc[~merged["iso3"].isin(case_econ) & merged["diff"].notna()]
    assert len(others) > 1000
    assert others["diff"].abs().max() < 1e-6
    for iso3, group in truth.groupby("iso3"):
        sub = merged.loc[(merged["iso3"] == iso3) & merged["diff"].notna()]
        first_opening = group["opening_year"].min()
        assert (sub["year"] < first_opening).any() and (sub["year"] >= first_opening).any()
        shift = [sum(g.true_att_pp for g in group.itertuples() if g.opening_year <= year) for year in sub["year"]]
        np.testing.assert_allclose(sub["diff"], shift, atol=1e-6)


def test_relative_truth_uses_the_untreated_mean_over_the_analysis_window(world):
    cf = world.counterfactual
    truth = world.drivers.merge(world.true_effects, on="case_id")
    for row in truth.itertuples():
        end = ANALYSIS_LAST_YEAR if row.opening_year <= ANALYSIS_LAST_YEAR else LAST
        sub = cf.loc[(cf["iso3"] == row.iso3) & cf["year"].between(row.opening_year, end), "receipts_pct_gdp_cf"]
        np.testing.assert_allclose(row.true_att_rel_pct, 100.0 * row.true_att_pp / sub.mean(), rtol=1e-12)


def test_untreated_outcome_is_positive_and_has_three_factor_structure(world):
    cf = world.counterfactual.pivot(index="year", columns="iso3", values="receipts_pct_gdp_cf")
    assert (cf.to_numpy() > 0).all()
    s = np.linalg.svd(cf.to_numpy() - cf.to_numpy().mean(axis=0), compute_uv=False)
    share = s**2 / np.sum(s**2)
    assert share[:3].sum() > 0.8
    assert share[3] < 0.03


# ----------------------------------------------------------------------------
# Feasibility by design
# ----------------------------------------------------------------------------
def members_of(episodes: pd.DataFrame) -> dict[str, list[str]]:
    """Case identifiers of the members of every episode."""
    return {case_id: ids.split(";") for case_id, ids in zip(episodes["case_id"], episodes["member_case_ids"])}


def failing_cases(episodes: pd.DataFrame, feas: pd.DataFrame) -> set[str]:
    """Identifiers of the cases that belong to an infeasible episode."""
    members = members_of(episodes)
    return {case for case_id in feas.loc[~feas["feasible"], "case_id"] for case in members[case_id]}


def test_at_least_14_cases_satisfy_the_feasibility_rules(world, panel, cases):
    episodes = build_episodes(cases)
    feas = check_feasibility(episodes, panel, catalogue=cases)
    assert failing_cases(episodes, feas) == set(world.infeasible_case_ids)
    assert len(world.infeasible_case_ids) == 2
    covered = int(episodes.loc[feas["feasible"].to_numpy(), "n_openings"].sum())
    assert covered == len(cases) - 2 and covered >= 14
    assert (feas.loc[feas["feasible"], "reason"] == "").all()
    assert (feas.loc[~feas["feasible"], "reason"] != "").all()
    members = members_of(episodes)
    per_case = {case: bool(ok) for case_id, ok in zip(feas["case_id"], feas["feasible"]) for case in members[case_id]}
    assert cases.set_index("case_id")["panel_feasible"].astype(bool).to_dict() == per_case
    assert episodes["panel_feasible"].astype(bool).tolist() == feas["feasible"].tolist()


def test_cases_of_an_economy_of_their_own_are_feasible_one_by_one(plain_world):
    cs = load_cases(plain_world.cases_path)
    pan = build_global_panel(plain_world.raw_dir)
    feas = check_feasibility(cs, pan)
    assert int(feas["feasible"].sum()) == 14
    assert set(feas.loc[~feas["feasible"], "case_id"]) == set(plain_world.infeasible_case_ids)
    assert not feas["reason"].str.contains("other_openings").any()
    episodes = build_episodes(cs)
    assert (episodes["n_openings"] == 1).all()
    by_episode = check_feasibility(episodes, pan, catalogue=cs).assign(case_id=episodes["member_case_ids"].to_numpy())
    pd.testing.assert_frame_equal(by_episode.set_index("case_id").loc[feas["case_id"]].reset_index(), feas, check_dtype=False)


def test_built_in_infeasible_cases_fail_for_the_documented_reasons(world, panel, cases):
    feas = check_feasibility(cases, panel).set_index("case_id")
    bad = feas.loc[world.infeasible_case_ids]
    assert bad["reason"].str.contains("n_pre").sum() == 1
    assert bad["reason"].str.contains("n_post").sum() == 1
    assert (cases.set_index("case_id").loc[world.infeasible_case_ids, "opening_year"].isin([FIRST + 2, LAST])).all()


def test_missing_data_patterns_exist_for_the_downstream_code(world, panel, cases):
    window = panel.loc[panel["year"].between(FIRST, ANALYSIS_LAST_YEAR)]
    complete = window.groupby("iso3")["receipts_pct_gdp"].apply(lambda s: bool(s.notna().all()))
    case_econ = set(cases["iso3"])
    gappy_others = [c for c in complete.index[~complete] if c not in case_econ]
    assert len(gappy_others) >= 2
    feas = check_feasibility(cases, panel)
    ok = feas.loc[feas["feasible"]]
    assert (ok["n_pre"] < ok["opening_year"] - FIRST).sum() == 1
    feats = case_features(cases, panel)
    assert feats[["arrivals_per_capita", "air_pax_per_capita", "log_receipts_per_arrival"]].isna().any().any()


# ----------------------------------------------------------------------------
# Other sizes and invalid arguments
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_econ,n_cases,first,last,n_feasible",
    [(30, 8, 1995, 2019, 7), (24, 3, 1995, 2024, 3), (40, 20, 1995, 2024, 18), (30, 4, 2000, 2012, 4)],
)
def test_other_world_sizes(tmp_path, n_econ, n_cases, first, last, n_feasible):
    w = simulate_global_world(tmp_path, n_econ=n_econ, first_year=first, last_year=last, n_cases=n_cases, seed=2)
    pan = build_global_panel(w.raw_dir)
    assert pan["iso3"].nunique() == n_econ and pan["year"].min() == first and pan["year"].max() == last
    cs = load_cases(w.cases_path)
    assert len(cs) == n_cases
    feas = check_feasibility(cs, pan, first_year=first, last_year=min(last, ANALYSIS_LAST_YEAR))
    assert int(feas["feasible"].sum()) == n_feasible
    assert feas["n_donors"].min() >= 15


def test_economy_codes_widen_with_many_economies(tmp_path):
    w = simulate_global_world(tmp_path, n_econ=110, n_cases=2, seed=1)
    meta = pd.read_csv(w.paths["metadata"])
    assert meta["iso3"].iloc[0] == "SA000" and meta["iso3"].iloc[-1] == "SA109"
    assert build_global_panel(w.raw_dir)["iso3"].nunique() == 110


@pytest.mark.parametrize(
    "kwargs,pattern",
    [
        ({"n_cases": 0}, "n_cases"),
        ({"n_econ": 30, "n_cases": 16}, "n_econ"),
        ({"first_year": 2010, "last_year": 2015}, "year range"),
    ],
)
def test_invalid_arguments_raise_value_error(tmp_path, kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        simulate_global_world(tmp_path / "never_created", **kwargs)
    assert not (tmp_path / "never_created").exists()


def test_output_directory_is_created(tmp_path):
    target = tmp_path / "nested" / "world"
    w = simulate_global_world(target, n_econ=30, n_cases=4, seed=1)
    assert target.is_dir() and w.raw_dir == target


# ----------------------------------------------------------------------------
# Shared economies
# ----------------------------------------------------------------------------
def opening_years_by_economy(catalogue: pd.DataFrame) -> dict[str, list[int]]:
    """Opening years of each economy that hosts more than one case, in order."""
    grouped = catalogue.groupby("iso3")["opening_year"].apply(lambda s: [int(y) for y in s])
    return {iso3: years for iso3, years in grouped.items() if len(years) > 1}


def test_plain_world_has_one_case_per_economy_and_no_overlap_entries(plain_world):
    cs = load_cases(plain_world.cases_path)
    assert cs["iso3"].is_unique and not opening_years_by_economy(cs)
    _, rows = read_rows(plain_world.cases_path)
    assert {r["other_openings_same_economy_within_5y"] for r in rows} == {"0"}


def test_default_call_equals_cluster_economies_false(tmp_path):
    a = simulate_global_world(tmp_path / "a", seed=3)
    b = simulate_global_world(tmp_path / "b", seed=3, cluster_economies=False)
    names = sorted(p.name for p in a.raw_dir.iterdir())
    assert names == sorted(p.name for p in b.raw_dir.iterdir()) and len(names) == 22
    assert all((a.raw_dir / n).read_bytes() == (b.raw_dir / n).read_bytes() for n in names)
    pd.testing.assert_frame_equal(a.true_effects, b.true_effects)
    pd.testing.assert_frame_equal(a.drivers, b.drivers)
    pd.testing.assert_frame_equal(a.counterfactual, b.counterfactual)
    assert a.infeasible_case_ids == b.infeasible_case_ids and not a.cluster_economies and not b.cluster_economies


def test_cluster_economies_host_three_and_two_cases_a_few_years_apart(cluster_cases):
    shared = opening_years_by_economy(cluster_cases)
    assert len(cluster_cases) == 16 and cluster_cases["case_id"].is_unique and cluster_cases["iso3"].nunique() == 13
    assert sorted(len(years) for years in shared.values()) == [2, 3]
    three = next(years for years in shared.values() if len(years) == 3)
    two = next(years for years in shared.values() if len(years) == 2)
    assert all(2 <= gap <= 4 for gap in np.diff(three))
    assert 3 <= two[1] - two[0] <= 5
    assert (cluster_cases["opening_year"].diff().dropna() >= 0).all()
    assert cluster_cases.groupby("iso3")[["economy", "location"]].nunique().to_numpy().max() == 1
    assert cluster_cases["opening_year"].between(FIRST, LAST).all()


def test_overlap_column_lists_the_other_openings_of_the_economy_within_five_years(cluster_world, cluster_cases):
    _, rows = read_rows(cluster_world.cases_path)
    year = dict(zip(cluster_cases["case_id"], cluster_cases["opening_year"]))
    economy = dict(zip(cluster_cases["case_id"], cluster_cases["iso3"]))
    listed = 0
    for row in rows:
        this = row["case_id"]
        expected = sorted(c for c in year if c != this and economy[c] == economy[this] and abs(year[c] - year[this]) <= 5)
        got = sorted(c for c in row["other_openings_same_economy_within_5y"].split(";") if c)
        assert got == expected
        listed += len(got)
    assert listed >= 6
    singles = set(cluster_cases["iso3"]) - set(opening_years_by_economy(cluster_cases))
    assert {r["other_openings_same_economy_within_5y"] for r in rows if economy[r["case_id"]] in singles} == {""}


def test_level_shifts_of_one_economy_add_up_from_each_opening_year(cluster_world, cluster_panel):
    merged = cluster_panel.merge(cluster_world.counterfactual, on=["iso3", "year"], validate="one_to_one")
    merged["diff"] = merged["receipts_pct_gdp"] - merged["receipts_pct_gdp_cf"]
    truth = cluster_world.drivers.merge(cluster_world.true_effects, on="case_id")
    for iso3, years in opening_years_by_economy(truth).items():
        group = truth.loc[truth["iso3"] == iso3]
        sub = merged.loc[(merged["iso3"] == iso3) & merged["diff"].notna()].set_index("year")["diff"]
        steps = np.diff(sub.to_numpy())
        for year, effect in zip(group["opening_year"], group["true_att_pp"]):
            assert sub[year] - sub[year - 1] == pytest.approx(effect, abs=1e-6)
        total = group["true_att_pp"].sum()
        assert sub.loc[years[-1]:].to_numpy() == pytest.approx(total, abs=1e-6)
        assert (np.abs(steps) > 1e-6).sum() == len(years)


def test_window_effect_is_the_mean_level_shift_of_the_economy_over_the_years(cluster_world, cluster_panel):
    merged = cluster_panel.merge(cluster_world.counterfactual, on=["iso3", "year"], validate="one_to_one")
    merged["diff"] = merged["receipts_pct_gdp"] - merged["receipts_pct_gdp_cf"]
    truth = cluster_world.drivers.merge(cluster_world.true_effects, on="case_id")
    for iso3 in sorted(set(truth["iso3"])):
        sub = merged.loc[(merged["iso3"] == iso3) & merged["diff"].notna()].set_index("year")["diff"]
        annual = pd.Series({y: cluster_world.window_effect(iso3, y, y) for y in range(FIRST, LAST + 1)})
        np.testing.assert_allclose(annual.loc[sub.index], sub, atol=1e-6)
        for first, last in [(FIRST, LAST), (2000, 2008), (2003, 2003), (2010, 2019)]:
            assert cluster_world.window_effect(iso3, first, last) == pytest.approx(annual.loc[first:last].mean(), abs=1e-12)
        opened = truth.loc[truth["iso3"] == iso3, "opening_year"].min()
        assert cluster_world.window_effect(iso3, FIRST, opened - 1) == 0.0
    single = truth.groupby("iso3").filter(lambda g: len(g) == 1).iloc[0]
    assert cluster_world.window_effect(single["iso3"], single["opening_year"], 2019) == pytest.approx(single["true_att_pp"], abs=1e-12)
    never = sorted(set(cluster_panel["iso3"]) - set(truth["iso3"]))[0]
    assert cluster_world.window_effect(never, FIRST, LAST) == 0.0
    with pytest.raises(ValueError, match="empty year range"):
        cluster_world.window_effect(single["iso3"], 2010, 2009)


def test_drivers_of_later_cases_in_a_shared_economy_include_the_earlier_shifts(cluster_world, cluster_panel, cluster_cases):
    feats = case_features(cluster_cases, cluster_panel).set_index("case_id")
    drivers = cluster_world.drivers.set_index("case_id")
    np.testing.assert_allclose(feats["capex_pct_gdp"], drivers["capex_pct_gdp"], rtol=1e-9)
    np.testing.assert_allclose(feats["receipts_pct_gdp"], drivers["receipts_pct_gdp_pre"], atol=1e-6)
    b = cluster_world.coefficients
    merged = drivers.join(cluster_world.true_effects.set_index("case_id"))
    expected = b["b0"] + b["b1"] * merged["capex_pct_gdp"] + b["b2"] * merged["receipts_pct_gdp_pre"] + merged["effect_noise"]
    np.testing.assert_allclose(merged["true_att_pp"], expected, atol=1e-12)
    later = [case_id for _, group in cluster_cases.groupby("iso3") for case_id in group["case_id"].iloc[1:]]
    assert len(later) == 3
    untreated = cluster_world.counterfactual.set_index(["iso3", "year"])["receipts_pct_gdp_cf"]
    for case_id in later:
        iso3, opening = drivers.loc[case_id, ["iso3", "opening_year"]]
        untreated_pre = untreated.loc[[(iso3, y) for y in range(opening - 3, opening)]].mean()
        assert drivers.loc[case_id, "receipts_pct_gdp_pre"] > untreated_pre + 0.1


def test_the_first_opening_share_of_an_episode_is_the_driver_of_its_first_case(cluster_world, cluster_panel, cluster_cases):
    episodes = build_episodes(cluster_cases)
    feats = case_features(episodes, cluster_panel).set_index("case_id")
    drivers = cluster_world.drivers.set_index("case_id")
    for row in episodes.itertuples():
        first = row.member_case_ids.split(";")[0]
        assert feats.loc[row.case_id, "capex_first_pct_gdp"] == pytest.approx(drivers.loc[first, "capex_pct_gdp"], rel=1e-9), row.case_id
    shared = episodes.loc[episodes["n_openings"] > 1, "case_id"]
    assert len(shared) == 2
    assert (feats.loc[shared, "capex_pct_gdp"] >= feats.loc[shared, "capex_first_pct_gdp"]).all()
    assert (feats.loc[shared, "capex_pct_gdp"] > feats.loc[shared, "capex_first_pct_gdp"]).any()


def test_cluster_world_satisfies_the_feasibility_rules_by_design(cluster_world, cluster_panel, cluster_cases):
    episodes = build_episodes(cluster_cases)
    feas = check_feasibility(episodes, cluster_panel, catalogue=cluster_cases)
    assert len(episodes) == 13 and int(feas["feasible"].sum()) == 11
    assert failing_cases(episodes, feas) == set(cluster_world.infeasible_case_ids)
    assert len(cluster_world.infeasible_case_ids) == 2
    shared = set(opening_years_by_economy(cluster_cases))
    assert feas.loc[feas["iso3"].isin(shared), "feasible"].all()
    assert feas.loc[feas["iso3"].isin(shared), "n_donors"].min() >= 15
    members = members_of(episodes)
    per_case = {case: bool(ok) for case_id, ok in zip(feas["case_id"], feas["feasible"]) for case in members[case_id]}
    assert cluster_cases.set_index("case_id")["panel_feasible"].astype(bool).to_dict() == per_case
    assert int(episodes.loc[feas["feasible"].to_numpy(), "n_openings"].sum()) >= 14


def test_members_of_a_shared_economy_followed_by_another_opening_fail_alone_on_that_rule(cluster_panel, cluster_cases):
    raw = check_feasibility(cluster_cases, cluster_panel).set_index("case_id")
    followed = raw.loc[raw["reason"].str.contains("other_openings")]
    shared = set(opening_years_by_economy(cluster_cases))
    assert len(followed) >= 1 and set(followed["iso3"]) <= shared
    assert followed["reason"].str.fullmatch(r"other_openings \d+ in post window \(years [\d, ]+\)").all()
    assert not followed["feasible"].any()
    last_of_economy = cluster_cases.groupby("iso3")["case_id"].last()
    assert raw.loc[last_of_economy.loc[sorted(shared)], "feasible"].all()


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_cluster_structure_and_feasibility_hold_for_other_seeds(tmp_path, seed):
    w = simulate_global_world(tmp_path, seed=seed, cluster_economies=True)
    cs = load_cases(w.cases_path)
    shared = opening_years_by_economy(cs)
    assert sorted(len(years) for years in shared.values()) == [2, 3] and cs["iso3"].nunique() == 13
    three = next(years for years in shared.values() if len(years) == 3)
    two = next(years for years in shared.values() if len(years) == 2)
    assert all(2 <= gap <= 4 for gap in np.diff(three)) and 3 <= two[1] - two[0] <= 5
    episodes = build_episodes(cs)
    feas = check_feasibility(episodes, build_global_panel(w.raw_dir), catalogue=cs)
    assert len(episodes) == 13 and int(feas["feasible"].sum()) == 11
    assert failing_cases(episodes, feas) == set(w.infeasible_case_ids)
    assert int(episodes.loc[feas["feasible"].to_numpy(), "n_openings"].sum()) >= 14
    assert (feas.loc[feas["feasible"], "n_pre"] < feas.loc[feas["feasible"], "opening_year"] - FIRST).sum() == 1


@pytest.mark.parametrize("n_econ,n_cases,first,last,n_feasible", [(30, 8, 1995, 2019, 7), (36, 12, 1995, 2019, 11), (40, 20, 1995, 2024, 18)])
def test_other_world_sizes_with_shared_economies(tmp_path, n_econ, n_cases, first, last, n_feasible):
    w = simulate_global_world(tmp_path, n_econ=n_econ, first_year=first, last_year=last, n_cases=n_cases, seed=2, cluster_economies=True)
    pan = build_global_panel(w.raw_dir)
    cs = load_cases(w.cases_path)
    assert len(cs) == n_cases and cs["iso3"].nunique() == n_cases - 3
    assert sorted(len(y) for y in opening_years_by_economy(cs).values()) == [2, 3]
    episodes = build_episodes(cs)
    feas = check_feasibility(episodes, pan, first_year=first, last_year=min(last, ANALYSIS_LAST_YEAR), catalogue=cs)
    assert len(episodes) == n_cases - 3 and failing_cases(episodes, feas) == set(w.infeasible_case_ids)
    assert int(episodes.loc[feas["feasible"].to_numpy(), "n_openings"].sum()) == n_feasible
    assert int(feas["feasible"].sum()) == n_feasible - 3 and feas["n_donors"].min() >= 15


def test_shared_economies_are_reproducible_and_differ_from_the_plain_world(tmp_path):
    a = simulate_global_world(tmp_path / "a", seed=3, cluster_economies=True)
    b = simulate_global_world(tmp_path / "b", seed=3, cluster_economies=True)
    c = simulate_global_world(tmp_path / "c", seed=3)
    names = sorted(p.name for p in a.raw_dir.iterdir())
    assert all((a.raw_dir / n).read_bytes() == (b.raw_dir / n).read_bytes() for n in names)
    pd.testing.assert_frame_equal(a.true_effects, b.true_effects)
    pd.testing.assert_frame_equal(a.drivers, b.drivers)
    assert (a.raw_dir / "cases_catalogue.csv").read_bytes() != (c.raw_dir / "cases_catalogue.csv").read_bytes()


@pytest.mark.parametrize("kwargs,pattern", [({"n_cases": 7}, "cluster_economies"), ({"last_year": 2010}, "cluster_economies")])
def test_cluster_option_rejects_worlds_too_small_for_it(tmp_path, kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        simulate_global_world(tmp_path / "never_created", cluster_economies=True, **kwargs)
    assert not (tmp_path / "never_created").exists()
    plain = simulate_global_world(tmp_path / "plain", cluster_economies=False, **kwargs)
    assert plain.cluster_economies is False


def test_smallest_worlds_that_accept_the_cluster_option(tmp_path):
    few = simulate_global_world(tmp_path / "few", n_cases=8, seed=4, cluster_economies=True)
    assert len(few.true_effects) == 8 and sorted(len(y) for y in opening_years_by_economy(load_cases(few.cases_path)).values()) == [2, 3]
    short = simulate_global_world(tmp_path / "short", last_year=2011, seed=4, cluster_economies=True)
    cs = load_cases(short.cases_path)
    assert sorted(len(y) for y in opening_years_by_economy(cs).values()) == [2, 3] and cs["opening_year"].between(FIRST, 2011).all()
