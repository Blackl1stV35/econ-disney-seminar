"""Tests of the run context used by the notebooks (simulated data only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dtt import impact, workflow as wf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sim_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("sim_world")
    return {wf.WORLD_VARIABLE: "sim", wf.SIM_DIR_VARIABLE: str(base / "world")}


@pytest.fixture()
def keep_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


def test_find_root_from_a_subdirectory():
    assert wf.find_root(ROOT / "tests") == ROOT


def test_find_root_fails_outside_a_repository(tmp_path):
    with pytest.raises(FileNotFoundError):
        wf.find_root(tmp_path)


@pytest.mark.parametrize(
    "env, expected",
    [
        ({wf.WORLD_VARIABLE: "sim"}, "sim"),
        ({wf.WORLD_VARIABLE: "SIM", wf.REAL_VARIABLE: "1"}, "sim"),
        ({wf.REAL_VARIABLE: "1"}, "real"),
        ({wf.WORLD_VARIABLE: "real", wf.REAL_VARIABLE: "1"}, "real"),
    ],
)
def test_resolve_mode(env, expected):
    assert wf.resolve_mode(env) == expected


@pytest.mark.parametrize(
    "env",
    [
        {},
        {wf.REAL_VARIABLE: "0"},
        {wf.REAL_VARIABLE: "true"},
        {wf.REAL_VARIABLE: "yes"},
        {wf.REAL_VARIABLE: "2"},
        {wf.WORLD_VARIABLE: "real"},
        {wf.WORLD_VARIABLE: "real", wf.REAL_VARIABLE: "on"},
    ],
)
def test_real_mode_needs_the_explicit_switch(env):
    with pytest.raises(RuntimeError, match="DTT_RUN_REAL"):
        wf.resolve_mode(env)


def test_unknown_world_is_rejected():
    with pytest.raises(ValueError):
        wf.resolve_mode({wf.WORLD_VARIABLE: "other"})


def test_sim_context_builds_a_world_and_a_target(sim_env, keep_sys_path):
    ctx = wf.setup_run("test_stage", root=ROOT, env=sim_env)
    assert ctx.mode == "sim" and not ctx.is_real
    assert ctx.guard is None
    cat = pd.read_csv(ctx.catalogue_path, encoding="utf-8-sig")
    assert ctx.target_iso3 not in set(cat["iso3"])
    assert str(ctx.results_dir).startswith(sim_env[wf.SIM_DIR_VARIABLE])
    assert ROOT / "results" not in ctx.results_dir.parents and ctx.results_dir != ROOT / "results"
    baseline = impact.load_thailand_baseline(ctx.baseline_path)
    assert {"gdp_usd_bn", "receipts_usd_bn", "receipts_pct_gdp", "fx_thb_per_usd"} <= set(baseline.columns)
    assert 2019 in baseline.index and 2024 in baseline.index
    scenario = ctx.primary_scenario()
    assert scenario["primary"] is True and scenario["capex_thb_bn"] > 0


def test_second_setup_reuses_the_world(sim_env, keep_sys_path):
    first = wf.setup_run("a", root=ROOT, env=sim_env)
    stamp = first.catalogue_path.stat().st_mtime_ns
    second = wf.setup_run("b", root=ROOT, env=sim_env)
    assert second.catalogue_path.stat().st_mtime_ns == stamp
    assert second.target_iso3 == first.target_iso3


def test_table_json_and_figure_round_trip(sim_env, keep_sys_path):
    ctx = wf.setup_run("roundtrip", root=ROOT, env=sim_env)
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]})
    path = ctx.save_table(df, "t_example")
    assert path.name == "t_example.csv"
    back = ctx.read_table("t_example")
    assert back.shape == (2, 2) and back["a"].tolist() == [1, 2]
    ctx.require("t_example")
    with pytest.raises(FileNotFoundError, match="missing_table"):
        ctx.require("t_example", "missing_table")
    with pytest.raises(FileNotFoundError):
        ctx.read_table("missing_table")

    jpath = ctx.save_json({"x": np.float64(1.5), "n": np.int64(3), "flag": np.bool_(True), "arr": np.arange(3), "nan": np.float64("nan")}, "j_example")
    loaded = json.loads(jpath.read_text(encoding="utf-8"))
    assert loaded == {"x": 1.5, "n": 3, "flag": True, "arr": [0, 1, 2], "nan": None}
    assert ctx.read_json("j_example") == loaded

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of GDP (percent)")
    ax.legend()
    fpath = ctx.save_figure(fig, "fig_example.png", show=False)
    assert fpath.is_file() and fpath.stat().st_size > 1000
    inventory = pd.read_csv(ctx.results_dir / "figure_inventory_roundtrip.csv")
    assert inventory.loc[0, "x labels"] == "Year"
    assert inventory.loc[0, "y labels"] == "Share of GDP (percent)"
    assert inventory.loc[0, "legend entries"] == 1


def test_results_directory_can_be_overridden(sim_env, tmp_path, keep_sys_path):
    env = dict(sim_env)
    env[wf.RESULTS_VARIABLE] = str(tmp_path / "out" / "r")
    env[wf.FIGURES_VARIABLE] = str(tmp_path / "out" / "f")
    ctx = wf.setup_run("override", root=ROOT, env=env)
    assert ctx.results_dir == tmp_path / "out" / "r" and ctx.results_dir.is_dir()
    assert ctx.figures_dir == tmp_path / "out" / "f" and ctx.figures_dir.is_dir()


def test_guard_is_optional_in_a_simulated_run(sim_env, tmp_path, keep_sys_path):
    env = dict(sim_env)
    env[wf.RESULTS_VARIABLE] = str(tmp_path / "r")
    env[wf.GUARD_VARIABLE] = "1"
    ctx = wf.setup_run("guarded", root=ROOT, env=env)
    assert callable(ctx.guard) and ctx.guard_object is not None
    ctx.guard()
    assert (tmp_path / "r" / "thermal_log.csv").is_file()


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "dtt").mkdir(parents=True)
    (root / "src" / "dtt" / "__init__.py").write_text("", encoding="utf-8")
    (root / "data" / "raw" / "wdi").mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "raw" / "wdi" / "country_metadata.csv").write_text("iso3\n", encoding="utf-8")
    (root / "data" / "processed" / "cases_catalogue.csv").write_text("case_id\n", encoding="utf-8")
    (root / "data" / "processed" / "thailand_baseline.csv").write_text("year,variable,value\n", encoding="utf-8")
    (root / "data" / "processed" / "thailand_proposal.json").write_text(
        json.dumps({"status": "x", "scenarios": [{"name": "s1", "capex_thb_bn": 10, "primary": False}, {"name": "s2", "capex_thb_bn": 20, "primary": True}]}),
        encoding="utf-8",
    )
    return root


def test_real_context_reads_no_data_and_uses_repository_paths(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    env = {wf.REAL_VARIABLE: "1", wf.GUARD_VARIABLE: "0"}
    ctx = wf.setup_run("real_stage", root=root, env=env)
    assert ctx.is_real and ctx.target_iso3 == "THA"
    assert ctx.raw_dir == root / "data" / "raw" / "wdi"
    assert ctx.results_dir == root / "results" and ctx.figures_dir == root / "figures"
    assert ctx.investment_fx_path is None
    assert ctx.guard is None
    assert ctx.primary_scenario()["name"] == "s2"


def test_real_context_uses_the_investment_file_when_present_and_guard_by_default(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    (root / "data" / "processed" / "cases_investment_fx.csv").write_text("case_id\n", encoding="utf-8")
    ctx = wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1"})
    assert ctx.investment_fx_path == root / "data" / "processed" / "cases_investment_fx.csv"
    assert callable(ctx.guard) and ctx.guard_object is not None


def test_real_context_reports_missing_inputs(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    (root / "data" / "processed" / "thailand_proposal.json").unlink()
    with pytest.raises(FileNotFoundError, match="thailand_proposal.json"):
        wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1"})


def test_real_context_without_the_switch_is_refused(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    with pytest.raises(RuntimeError):
        wf.setup_run("real_stage", root=root, env={})


def test_throttled_guard_skips_calls_inside_the_interval():
    now = [0.0]
    calls = []
    guard = wf._ThrottledGuard(lambda: calls.append(now[0]), min_interval=10.0, clock=lambda: now[0])
    for t in (0.0, 1.0, 9.9, 10.0, 12.0, 25.0):
        now[0] = t
        guard()
    assert calls == [0.0, 10.0, 25.0]
    assert guard.calls == 6 and guard.passed == 3


def test_describe_prints_mode_and_paths(sim_env, capsys, keep_sys_path):
    ctx = wf.setup_run("describe", root=ROOT, env=sim_env)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Mode: sim" in out and "Results:" in out and "Headroom" in out


def test_describe_handles_unreadable_headroom(sim_env, capsys, monkeypatch, keep_sys_path):
    from dtt import thermal

    unreadable = {"cpu_percent": None, "mem_available_gb": None, "temperature_c": None, "claude_processes": None}
    monkeypatch.setattr(thermal, "headroom", lambda: unreadable)
    ctx = wf.setup_run("describe", root=ROOT, env=sim_env)
    ctx.describe()
    out = capsys.readouterr().out
    assert "CPU unavailable" in out and "memory available unavailable" in out
    assert "temperature unavailable" in out and "Claude-related processes unavailable" in out


def test_explicit_real_mode_still_needs_the_switch(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    with pytest.raises(RuntimeError, match="DTT_RUN_REAL"):
        wf.setup_run("real_stage", mode="real", root=root, env={})
    with pytest.raises(RuntimeError, match="sim"):
        wf.setup_run("real_stage", mode="real", root=root, env={wf.WORLD_VARIABLE: "sim", wf.REAL_VARIABLE: "1"})


def test_invalid_mode_argument_is_rejected(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        wf.setup_run("stage", mode="other", root=root, env={wf.REAL_VARIABLE: "1"})


@pytest.mark.parametrize("value", ["yes", "true", "2", "on"])
def test_guard_switch_accepts_only_zero_or_one(tmp_path, keep_sys_path, value):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match=wf.GUARD_VARIABLE):
        wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1", wf.GUARD_VARIABLE: value})
    env = {wf.WORLD_VARIABLE: "sim", wf.SIM_DIR_VARIABLE: str(tmp_path / "world"), wf.GUARD_VARIABLE: value}
    with pytest.raises(ValueError, match=wf.GUARD_VARIABLE):
        wf.setup_run("sim_stage", root=root, env=env)
    assert not (tmp_path / "world").exists()


@pytest.mark.parametrize(
    "variable, relative",
    [
        (wf.SIM_DIR_VARIABLE, "data/world"),
        (wf.SIM_DIR_VARIABLE, "data"),
        (wf.SIM_DIR_VARIABLE, "results/world"),
        (wf.SIM_DIR_VARIABLE, "."),
        (wf.SIM_DIR_VARIABLE, ".."),
        (wf.RESULTS_VARIABLE, "results"),
        (wf.RESULTS_VARIABLE, "data/processed"),
        (wf.FIGURES_VARIABLE, "figures/sim"),
    ],
)
def test_simulated_run_refuses_repository_outputs(tmp_path, keep_sys_path, variable, relative):
    root = _fake_root(tmp_path)
    env = {wf.WORLD_VARIABLE: "sim", wf.SIM_DIR_VARIABLE: str(tmp_path / "world"), variable: str(root / relative)}
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="overlaps"):
        wf.setup_run("sim_stage", root=root, env=env)
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert after == before


def test_directory_names_that_only_share_a_prefix_are_allowed(sim_env, tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    env = dict(sim_env)
    env[wf.RESULTS_VARIABLE] = str(root / "results_old")
    env[wf.FIGURES_VARIABLE] = str(root / "figures_old")
    ctx = wf.setup_run("prefix", root=root, env=env)
    assert ctx.results_dir == root / "results_old" and ctx.results_dir.is_dir()
    assert ctx.figures_dir == root / "figures_old" and ctx.figures_dir.is_dir()


def test_figure_level_legend_entries_are_counted(sim_env, keep_sys_path):
    from matplotlib.lines import Line2D

    ctx = wf.setup_run("legends", root=ROOT, env=sim_env)
    fig, (left, right) = plt.subplots(1, 2)
    left.plot([0, 1], [0, 1], label="one")
    right.plot([0, 1], [1, 0], label="two")
    for ax in (left, right):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    left.legend()
    fig.legend(handles=[Line2D([0], [0], label=name) for name in ("A", "B", "C")], loc="lower center")
    ctx.save_figure(fig, "fig_legends.png", show=False)
    inventory = pd.read_csv(ctx.results_dir / "figure_inventory_legends.csv")
    assert inventory.loc[0, "legend entries"] == 4


def test_simulated_target_baseline_is_complete(sim_env, keep_sys_path):
    ctx = wf.setup_run("target", root=ROOT, env=sim_env)
    baseline = pd.read_csv(ctx.baseline_path)
    assert {"intl_tourism_receipts_usd_bn", "gdp_current_usd_bn", "intl_arrivals_million"} <= set(baseline["variable"])
    for variable, group in baseline.groupby("variable"):
        years = sorted(group.loc[group["value"].notna(), "year"].astype(int))
        assert years == list(range(2015, 2025)), variable


def test_seeds_are_copied_for_every_context(sim_env, keep_sys_path):
    first = wf.setup_run("a", root=ROOT, env=sim_env)
    first.seeds["effects"] = 99
    second = wf.setup_run("b", root=ROOT, env=sim_env)
    assert second.seeds["effects"] == 0 and wf.SEEDS["effects"] == 0
    with pytest.raises(TypeError):
        wf.SEEDS["effects"] = 1
