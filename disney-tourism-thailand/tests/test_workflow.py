"""Tests of the run context used by the notebooks (simulated data and stub files only)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dtt import impact, thermal, workflow as wf

ROOT = Path(__file__).resolve().parents[1]
REAL_ENV = {wf.REAL_VARIABLE: "1", wf.GUARD_VARIABLE: "0"}
FLOAT_TEXTS = [
    "0.0034558419206478603",
    "0.0003304370761833871",
    "-0.001303157231604361",
    "0.004463745723640113",
    "0.005811181041963531",
    "-0.00016290994799305277",
    "0.30000000000000004",
]


@pytest.fixture(scope="module")
def sim_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("sim_world")
    return {wf.WORLD_VARIABLE: "sim", wf.SIM_DIR_VARIABLE: str(base / "world")}


@pytest.fixture()
def keep_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


class FakeClock:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture()
def fake_guard(monkeypatch):
    """Build every thermal guard with injected readers, a fake clock and a fake sleep.

    The attributes ``temperature`` and ``cpu`` set what the readers return.  The real ``time.sleep`` raises, so a
    test that waits in real time fails at once.
    """
    state = types.SimpleNamespace(clock=FakeClock(), temperature=None, cpu=10.0, kwargs=None)
    real = thermal.ThermalGuard

    def factory(*args, **kwargs):
        state.kwargs = dict(kwargs)
        kwargs.update(
            read_temp=lambda: state.temperature,
            read_cpu=lambda: state.cpu,
            sleep=state.clock.sleep,
            clock=state.clock,
        )
        return real(*args, **kwargs)

    def refuse(seconds):
        raise AssertionError("time.sleep was called")

    monkeypatch.setattr(thermal, "ThermalGuard", factory)
    monkeypatch.setattr(time, "sleep", refuse)
    return state


def _sim_dirs(sim_env, tmp_path, **extra):
    """Environment of a simulated run that shares the world and writes below ``tmp_path``."""
    env = dict(sim_env)
    env[wf.RESULTS_VARIABLE] = str(tmp_path / "r")
    env[wf.FIGURES_VARIABLE] = str(tmp_path / "f")
    env.update(extra)
    return env


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
    assert ctx.investment_fx_path is None
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


def test_table_json_and_figure_round_trip(sim_env, tmp_path, keep_sys_path):
    ctx = wf.setup_run("roundtrip", root=ROOT, env=_sim_dirs(sim_env, tmp_path))
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


# ----------------------------------------------------------------------------
# Thermal guard
# ----------------------------------------------------------------------------
def test_guard_is_optional_in_a_simulated_run(sim_env, tmp_path, fake_guard, capsys, keep_sys_path):
    fake_guard.temperature = 50.0
    ctx = wf.setup_run("guarded", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    assert callable(ctx.guard) and ctx.guard_object is not None
    ctx.guard()
    log = pd.read_csv(tmp_path / "r" / "thermal_log.csv")
    assert log["action"].tolist() == ["ok"] and log["temperature_c"].tolist() == [50.0]
    assert fake_guard.clock.sleeps == []
    assert "thermal guard" not in capsys.readouterr().out


def test_guard_is_built_with_the_cpu_wait_and_the_printing_switched_on(sim_env, tmp_path, fake_guard, keep_sys_path):
    ctx = wf.setup_run("guarded", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    assert fake_guard.kwargs["max_cpu_wait_seconds"] == wf.GUARD_CPU_WAIT_SECONDS == 300.0
    assert fake_guard.kwargs["verbose"] is True
    assert Path(fake_guard.kwargs["log_path"]) == tmp_path / "r" / "thermal_log.csv"
    assert isinstance(ctx.guard, wf._ThrottledGuard) and ctx.guard._min_interval == wf.GUARD_MIN_INTERVAL_SECONDS == 10.0


def test_guard_waits_for_cpu_load_on_the_fake_clock_and_reports_the_timeout(sim_env, tmp_path, fake_guard, capsys, keep_sys_path):
    fake_guard.cpu = 99.0
    ctx = wf.setup_run("busy", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    ctx.guard()
    assert fake_guard.clock.now == pytest.approx(wf.GUARD_CPU_WAIT_SECONDS)
    assert sum(fake_guard.clock.sleeps) == pytest.approx(wf.GUARD_CPU_WAIT_SECONDS)
    log = pd.read_csv(tmp_path / "r" / "thermal_log.csv")
    assert log["action"].tolist() == ["cpu_timeout"]
    out = capsys.readouterr().out
    assert "thermal guard: waiting" in out and "continuing" in out
    summary = ctx.guard_summary()
    assert summary["cpu_timeouts"] == 1 == ctx.guard_object.summary()["cpu_timeouts"]
    assert summary["active"] is True and summary["calls"] == 1 and summary["passed"] == 1
    assert summary["waits"] == 1 and summary["waited_seconds"] == pytest.approx(wf.GUARD_CPU_WAIT_SECONDS)
    sleeps_before = len(fake_guard.clock.sleeps)
    ctx.guard_object()
    assert len(fake_guard.clock.sleeps) == sleeps_before
    assert pd.read_csv(tmp_path / "r" / "thermal_log.csv")["action"].tolist() == ["cpu_timeout", "ok"]


def test_guard_still_raises_for_heat_on_the_fake_clock(sim_env, tmp_path, fake_guard, keep_sys_path):
    fake_guard.temperature = 95.0
    ctx = wf.setup_run("hot", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    with pytest.raises(thermal.ThermalTimeout):
        ctx.guard()
    assert fake_guard.clock.now == pytest.approx(ctx.guard_object.max_wait_seconds)
    assert ctx.guard_summary()["cpu_timeouts"] == 0


def test_guard_summary_without_a_guard_reports_zero_cpu_timeouts(sim_env, capsys, keep_sys_path):
    ctx = wf.setup_run("noguard", root=ROOT, env=sim_env)
    summary = ctx.guard_summary(show=True)
    assert summary["active"] is False and summary["cpu_timeouts"] == 0 and summary["checks"] == 0
    assert summary["max_temperature_c"] is None and summary["temperature_source"] is None
    assert "Thermal guard: off." in capsys.readouterr().out


def test_guard_summary_line_reports_the_totals(sim_env, tmp_path, fake_guard, capsys, keep_sys_path):
    fake_guard.temperature = 61.5
    ctx = wf.setup_run("line", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    ctx.guard()
    capsys.readouterr()
    summary = ctx.guard_summary(show=True)
    out = capsys.readouterr().out
    assert summary["max_temperature_c"] == 61.5 and summary["temperature_source"] == "custom"
    assert "1 checks, 0 waits" in out and "0 CPU timeouts" in out and "61.5 C (custom)" in out


def test_throttled_guard_skips_calls_inside_the_interval():
    now = [0.0]
    calls = []
    guard = wf._ThrottledGuard(lambda: calls.append(now[0]), min_interval=10.0, clock=lambda: now[0])
    for t in (0.0, 1.0, 9.9, 10.0, 12.0, 25.0):
        now[0] = t
        guard()
    assert calls == [0.0, 10.0, 25.0]
    assert guard.calls == 6 and guard.passed == 3


def _fake_root(tmp_path: Path, name: str = "repo") -> Path:
    root = tmp_path / name
    (root / "src" / "dtt").mkdir(parents=True)
    (root / "src" / "dtt" / "__init__.py").write_text("", encoding="utf-8")
    (root / "data" / "raw" / "wdi").mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "raw" / "wdi" / "country_metadata.csv").write_text("iso3\n", encoding="utf-8")
    (root / "data" / "processed" / "cases_catalogue.csv").write_text("case_id\n", encoding="utf-8")
    (root / "data" / "processed" / "cases_investment_fx.csv").write_text("case_id\n", encoding="utf-8")
    (root / "data" / "processed" / "thailand_baseline.csv").write_text("year,variable,value\n", encoding="utf-8")
    (root / "data" / "processed" / "thailand_proposal.json").write_text(
        json.dumps({"status": "x", "scenarios": [{"name": "s1", "capex_thb_bn": 10, "primary": False}, {"name": "s2", "capex_thb_bn": 20, "primary": True}]}),
        encoding="utf-8",
    )
    return root


def test_real_context_reads_no_data_and_uses_repository_paths(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = wf.setup_run("real_stage", root=root, env=REAL_ENV)
    assert ctx.is_real and ctx.target_iso3 == "THA"
    assert ctx.raw_dir == root / "data" / "raw" / "wdi"
    assert ctx.results_dir == root / "results" and ctx.figures_dir == root / "figures"
    assert ctx.investment_fx_path == root / "data" / "processed" / "cases_investment_fx.csv"
    assert ctx.guard is None
    assert ctx.primary_scenario()["name"] == "s2"


def test_real_context_builds_the_guard_by_default(tmp_path, fake_guard, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1"})
    assert callable(ctx.guard) and ctx.guard_object is not None
    assert Path(fake_guard.kwargs["log_path"]) == root / "results" / "thermal_log.csv"


def test_real_context_reports_missing_inputs(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    (root / "data" / "processed" / "thailand_proposal.json").unlink()
    with pytest.raises(FileNotFoundError, match="thailand_proposal.json"):
        wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1"})


@pytest.mark.parametrize(
    "relative",
    [
        "data/raw/wdi/country_metadata.csv",
        "data/processed/cases_catalogue.csv",
        "data/processed/cases_investment_fx.csv",
        "data/processed/thailand_baseline.csv",
        "data/processed/thailand_proposal.json",
    ],
)
def test_real_context_names_each_missing_input_without_an_absolute_path(tmp_path, keep_sys_path, relative):
    root = _fake_root(tmp_path)
    (root / relative).unlink()
    with pytest.raises(FileNotFoundError) as caught:
        wf.setup_run("real_stage", root=root, env=REAL_ENV)
    message = str(caught.value)
    assert message.startswith("input files missing: ") and relative in message
    assert str(tmp_path) not in message


def test_real_context_requires_the_investment_costs_with_a_clear_message(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    (root / "data" / "processed" / "cases_investment_fx.csv").unlink()
    with pytest.raises(FileNotFoundError) as caught:
        wf.setup_run("real_stage", root=root, env=REAL_ENV)
    message = str(caught.value)
    assert "data/processed/cases_investment_fx.csv" in message
    assert "derived dollar costs" in message and "a real run needs it" in message
    assert not (root / "results").exists()


def test_real_context_lists_every_missing_input_at_once(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    (root / "data" / "processed" / "cases_investment_fx.csv").unlink()
    (root / "data" / "processed" / "thailand_baseline.csv").unlink()
    with pytest.raises(FileNotFoundError) as caught:
        wf.setup_run("real_stage", root=root, env=REAL_ENV)
    assert "thailand_baseline.csv" in str(caught.value) and "cases_investment_fx.csv" in str(caught.value)


def test_real_context_without_the_switch_is_refused(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    with pytest.raises(RuntimeError):
        wf.setup_run("real_stage", root=root, env={})


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


def test_figure_level_legend_entries_are_counted(sim_env, tmp_path, keep_sys_path):
    from matplotlib.lines import Line2D

    ctx = wf.setup_run("legends", root=ROOT, env=_sim_dirs(sim_env, tmp_path))
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


# ----------------------------------------------------------------------------
# describe
# ----------------------------------------------------------------------------
def _readings(temperature_c=None):
    return {
        "cpu_percent": 12.0,
        "mem_available_gb": 3.0,
        "mem_percent": 50.0,
        "temperature_c": temperature_c,
        "claude_processes": {"count": 0, "names": []},
    }


def _thermal_lines(out):
    return [line for line in out.splitlines() if line.startswith("Thermal guard:")]


def _requirements_root(tmp_path, text):
    (tmp_path / "requirements.txt").write_text(text, encoding="utf-8")
    return tmp_path


def test_check_requirements_accepts_the_installed_versions_of_the_repository():
    wf.check_requirements(ROOT)


def test_check_requirements_names_the_package_that_is_too_old(tmp_path):
    root = _requirements_root(tmp_path, "# comment\nnumpy>=99.0.0\npandas>=0.1\nsome-other-package>=1.0\n")
    with pytest.raises(RuntimeError) as caught:
        wf.check_requirements(root)
    message = str(caught.value)
    assert "numpy" in message and "99.0.0" in message and "pandas" not in message
    assert sys.executable in message and "ipykernel install" in message and "kernel_name" in message


def test_check_requirements_names_a_missing_package(tmp_path, monkeypatch):
    import importlib.metadata as md

    real = md.version

    def version(name):
        if name == "statsmodels":
            raise md.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(md, "version", version)
    root = _requirements_root(tmp_path, "statsmodels>=0.15.0\n")
    with pytest.raises(RuntimeError, match="statsmodels is not installed"):
        wf.check_requirements(root)


def test_check_requirements_compares_numeric_releases_not_text(tmp_path, monkeypatch):
    import importlib.metadata as md

    real = md.version
    monkeypatch.setattr(md, "version", lambda name: "10.0.0" if name == "numpy" else real(name))
    wf.check_requirements(_requirements_root(tmp_path, "numpy>=9.9.9\n"))
    monkeypatch.setattr(md, "version", lambda name: "2.10.0rc1" if name == "numpy" else real(name))
    wf.check_requirements(_requirements_root(tmp_path, "numpy>=2.9\n"))
    with pytest.raises(RuntimeError, match="numpy 2.10.0rc1 is older"):
        wf.check_requirements(_requirements_root(tmp_path, "numpy>=2.11\n"))


def test_check_requirements_is_skipped_without_the_file(tmp_path):
    wf.check_requirements(tmp_path)


def test_setup_run_refuses_an_environment_below_requirements(sim_env, tmp_path, monkeypatch, keep_sys_path):
    import importlib.metadata as md

    real = md.version
    monkeypatch.setattr(md, "version", lambda name: "0.0.1" if name == "scikit-learn" else real(name))
    with pytest.raises(RuntimeError, match="scikit-learn 0.0.1 is older"):
        wf.setup_run("requirements", root=ROOT, env=_sim_dirs(sim_env, tmp_path))


def test_describe_prints_mode_and_paths(sim_env, capsys, monkeypatch, keep_sys_path):
    monkeypatch.setattr(thermal, "headroom", lambda: _readings(55.0))
    ctx = wf.setup_run("describe", root=ROOT, env=sim_env)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Mode: sim" in out and "Results:" in out and "Headroom" in out
    assert "temperature 55 C" in out
    assert "Claude" not in out and "process" not in out.split("Headroom")[1].splitlines()[0]


def test_describe_runs_with_the_real_headroom(sim_env, capsys, keep_sys_path):
    ctx = wf.setup_run("describe", root=ROOT, env=sim_env)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Headroom: CPU" in out and len(_thermal_lines(out)) == 1


def test_describe_handles_unreadable_headroom(sim_env, capsys, monkeypatch, keep_sys_path):
    unreadable = {"cpu_percent": None, "mem_available_gb": None, "temperature_c": None, "claude_processes": None}
    monkeypatch.setattr(thermal, "headroom", lambda: unreadable)
    ctx = wf.setup_run("describe", root=ROOT, env=sim_env)
    ctx.describe()
    out = capsys.readouterr().out
    assert "CPU unavailable" in out and "memory available unavailable" in out
    assert "temperature unavailable" in out and "Claude" not in out
    assert _thermal_lines(out) == ["Thermal guard: off; temperature reading: not available on this machine; load reader: none."]


def test_describe_states_in_one_line_that_a_temperature_is_available(sim_env, tmp_path, fake_guard, capsys, monkeypatch, keep_sys_path):
    monkeypatch.setattr(thermal, "headroom", lambda: _readings(61.0))
    ctx = wf.setup_run("describe", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    ctx.describe()
    lines = _thermal_lines(capsys.readouterr().out)
    assert lines == ["Thermal guard: on; temperature reading: available, the guard waits while it exceeds 85 C; load reader: custom function <lambda>."]


def test_describe_states_in_one_line_that_no_temperature_is_available(sim_env, tmp_path, fake_guard, capsys, monkeypatch, keep_sys_path):
    monkeypatch.setattr(thermal, "headroom", lambda: _readings(None))
    ctx = wf.setup_run("describe", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    ctx.describe()
    lines = _thermal_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "temperature reading: not available on this machine" in lines[0]
    assert "waits while the load exceeds 92 percent, for at most 300 seconds per wait" in lines[0]


def test_describe_names_the_default_load_reader_of_the_guard(sim_env, tmp_path, capsys, monkeypatch, keep_sys_path):
    monkeypatch.setattr(thermal, "headroom", lambda: _readings(None))
    ctx = wf.setup_run("describe", root=ROOT, env=_sim_dirs(sim_env, tmp_path, **{wf.GUARD_VARIABLE: "1"}))
    assert type(ctx.guard_object._read_cpu).__name__ in wf._LOAD_READER_NAMES
    ctx.describe()
    lines = _thermal_lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].endswith("load reader: system load without this program and its child processes.")


def test_describe_survives_a_failing_headroom(sim_env, capsys, monkeypatch, keep_sys_path):
    def broken():
        raise RuntimeError("no sensors")

    monkeypatch.setattr(thermal, "headroom", broken)
    ctx = wf.setup_run("describe", root=ROOT, env=sim_env)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Headroom: not available (RuntimeError)" in out
    assert _thermal_lines(out) == ["Thermal guard: off; temperature reading: not checked; load reader: none."]


# ----------------------------------------------------------------------------
# read_table
# ----------------------------------------------------------------------------
def test_read_table_returns_exactly_the_floats_that_were_written(tmp_path, keep_sys_path):
    ctx = wf.setup_run("round_trip", root=_fake_root(tmp_path), env=REAL_ENV)
    ctx.table_path("floats").write_text("x\n" + "\n".join(FLOAT_TEXTS) + "\n", encoding="utf-8")
    back = ctx.read_table("floats")["x"].tolist()
    assert back == [float(text) for text in FLOAT_TEXTS]


def test_saved_floats_survive_a_save_and_read_cycle(tmp_path, keep_sys_path):
    ctx = wf.setup_run("cycle", root=_fake_root(tmp_path), env=REAL_ENV)
    rng = np.random.default_rng(5)
    values = rng.normal(size=2000) * 10.0 ** rng.integers(-4, 5, size=2000)
    ctx.save_table(pd.DataFrame({"x": values}), "values")
    assert np.array_equal(ctx.read_table("values")["x"].to_numpy(), values)


def test_read_table_passes_options_and_respects_an_explicit_float_precision(tmp_path, monkeypatch, keep_sys_path):
    ctx = wf.setup_run("options", root=_fake_root(tmp_path), env=REAL_ENV)
    ctx.save_table(pd.DataFrame({"m": ["a"], "x": [1.5]}), "t")
    seen = []
    real = pd.read_csv

    def spy(path, **kwargs):
        seen.append(dict(kwargs))
        return real(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    ctx.read_table("t", index_col="m")
    ctx.read_table("t", float_precision="high")
    ctx.read_table("t", engine="python")
    assert seen[0] == {"index_col": "m", "float_precision": "round_trip"}
    assert seen[1] == {"float_precision": "high"}
    assert seen[2] == {"engine": "python"}


# ----------------------------------------------------------------------------
# Printed paths
# ----------------------------------------------------------------------------
def _simple_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share (percent)")
    ax.legend()
    return fig


def test_saving_prints_repository_relative_paths_only(tmp_path, capsys, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = wf.setup_run("paths", root=root, env=REAL_ENV)
    ctx.save_table(pd.DataFrame({"a": [1]}), "t_one")
    ctx.save_table(pd.DataFrame({"a": [1]}), "sub/t_two.csv")
    ctx.save_json({"a": 1}, "j_one")
    ctx.save_figure(_simple_figure(), "fig_one.png", show=False)
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "Table saved: results/t_one.csv",
        "Table saved: results/sub/t_two.csv",
        "JSON saved: results/j_one.json",
        "Figure saved: figures/fig_one.png",
    ]
    assert str(tmp_path) not in out


def test_saving_in_a_simulated_run_prints_labels_not_the_output_folders(sim_env, tmp_path, capsys, keep_sys_path):
    ctx = wf.setup_run("paths", root=ROOT, env=_sim_dirs(sim_env, tmp_path))
    ctx.save_table(pd.DataFrame({"a": [1]}), "t_one")
    ctx.save_json({"a": 1}, "j_one")
    ctx.save_figure(_simple_figure(), "fig_one.png", show=False)
    out = capsys.readouterr().out
    assert out.splitlines() == ["Table saved: results/t_one.csv", "JSON saved: results/j_one.json", "Figure saved: figures/fig_one.png"]
    assert str(tmp_path) not in out and str(Path.home()) not in out


def test_describe_prints_no_absolute_path(tmp_path, capsys, monkeypatch, keep_sys_path):
    monkeypatch.setattr(thermal, "headroom", lambda: _readings(None))
    ctx = wf.setup_run("paths", root=_fake_root(tmp_path), env=REAL_ENV)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Raw files:   data/raw/wdi" in out and "Catalogue:   data/processed/cases_catalogue.csv" in out
    assert "Results:     results" in out and "Figures:     figures" in out
    assert "Target:      Thailand (THA)" in out
    assert str(tmp_path) not in out and str(Path.home()) not in out


def test_describe_of_a_simulated_run_prints_no_absolute_path(sim_env, tmp_path, capsys, monkeypatch, keep_sys_path):
    monkeypatch.setattr(thermal, "headroom", lambda: _readings(None))
    ctx = wf.setup_run("paths", root=ROOT, env=_sim_dirs(sim_env, tmp_path))
    ctx.describe()
    out = capsys.readouterr().out
    assert "Raw files:   simulated_world" in out and "Catalogue:   simulated_world/cases_catalogue.csv" in out
    assert "Results:     results" in out and "Figures:     figures" in out
    target_line = next(line for line in out.splitlines() if line.startswith("Target:"))
    assert target_line == f"Target:      {ctx.target_iso3} (simulated)"
    assert str(tmp_path) not in out and sim_env[wf.SIM_DIR_VARIABLE] not in out and str(Path.home()) not in out


def test_the_representation_of_the_context_holds_no_path(tmp_path, keep_sys_path):
    ctx = wf.setup_run("repr_stage", root=_fake_root(tmp_path), env=REAL_ENV)
    assert repr(ctx) == "RunContext(stage='repr_stage', mode='real')"
    assert str(tmp_path) not in repr(ctx)


def test_require_and_read_errors_name_labels_not_absolute_paths(tmp_path, keep_sys_path):
    ctx = wf.setup_run("errors", root=_fake_root(tmp_path), env=REAL_ENV)
    with pytest.raises(FileNotFoundError) as required:
        ctx.require("absent_table", "absent_json.json")
    assert "results/absent_table.csv, results/absent_json.json" in str(required.value)
    with pytest.raises(FileNotFoundError) as table:
        ctx.read_table("absent_table")
    with pytest.raises(FileNotFoundError) as data:
        ctx.read_json("absent_json")
    for error in (required, table, data):
        assert str(tmp_path) not in str(error.value)


# ----------------------------------------------------------------------------
# Figure inventory
# ----------------------------------------------------------------------------
def test_saving_a_figure_again_replaces_its_inventory_row(tmp_path, keep_sys_path):
    ctx = wf.setup_run("rerun", root=_fake_root(tmp_path), env=REAL_ENV)
    ctx.save_figure(_simple_figure(), "fig_a.png", show=False)
    ctx.save_figure(_simple_figure(), "fig_b.png", show=False)
    fig, (left, right) = plt.subplots(1, 2)
    left.set_xlabel("Changed")
    ctx.save_figure(fig, "fig_a.png", show=False)
    ctx.save_figure(_simple_figure(), "fig_a.png", show=False)
    ctx.save_figure(_simple_figure(), "fig_a.png", show=False)
    inventory = pd.read_csv(ctx.results_dir / "figure_inventory_rerun.csv")
    assert inventory["file"].tolist() == ["fig_a.png", "fig_b.png"]
    assert inventory.loc[0, "panels"] == 1 and inventory.loc[0, "x labels"] == "Year"
    assert [row["file"] for row in ctx.figure_log] == ["fig_a.png", "fig_b.png"]


def test_a_changed_figure_replaces_the_row_in_place(tmp_path, keep_sys_path):
    ctx = wf.setup_run("replace", root=_fake_root(tmp_path), env=REAL_ENV)
    ctx.save_figure(_simple_figure(), "fig_a.png", show=False)
    ctx.save_figure(_simple_figure(), "fig_b.png", show=False)
    fig, axes = plt.subplots(1, 3)
    axes[0].set_xlabel("Three panels")
    ctx.save_figure(fig, "fig_a.png", show=False)
    inventory = pd.read_csv(ctx.results_dir / "figure_inventory_replace.csv")
    assert inventory["file"].tolist() == ["fig_a.png", "fig_b.png"]
    assert inventory.loc[0, "panels"] == 3 and inventory.loc[0, "x labels"] == "Three panels"
    assert inventory.loc[1, "panels"] == 1


# ----------------------------------------------------------------------------
# Stamps
# ----------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamps(ctx) -> dict:
    return json.loads((ctx.results_dir / wf.STAMP_FILE).read_text(encoding="utf-8"))["files"]


def _stage(root, name, reads=(), tables=None, env=None):
    """Run one stage of a small chain: require files, then save the tables given as name to DataFrame."""
    ctx = wf.setup_run(name, root=root, env=REAL_ENV if env is None else env)
    if reads:
        ctx.require(*reads)
    for table, frame in (tables or {}).items():
        ctx.save_table(frame, table)
    return ctx


def _frame(value):
    return pd.DataFrame({"a": [1.0, 2.0, value]})


def _chain(root, effects=3.0):
    """Stages 02 (effects), 03 (weights from effects) and 05 (impact from weights and effects)."""
    _stage(root, "02_effects", tables={"case_effects": _frame(effects)})
    _stage(root, "03_weights", reads=("case_effects",), tables={"weights": _frame(effects + 1)})
    _stage(root, "05_impact", reads=("weights", "case_effects"), tables={"impact": _frame(effects + 2)})


def test_every_saved_file_gets_a_stamp_with_digest_stage_and_inputs(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _stage(root, "02_effects", tables={"case_effects": _frame(3.0)})
    ctx = _stage(root, "03_weights", reads=("case_effects",))
    ctx.save_table(_frame(4.0), "weights")
    ctx.save_json({"n": 1}, "summary")
    ctx.save_figure(_simple_figure(), "fig.png", show=False)
    stamps = _stamps(ctx)
    effects_digest = _sha(ctx.results_dir / "case_effects.csv")
    assert stamps["results/case_effects.csv"] == {"sha256": effects_digest, "stage": "02_effects", "inputs": {}}
    inputs = {"results/case_effects.csv": effects_digest}
    assert stamps["results/weights.csv"] == {"sha256": _sha(ctx.results_dir / "weights.csv"), "stage": "03_weights", "inputs": inputs}
    assert stamps["results/summary.json"]["sha256"] == _sha(ctx.results_dir / "summary.json")
    assert stamps["results/summary.json"]["inputs"] == inputs
    assert stamps["figures/fig.png"] == {"sha256": _sha(ctx.figures_dir / "fig.png"), "stage": "03_weights", "inputs": inputs}
    inventory = "results/figure_inventory_03_weights.csv"
    assert stamps[inventory]["sha256"] == _sha(ctx.results_dir / "figure_inventory_03_weights.csv")
    assert ctx.inputs_registered == inputs


def test_stamp_file_is_readable_sorted_json_without_absolute_paths_or_times(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    text = (root / "results" / wf.STAMP_FILE).read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(root) not in text and "\\" not in text
    data = json.loads(text)
    assert data["format"] == 1 and list(data["files"]) == sorted(data["files"])
    assert text == json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert b"\r" not in (root / "results" / wf.STAMP_FILE).read_bytes()
    assert sorted(p.name for p in (root / "results").iterdir()) == ["case_effects.csv", wf.STAMP_FILE, "impact.csv", "weights.csv"]


def test_stamps_do_not_depend_on_where_the_results_directory_is(tmp_path, keep_sys_path):
    first = _fake_root(tmp_path / "a", "repo")
    second = _fake_root(tmp_path / "b" / "deeper" / "still", "other_name")
    for root in (first, second):
        _chain(root)
        ctx = _stage(root, "03_weights", tables={"x": _frame(1.0)})
        ctx.save_figure(_simple_figure(), "fig.png", show=False)
    assert (first / "results" / wf.STAMP_FILE).read_bytes() == (second / "results" / wf.STAMP_FILE).read_bytes()
    elsewhere = tmp_path / "elsewhere"
    env = {wf.REAL_VARIABLE: "1", wf.GUARD_VARIABLE: "0", wf.RESULTS_VARIABLE: str(elsewhere / "r"), wf.FIGURES_VARIABLE: str(elsewhere / "f")}
    _stage(first, "02_effects", tables={"case_effects": _frame(3.0)}, env=env)
    moved = json.loads((elsewhere / "r" / wf.STAMP_FILE).read_text(encoding="utf-8"))["files"]
    original = json.loads((first / "results" / wf.STAMP_FILE).read_text(encoding="utf-8"))["files"]
    assert moved["results/case_effects.csv"] == original["results/case_effects.csv"]


def test_a_repeated_run_writes_identical_stamps(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    before = (root / "results" / wf.STAMP_FILE).read_bytes()
    _chain(root)
    assert (root / "results" / wf.STAMP_FILE).read_bytes() == before


def test_the_unchanged_chain_passes_and_registers_the_inputs(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    ctx = wf.setup_run("report", root=root, env=REAL_ENV)
    ctx.require("impact", "weights.csv", "case_effects")
    assert sorted(ctx.inputs_registered) == ["results/case_effects.csv", "results/impact.csv", "results/weights.csv"]
    for label, digest in ctx.inputs_registered.items():
        assert digest == _sha(root / label)


def test_editing_a_table_by_hand_is_caught_by_the_stage_that_reads_it(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    table = root / "results" / "case_effects.csv"
    table.write_text(table.read_text(encoding="utf-8").replace("3.0", "3.1"), encoding="utf-8")
    ctx = wf.setup_run("05_impact", root=root, env=REAL_ENV)
    with pytest.raises(wf.StaleResultError) as caught:
        ctx.require("weights", "case_effects")
    message = str(caught.value)
    assert "results/case_effects.csv was written by stage 02_effects and its content has changed since" in message
    assert "results/weights.csv (stage 03_weights) was computed from" not in message
    assert "02_effects" in message.splitlines()[-1]
    assert str(tmp_path) not in message
    assert ctx.inputs_registered == {}
    assert (root / "results" / "impact.csv").is_file()


def test_editing_a_table_by_hand_is_caught_through_the_recorded_inputs(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    table = root / "results" / "case_effects.csv"
    table.write_text(table.read_text(encoding="utf-8").replace("3.0", "3.1"), encoding="utf-8")
    ctx = wf.setup_run("05_impact", root=root, env=REAL_ENV)
    with pytest.raises(wf.StaleResultError, match="results/case_effects.csv was written by stage 02_effects") as caught:
        ctx.require("weights")
    assert "03_weights" not in caught.value.args[0].splitlines()[-1]
    ctx.save_table(_frame(9.0), "late")
    assert (root / "results" / "late.csv").is_file()


def test_running_only_the_last_stage_again_after_a_hand_edit_raises_before_any_output(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    impact_before = (root / "results" / "impact.csv").read_bytes()
    stamps_before = (root / "results" / wf.STAMP_FILE).read_bytes()
    table = root / "results" / "case_effects.csv"
    table.write_text(table.read_text(encoding="utf-8") + "9.0\n", encoding="utf-8")
    with pytest.raises(wf.StaleResultError, match="case_effects.csv"):
        _stage(root, "05_impact", reads=("weights", "case_effects"), tables={"impact": _frame(0.0)})
    assert (root / "results" / "impact.csv").read_bytes() == impact_before
    assert (root / "results" / wf.STAMP_FILE).read_bytes() == stamps_before


def test_running_the_producer_again_restores_the_chain(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    table = root / "results" / "case_effects.csv"
    original = table.read_bytes()
    table.write_bytes(original + b"9.0\n")
    with pytest.raises(wf.StaleResultError):
        _stage(root, "05_impact", reads=("weights",))
    _stage(root, "02_effects", tables={"case_effects": _frame(3.0)})
    assert table.read_bytes() == original
    _stage(root, "05_impact", reads=("weights", "case_effects"))


def test_a_producer_that_wrote_new_content_makes_its_consumers_stale(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    _stage(root, "02_effects", tables={"case_effects": _frame(7.0)})
    ctx = wf.setup_run("05_impact", root=root, env=REAL_ENV)
    with pytest.raises(wf.StaleResultError) as caught:
        ctx.require("weights")
    message = str(caught.value)
    assert "results/weights.csv (stage 03_weights) was computed from an earlier version of results/case_effects.csv" in message
    assert "which stage 02_effects has written again since" in message
    assert message.splitlines()[-1].endswith("03_weights.")
    with pytest.raises(wf.StaleResultError, match="weights.csv"):
        ctx.require("weights", "impact")
    _stage(root, "03_weights", reads=("case_effects",), tables={"weights": _frame(8.0)})
    ctx.require("weights")
    with pytest.raises(wf.StaleResultError, match=r"results/impact.csv \(stage 05_impact\) was computed from an earlier version of results/weights.csv"):
        ctx.require("impact")


def test_a_rewrite_with_equal_content_keeps_the_chain_valid(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    _stage(root, "02_effects", tables={"case_effects": _frame(3.0)})
    _stage(root, "05_impact", reads=("weights", "case_effects"))


def test_a_removed_input_is_reported_with_its_producer(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    (root / "results" / "case_effects.csv").unlink()
    ctx = wf.setup_run("05_impact", root=root, env=REAL_ENV)
    with pytest.raises(wf.StaleResultError) as caught:
        ctx.require("weights")
    assert "results/case_effects.csv was written by stage 02_effects and no longer exists" in str(caught.value)
    with pytest.raises(FileNotFoundError, match="results/case_effects.csv"):
        ctx.require("case_effects")


def test_files_without_a_stamp_are_accepted_and_their_changes_are_still_caught(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    raw = root / "results" / "outside_table.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("a\n1\n", encoding="utf-8")
    ctx = _stage(root, "02_effects", reads=("outside_table",), tables={"derived": _frame(1.0)})
    assert _stamps(ctx)["results/derived.csv"]["inputs"] == {"results/outside_table.csv": _sha(raw)}
    assert "results/outside_table.csv" not in _stamps(ctx)
    _stage(root, "03_weights", reads=("derived", "outside_table"))
    raw.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(wf.StaleResultError, match=r"results/derived.csv \(stage 02_effects\) was computed from results/outside_table.csv, which has changed since"):
        _stage(root, "03_weights", reads=("derived",))


def test_a_removed_file_without_a_stamp_is_reported_by_the_file_computed_from_it(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    raw = root / "results" / "outside_table.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("a\n1\n", encoding="utf-8")
    _stage(root, "02_effects", reads=("outside_table",), tables={"derived": _frame(1.0)})
    raw.unlink()
    pattern = r"results/derived.csv \(stage 02_effects\) was computed from results/outside_table.csv, which no longer exists"
    with pytest.raises(wf.StaleResultError, match=pattern) as caught:
        _stage(root, "03_weights", reads=("derived",))
    assert caught.value.args[0].splitlines()[-1].endswith("02_effects.")


def test_a_raw_data_file_of_the_repository_is_an_input_with_a_repository_label(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    metadata = root / "data" / "raw" / "wdi" / "country_metadata.csv"
    ctx = wf.setup_run("00_panel", root=root, env=REAL_ENV)
    ctx.require(metadata)
    ctx.save_table(_frame(1.0), "from_raw")
    label = "data/raw/wdi/country_metadata.csv"
    assert _stamps(ctx)["results/from_raw.csv"]["inputs"] == {label: _sha(metadata)}
    metadata.write_text("iso3\nTHA\n", encoding="utf-8")
    with pytest.raises(wf.StaleResultError, match=r"was computed from data/raw/wdi/country_metadata.csv, which has changed since"):
        _stage(root, "02_effects", reads=("from_raw",))


def test_stamp_file_stamps_a_file_written_elsewhere(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = wf.setup_run("00_panel", root=root, env=REAL_ENV)
    panel = ctx.processed_dir / "global_panel.csv"
    panel.write_text("iso3,year\nTHA,2019\n", encoding="utf-8")
    assert ctx.stamp_file(panel) == panel
    assert _stamps(ctx)["data/processed/global_panel.csv"] == {"sha256": _sha(panel), "stage": "00_panel", "inputs": {}}
    later = _stage(root, "02_effects", reads=(panel,), tables={"derived": _frame(1.0)})
    assert later.inputs_registered == {"data/processed/global_panel.csv": _sha(panel)}
    panel.write_text("iso3,year\nTHA,2018\n", encoding="utf-8")
    with pytest.raises(wf.StaleResultError, match=r"data/processed/global_panel.csv was written by stage 00_panel and its content has changed"):
        _stage(root, "03_weights", reads=("derived",))
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        ctx.stamp_file(ctx.processed_dir / "absent.csv")
    (tmp_path / "outside.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        ctx.stamp_file(tmp_path / "outside.csv")


def test_a_damaged_stamp_file_stops_the_run_with_a_clear_message(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _chain(root)
    stamp = root / "results" / wf.STAMP_FILE
    for damaged in ("not json", json.dumps({"format": 2, "files": {}}), json.dumps({"format": 1, "files": {"x": {"sha256": 1}}}), "[]"):
        stamp.write_text(damaged, encoding="utf-8")
        ctx = wf.setup_run("05_impact", root=root, env=REAL_ENV)
        with pytest.raises(ValueError, match=r"the stamp file results/file_stamps.json cannot be used"):
            ctx.require("weights")
        with pytest.raises(ValueError, match="stamp file"):
            ctx.save_table(_frame(0.0), "never")
        with pytest.raises(ValueError, match="stamp file"):
            ctx.save_json({"a": 1}, "never")
        with pytest.raises(ValueError, match="stamp file"):
            ctx.save_figure(_simple_figure(), "never.png", show=False)
        assert not (root / "results" / "never.csv").exists() and not (root / "results" / "never.json").exists()
        assert not (root / "figures" / "never.png").exists()
    stamp.unlink()
    wf.setup_run("05_impact", root=root, env=REAL_ENV).require("weights")


def test_saved_files_use_line_feeds_and_leave_no_temporary_files(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = wf.setup_run("lf", root=root, env=REAL_ENV)
    ctx.save_table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), "t")
    ctx.save_json({"a": [1, 2]}, "j")
    assert b"\r" not in (ctx.results_dir / "t.csv").read_bytes()
    assert b"\r" not in (ctx.results_dir / "j.json").read_bytes()
    assert (ctx.results_dir / "t.csv").read_bytes() == b"a,b\n1,x\n2,y\n"
    assert not [p for p in ctx.results_dir.iterdir() if p.name.endswith(".tmp")]


def test_the_stamp_file_is_replaced_again_after_a_short_denial(tmp_path, monkeypatch):
    target = tmp_path / "stamps.json"
    attempts = []
    replace = os.replace

    def busy_twice(source, destination):
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError("the file is in use")
        return replace(source, destination)

    monkeypatch.setattr(wf.os, "replace", busy_twice)
    monkeypatch.setattr(wf.time, "sleep", lambda seconds: None)
    wf._write_text_atomic(target, "content\n")
    assert len(attempts) == 3 and target.read_bytes() == b"content\n"
    assert [p.name for p in tmp_path.iterdir()] == ["stamps.json"]


def test_a_lasting_denial_raises_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    attempts = []

    def always_busy(source, destination):
        attempts.append(source)
        raise PermissionError("the file is in use")

    monkeypatch.setattr(wf.os, "replace", always_busy)
    monkeypatch.setattr(wf.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        wf._write_text_atomic(tmp_path / "stamps.json", "content\n")
    assert len(attempts) == 5 and list(tmp_path.iterdir()) == []


def test_stamps_are_written_in_a_simulated_run_too(sim_env, tmp_path, keep_sys_path):
    env = _sim_dirs(sim_env, tmp_path)
    first = wf.setup_run("02_effects", root=ROOT, env=env)
    first.save_table(_frame(3.0), "case_effects")
    first.save_figure(_simple_figure(), "fig.png", show=False)
    second = wf.setup_run("03_weights", root=ROOT, env=env)
    second.require("case_effects")
    second.save_json({"n": 1}, "weights")
    stamps = _stamps(second)
    assert sorted(stamps) == ["figures/fig.png", "results/case_effects.csv", "results/figure_inventory_02_effects.csv", "results/weights.json"]
    assert stamps["results/weights.json"]["inputs"] == {"results/case_effects.csv": _sha(first.results_dir / "case_effects.csv")}
    text = (tmp_path / "r" / wf.STAMP_FILE).read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    table = tmp_path / "r" / "case_effects.csv"
    table.write_text(table.read_text(encoding="utf-8") + "9.0\n", encoding="utf-8")
    with pytest.raises(wf.StaleResultError, match="results/case_effects.csv was written by stage 02_effects"):
        wf.setup_run("05_impact", root=ROOT, env=env).require("weights.json")
