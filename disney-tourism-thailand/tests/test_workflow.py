"""Tests of the run context used by the notebooks, on temporary files and small fake layouts only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dtt import impact, thermal, workflow as wf

ROOT = Path(__file__).resolve().parents[1]
IDLE_HEADROOM = {"cpu_percent": 5.0, "mem_available_gb": 4.0, "temperature_c": None, "claude_processes": {"count": 0}}
#: Readers that replace the sensors of the thermal guard, so that no test depends on the host load.
FAKE_READERS = {"read_temp": lambda: None, "read_cpu": lambda: 5.0, "sleep": lambda seconds: None}
REAL_ENV = {wf.REAL_VARIABLE: "1", wf.GUARD_VARIABLE: "0"}


@pytest.fixture()
def run_env(tmp_path):
    """Environment of a run on the dataset whose tables and figures go to a directory of the test."""
    return {
        wf.REAL_VARIABLE: "1",
        wf.GUARD_VARIABLE: "0",
        wf.RESULTS_VARIABLE: str(tmp_path / "out" / "results"),
        wf.FIGURES_VARIABLE: str(tmp_path / "out" / "figures"),
    }


@pytest.fixture()
def keep_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


@pytest.fixture(autouse=True)
def idle_machine(monkeypatch):
    """Report an idle machine to :meth:`RunContext.describe`, so that no test reads the host load."""
    monkeypatch.setattr(thermal, "headroom", lambda: dict(IDLE_HEADROOM))


def _fake_root(tmp_path: Path) -> Path:
    """Create a small layout with the files that a real run reads, in a temporary directory."""
    root = tmp_path / "repo"
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


def _stage(name: str, root: Path, env: dict[str, str]) -> wf.RunContext:
    """Create the context of a stage with the fake sensor readers."""
    return wf.setup_run(name, root=root, env=env, guard_options=FAKE_READERS)


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame({"value": list(values)})


def _figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of GDP (percent)")
    ax.legend()
    return fig


# ----------------------------------------------------------------------------
# Modes and roots
# ----------------------------------------------------------------------------
def test_find_root_from_a_subdirectory():
    assert wf.find_root(ROOT / "tests") == ROOT


def test_find_root_fails_outside_a_repository(tmp_path):
    with pytest.raises(FileNotFoundError):
        wf.find_root(tmp_path)


def test_resolve_mode_returns_real_when_the_switch_is_set():
    assert wf.resolve_mode({wf.REAL_VARIABLE: "1"}) == "real"


@pytest.mark.parametrize(
    "env",
    [
        {},
        {wf.REAL_VARIABLE: "0"},
        {wf.REAL_VARIABLE: "true"},
        {wf.REAL_VARIABLE: "yes"},
        {wf.REAL_VARIABLE: "2"},
        {wf.REAL_VARIABLE: "on"},
    ],
)
def test_real_mode_needs_the_explicit_switch(env):
    with pytest.raises(RuntimeError, match="DTT_RUN_REAL"):
        wf.resolve_mode(env)


def test_results_directory_can_be_overridden(tmp_path, keep_sys_path):
    env = dict(REAL_ENV)
    env[wf.RESULTS_VARIABLE] = str(tmp_path / "out" / "r")
    env[wf.FIGURES_VARIABLE] = str(tmp_path / "out" / "f")
    ctx = wf.setup_run("override", root=_fake_root(tmp_path), env=env)
    assert ctx.results_dir == tmp_path / "out" / "r" and ctx.results_dir.is_dir()
    assert ctx.figures_dir == tmp_path / "out" / "f" and ctx.figures_dir.is_dir()


def test_guard_uses_the_readers_it_is_given(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    readers = {"read_temp": lambda: 50.0, "read_cpu": lambda: 1.0, "sleep": lambda seconds: pytest.fail("no wait expected")}
    ctx = wf.setup_run("stage_guard", root=root, env={wf.REAL_VARIABLE: "1"}, guard_options=readers)
    assert callable(ctx.guard) and isinstance(ctx.guard_object, thermal.ThermalGuard)
    ctx.guard()
    log = ctx.results_dir / "thermal_log.csv"
    assert log.is_file() and len(pd.read_csv(log)) == 1


def test_real_context_reads_no_data_and_uses_repository_paths(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1", wf.GUARD_VARIABLE: "0"})
    assert ctx.is_real and ctx.target_iso3 == "THA"
    assert ctx.raw_dir == root / "data" / "raw" / "wdi"
    assert ctx.results_dir == root / "results" and ctx.figures_dir == root / "figures"
    assert ctx.investment_fx_path == root / "data" / "processed" / "cases_investment_fx.csv"
    assert ctx.guard is None
    assert ctx.primary_scenario()["name"] == "s2"


def test_real_context_builds_the_guard_by_default(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1"}, guard_options=FAKE_READERS)
    assert callable(ctx.guard) and ctx.guard_object is not None


def test_real_context_reports_missing_inputs(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    (root / "data" / "processed" / "thailand_proposal.json").unlink()
    with pytest.raises(FileNotFoundError, match="thailand_proposal.json"):
        wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1"})


def test_real_context_requires_the_investment_file(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    (root / "data" / "processed" / "cases_investment_fx.csv").unlink()
    with pytest.raises(FileNotFoundError) as info:
        wf.setup_run("real_stage", root=root, env=REAL_ENV)
    message = str(info.value)
    assert "data/processed/cases_investment_fx.csv" in message
    assert "from 28 to 12 of 44" in message
    assert str(root) not in message


def test_real_context_without_the_switch_is_refused(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    with pytest.raises(RuntimeError):
        wf.setup_run("real_stage", root=root, env={})


def test_throttled_guard_skips_calls_inside_the_interval():
    moment = [0.0]
    calls = []
    guard = wf._ThrottledGuard(lambda: calls.append(moment[0]), min_interval=10.0, clock=lambda: moment[0])
    for t in (0.0, 1.0, 9.9, 10.0, 12.0, 25.0):
        moment[0] = t
        guard()
    assert calls == [0.0, 10.0, 25.0]
    assert guard.calls == 6 and guard.passed == 3


def test_describe_prints_mode_and_paths(tmp_path, capsys, keep_sys_path):
    ctx = _stage("describe", _fake_root(tmp_path), REAL_ENV)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Stage: describe" in out and "Results:" in out and "Headroom" in out


def test_describe_handles_unreadable_headroom(tmp_path, capsys, monkeypatch, keep_sys_path):
    unreadable = {"cpu_percent": None, "mem_available_gb": None, "temperature_c": None, "claude_processes": None}
    monkeypatch.setattr(thermal, "headroom", lambda: unreadable)
    ctx = _stage("describe", _fake_root(tmp_path), REAL_ENV)
    ctx.describe()
    out = capsys.readouterr().out
    assert "CPU unavailable" in out and "memory available unavailable" in out
    assert "temperature unavailable" in out and "Claude-related processes unavailable" in out


def test_explicit_real_mode_still_needs_the_switch(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    with pytest.raises(RuntimeError, match="DTT_RUN_REAL"):
        wf.setup_run("real_stage", mode="real", root=root, env={})


def test_invalid_mode_argument_is_rejected(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        wf.setup_run("stage", mode="other", root=root, env={wf.REAL_VARIABLE: "1"})


@pytest.mark.parametrize("stage", ["", "../up", "a/b", "-start", "has space"])
def test_stage_names_are_plain_names(tmp_path, keep_sys_path, stage):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match="stage"):
        wf.setup_run(stage, root=root, env=REAL_ENV)


@pytest.mark.parametrize("value", ["yes", "true", "2", "on"])
def test_guard_switch_accepts_only_zero_or_one(tmp_path, keep_sys_path, value):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match=wf.GUARD_VARIABLE):
        wf.setup_run("real_stage", root=root, env={wf.REAL_VARIABLE: "1", wf.GUARD_VARIABLE: value})


def test_directory_names_that_only_share_a_prefix_are_allowed(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    env = dict(REAL_ENV)
    env[wf.RESULTS_VARIABLE] = str(root / "results_old")
    env[wf.FIGURES_VARIABLE] = str(root / "figures_old")
    ctx = wf.setup_run("prefix", root=root, env=env)
    assert ctx.results_dir == root / "results_old" and ctx.results_dir.is_dir()
    assert ctx.figures_dir == root / "figures_old" and ctx.figures_dir.is_dir()


def test_seeds_are_copied_for_every_context(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    first = _stage("a", root, REAL_ENV)
    first.seeds["effects"] = 99
    second = _stage("b", root, REAL_ENV)
    assert second.seeds["effects"] == 0 and wf.SEEDS["effects"] == 0
    with pytest.raises(TypeError):
        wf.SEEDS["effects"] = 1


# ----------------------------------------------------------------------------
# Tables, JSON files, figures and printed paths
# ----------------------------------------------------------------------------
def test_table_json_and_figure_round_trip(run_env, tmp_path, keep_sys_path):
    ctx = _stage("roundtrip", _fake_root(tmp_path), run_env)
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

    fpath = ctx.save_figure(_figure(), "fig_example.png", show=False)
    assert fpath.is_file() and fpath.stat().st_size > 1000
    inventory = pd.read_csv(ctx.results_dir / "figure_inventory_roundtrip.csv")
    assert inventory.loc[0, "x labels"] == "Year"
    assert inventory.loc[0, "y labels"] == "Share of GDP (percent)"
    assert inventory.loc[0, "legend entries"] == 1


def test_read_table_returns_floats_bit_for_bit(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_float", root, REAL_ENV)
    values = np.array([0.1 + 0.2, 1.0 / 3.0, 1e-300, 2.0**-1074, 123456.789012345678, np.nextafter(1.0, 2.0)])
    ctx.save_table(pd.DataFrame({"v": values}), "t_float")
    back = ctx.read_table("t_float")
    assert back["v"].to_numpy().tobytes() == values.tobytes()
    explicit = ctx.read_table("t_float", float_precision="high")
    assert explicit["v"].to_numpy().shape == values.shape


def test_saving_a_figure_again_leaves_one_inventory_row(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_fig", root, REAL_ENV)
    for _ in range(2):
        ctx.save_figure(_figure(), "fig_a.png", show=False)
    ctx.save_figure(_figure(), "fig_b.png", show=False)
    inventory = pd.read_csv(ctx.results_dir / "figure_inventory_stage_fig.csv")
    assert inventory["file"].tolist() == ["fig_a.png", "fig_b.png"]


def test_printed_paths_are_relative_in_a_repository_run(tmp_path, capsys, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_print", root, REAL_ENV)
    ctx.save_table(_frame([1.0]), "t_print")
    ctx.save_json({"x": 1}, "j_print")
    ctx.save_figure(_figure(), "f_print.png", show=False)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Table saved: results/t_print.csv" in out
    assert "JSON saved: results/j_print.json" in out
    assert "Figure saved: figures/f_print.png" in out
    assert "Results:     results" in out
    assert str(tmp_path) not in out


def test_printed_paths_are_relative_to_the_run_outside_the_repository(run_env, tmp_path, capsys, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_outside", root, run_env)
    ctx.save_table(_frame([1.0]), "t_outside")
    ctx.save_figure(_figure(), "f_outside.png", show=False)
    ctx.describe()
    out = capsys.readouterr().out
    assert "Table saved: t_outside.csv" in out
    assert "Figure saved: f_outside.png" in out
    assert "outside the repository" in out
    for location in (run_env[wf.RESULTS_VARIABLE], run_env[wf.FIGURES_VARIABLE], str(root)):
        assert location not in out


def test_display_path_names_files_from_their_directory(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_disp", root, REAL_ENV)
    assert ctx.display_path(root / "results" / "t.csv") == "results/t.csv"
    assert ctx.display_path(root / "src" / "dtt" / "workflow.py") == "src/dtt/workflow.py"
    assert ctx.display_path(tmp_path / "elsewhere" / "x.csv") == "x.csv"
    with pytest.raises(ValueError):
        ctx.stamp_key(tmp_path / "elsewhere" / "x.csv")


# ----------------------------------------------------------------------------
# Stamps
# ----------------------------------------------------------------------------
def test_digest_helpers(tmp_path):
    file = tmp_path / "x.bin"
    file.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    assert wf.file_digest(file) == expected
    assert wf.short_digest(expected) == expected[:12]
    assert wf.short_digest(None) == "none"
    assert issubclass(wf.StampError, ValueError)


def test_read_stamp_returns_none_when_absent_and_rejects_a_broken_file(tmp_path):
    assert wf.read_stamp(tmp_path, "absent") is None
    (tmp_path / wf.STAMP_DIRECTORY).mkdir()
    (tmp_path / wf.STAMP_DIRECTORY / "broken.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid stamp"):
        wf.read_stamp(tmp_path, "broken")


def test_a_saved_table_is_listed_as_an_output_with_its_digest(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_out", root, REAL_ENV)
    path = ctx.save_table(_frame([1.0, 2.0]), "t_out")
    stamp = wf.read_stamp(ctx.results_dir, "stage_out")
    assert stamp["format"] == wf.STAMP_FORMAT and stamp["stage"] == "stage_out"
    assert stamp["outputs"]["results/t_out.csv"]["sha256"] == wf.file_digest(path)
    assert stamp["inputs"] == {}
    assert wf.stamp_path(ctx.results_dir, "stage_out") == ctx.results_dir / "stamps" / "stage_out.json"


def test_reading_a_table_records_its_producer(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    producer = _stage("stage_a", root, REAL_ENV)
    producer.save_table(_frame([1.0]), "a")
    consumer = _stage("stage_b", root, REAL_ENV)
    consumer.read_table("a")
    stamp = wf.read_stamp(consumer.results_dir, "stage_b")
    assert stamp["inputs"]["results/a.csv"]["producer"] == "stage_a"
    assert stamp["inputs"]["results/a.csv"]["sha256"] == wf.read_stamp(producer.results_dir, "stage_a")["outputs"]["results/a.csv"]["sha256"]


def test_three_chained_stages_detect_an_edit_at_the_start(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _stage("stage_a", root, REAL_ENV).save_table(_frame([1.0]), "a")
    b = _stage("stage_b", root, REAL_ENV)
    b.read_table("a")
    b.save_table(_frame([2.0]), "b")
    _stage("stage_c", root, REAL_ENV).read_table("b")

    edited = root / "results" / "a.csv"
    edited.write_text("value\n9.0\n", encoding="utf-8")
    with pytest.raises(wf.StampError) as info:
        _stage("stage_c", root, REAL_ENV).read_table("b")
    error = info.value
    assert isinstance(error, ValueError)
    assert error.kind == "changed" and error.path == "results/a.csv" and error.stage == "stage_a"
    assert wf.short_digest(error.expected) in str(error) and wf.short_digest(error.found) in str(error)
    assert "stage_a" in str(error) and "results/a.csv" in str(error)


def test_a_consumer_that_read_an_older_version_is_detected(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _stage("stage_a", root, REAL_ENV).save_table(_frame([1.0]), "a")
    b = _stage("stage_b", root, REAL_ENV)
    b.read_table("a")
    b.save_table(_frame([2.0]), "b")
    _stage("stage_a", root, REAL_ENV).save_table(_frame([3.0]), "a")
    with pytest.raises(wf.StampError) as info:
        _stage("stage_c", root, REAL_ENV).read_table("b")
    assert info.value.kind == "stale"
    assert info.value.path == "results/a.csv" and info.value.stage == "stage_a" and info.value.consumer == "stage_b"


def test_an_output_edited_after_its_producer_ran_is_refused(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    producer = _stage("stage_a", root, REAL_ENV)
    producer.save_json({"x": 1}, "j")
    (producer.results_dir / "j.json").write_text('{"x": 2}', encoding="utf-8")
    with pytest.raises(wf.StampError, match="stage_a") as info:
        _stage("stage_b", root, REAL_ENV).read_json("j")
    assert info.value.kind == "changed"


def test_a_missing_producer_stamp_is_reported(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _stage("stage_a", root, REAL_ENV).save_table(_frame([1.0]), "a")
    _stage("stage_b", root, REAL_ENV).read_table("a")
    (root / "results" / "stamps" / "stage_a.json").unlink()
    with pytest.raises(wf.StampError) as info:
        _stage("stage_c", root, REAL_ENV).require("a", producer="stage_a")
    assert info.value.kind == "missing_stamp" and info.value.stage == "stage_a"
    assert wf.short_digest(info.value.found) in str(info.value)
    with pytest.raises(wf.StampError) as unknown:
        _stage("stage_c", root, REAL_ENV).read_table("a")
    assert unknown.value.kind == "unknown_producer"


def test_a_missing_stamp_upstream_is_found_through_the_consumer(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    _stage("stage_a", root, REAL_ENV).save_table(_frame([1.0]), "a")
    b = _stage("stage_b", root, REAL_ENV)
    b.read_table("a")
    b.save_table(_frame([2.0]), "b")
    (root / "results" / "stamps" / "stage_a.json").unlink()
    with pytest.raises(wf.StampError) as info:
        _stage("stage_c", root, REAL_ENV).read_table("b")
    assert info.value.kind == "missing_stamp" and info.value.stage == "stage_a"


def test_a_results_file_that_no_stamp_lists_cannot_be_read(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_c", root, REAL_ENV)
    pd.DataFrame({"value": [1.0]}).to_csv(ctx.results_dir / "written_by_hand.csv", index=False)
    with pytest.raises(wf.StampError) as info:
        ctx.read_table("written_by_hand")
    assert info.value.kind == "unknown_producer" and info.value.path == "results/written_by_hand.csv"


def test_a_raw_file_is_recorded_with_its_digest_and_no_producer(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    raw = root / "data" / "raw" / "wdi" / "extra.csv"
    raw.write_text("iso3,year,value\nAAA,2000,1\n", encoding="utf-8")
    user = _stage("stage_user", root, REAL_ENV)
    user.record_input(raw)
    user.save_table(_frame([1.0]), "t_user")
    stamp = wf.read_stamp(user.results_dir, "stage_user")
    assert stamp["inputs"]["data/raw/wdi/extra.csv"] == {"producer": None, "sha256": wf.file_digest(raw)}

    _stage("stage_next", root, REAL_ENV).read_table("t_user")
    raw.write_text("iso3,year,value\nAAA,2000,2\n", encoding="utf-8")
    with pytest.raises(wf.StampError) as info:
        _stage("stage_next", root, REAL_ENV).read_table("t_user")
    assert info.value.kind == "input_changed"
    assert info.value.path == "data/raw/wdi/extra.csv" and info.value.consumer == "stage_user"


def test_record_raw_inputs_lists_the_indicator_files_and_the_metadata(tmp_path, keep_sys_path):
    from dtt import panel as panel_mod

    root = _fake_root(tmp_path)
    names = [panel_mod.wdi_filename(g, code) for g in panel_mod.GROUPS for code in panel_mod.INDICATORS]
    for name in names:
        (root / "data" / "raw" / "wdi" / name).write_text("iso3,year,value\nAAA,2000,1\n", encoding="utf-8")
    ctx = _stage("stage_raw_all", root, REAL_ENV)
    keys = ctx.record_raw_inputs()
    assert len(keys) == 21
    stamp = wf.read_stamp(ctx.results_dir, "stage_raw_all")
    assert set(stamp["inputs"]) == {f"data/raw/wdi/{name}" for name in [*names, "country_metadata.csv"]}
    assert all(record["producer"] is None for record in stamp["inputs"].values())


def test_a_file_that_the_stage_read_and_then_wrote_is_an_output_only(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_rw", root, REAL_ENV)
    ctx.save_table(_frame([1.0]), "t_rw")
    ctx.read_table("t_rw")
    ctx.save_table(_frame([2.0]), "t_rw")
    stamp = wf.read_stamp(ctx.results_dir, "stage_rw")
    assert "results/t_rw.csv" in stamp["outputs"] and "results/t_rw.csv" not in stamp["inputs"]


def test_figures_and_inventory_are_listed_as_outputs(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_figs", root, REAL_ENV)
    ctx.save_figure(_figure(), "fig_stamp.png", show=False)
    stamp = wf.read_stamp(ctx.results_dir, "stage_figs")
    assert set(stamp["outputs"]) == {"figures/fig_stamp.png", "results/figure_inventory_stage_figs.csv"}


def test_stamps_hold_relative_paths_and_digests_only(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    env = dict(REAL_ENV)
    env[wf.RESULTS_VARIABLE] = str(tmp_path / "out" / "results")
    env[wf.FIGURES_VARIABLE] = str(tmp_path / "out" / "figures")
    ctx = _stage("stage_rel", root, env)
    ctx.save_table(_frame([1.0]), "t_rel")
    ctx.save_figure(_figure(), "f_rel.png", show=False)
    text = wf.stamp_path(ctx.results_dir, "stage_rel").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(root) not in text
    stamp = json.loads(text)
    assert set(stamp) == {"format", "stage", "inputs", "outputs"}
    for key in (*stamp["inputs"], *stamp["outputs"]):
        assert not key.startswith("/") and ".." not in key.split("/")
    assert set(stamp["outputs"]) == {"results/t_rel.csv", "figures/f_rel.png", "results/figure_inventory_stage_rel.csv"}


def test_two_identical_runs_write_identical_stamps(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)

    def run(name: str) -> dict[str, bytes]:
        env = dict(REAL_ENV)
        env[wf.RESULTS_VARIABLE] = str(tmp_path / name / "results")
        env[wf.FIGURES_VARIABLE] = str(tmp_path / name / "figures")
        a = _stage("stage_a", root, env)
        a.save_table(pd.DataFrame({"x": [0.1, 1.0 / 3.0]}), "a")
        b = _stage("stage_b", root, env)
        b.read_table("a")
        b.save_table(pd.DataFrame({"y": [2.0]}), "b")
        return {stage: wf.stamp_path(a.results_dir, stage).read_bytes() for stage in ("stage_a", "stage_b")}

    first = run("first")
    second = run("second")
    assert first == second


def test_stamp_checks_work_in_a_simulated_run(run_env, tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    producer = _stage("chain_a", root, run_env)
    producer.save_table(_frame([1.0]), "chain_table")
    consumer = _stage("chain_b", root, run_env)
    consumer.read_table("chain_table")
    assert wf.read_stamp(consumer.results_dir, "chain_b")["inputs"]["results/chain_table.csv"]["producer"] == "chain_a"
    (producer.results_dir / "chain_table.csv").write_text("value\n5.0\n", encoding="utf-8")
    with pytest.raises(wf.StampError) as info:
        _stage("chain_c", root, run_env).read_table("chain_table")
    assert info.value.kind == "changed" and info.value.stage == "chain_a"


def test_record_output_refuses_a_missing_file(tmp_path, keep_sys_path):
    root = _fake_root(tmp_path)
    ctx = _stage("stage_missing", root, REAL_ENV)
    with pytest.raises(FileNotFoundError):
        ctx.record_output(ctx.results_dir / "never_written.csv")
