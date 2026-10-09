"""Run context shared by the analysis notebooks.

A notebook opens with ``ctx = setup_run("02_case_effects")``.  The context holds
the paths of the inputs and outputs, the target economy, the scenarios of the
Thailand proposal, the thermal guard and the seeds, and it saves tables, JSON
files and figures below the results and figures directories.

Two modes exist.  The mode ``real`` reads the raw files of the repository and
needs the environment variable ``DTT_RUN_REAL=1``.  The mode ``sim`` is chosen
with ``DTT_WORLD=sim``: it builds once a simulated world (see
:mod:`dtt.simulate_global`) in the directory named by ``DTT_SIM_DIR`` and
writes every output below ``DTT_RESULTS_DIR`` and ``DTT_FIGURE_DIR``, which
default to sub-directories of the simulated world.  A simulated run never writes
to the results directory of the repository.

Environment variables
---------------------
DTT_RUN_REAL
    ``1`` allows a run on the real files; any other value does not.
DTT_WORLD
    ``real`` (default) or ``sim``.
DTT_SIM_DIR
    Directory of the simulated world (default: a directory in the system
    temporary directory).
DTT_RESULTS_DIR, DTT_FIGURE_DIR
    Override the directories that receive tables and figures.
DTT_GUARD
    ``1`` switches the thermal guard on and ``0`` switches it off; the default is
    on in a real run and off in a simulated run.

A simulated run refuses every output directory that lies inside the ``data``,
``results`` or ``figures`` directory of the repository, and every directory that
contains the repository.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "REAL_VARIABLE",
    "WORLD_VARIABLE",
    "SEEDS",
    "RunContext",
    "apply_style",
    "find_root",
    "resolve_mode",
    "setup_run",
]

REAL_VARIABLE = "DTT_RUN_REAL"
WORLD_VARIABLE = "DTT_WORLD"
SIM_DIR_VARIABLE = "DTT_SIM_DIR"
RESULTS_VARIABLE = "DTT_RESULTS_DIR"
FIGURES_VARIABLE = "DTT_FIGURE_DIR"
GUARD_VARIABLE = "DTT_GUARD"

#: Named seeds of the analysis; every run context holds its own copy.
SEEDS: Mapping[str, int] = MappingProxyType(
    {
        "effects": 0,
        "importance": 0,
        "importance bootstrap": 1,
        "transport": 0,
        "transport bootstrap": 2,
        "overlap test": 3,
        "meta": 0,
        "simulated world": 0,
    }
)

SIM_TARGET_FX = 33.0
SIM_TARGET_CAPEX_SHARE = 0.015


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Read a switch that is ``0`` or ``1``; an unset or empty variable gives ``default``."""
    raw = str(env.get(name, "")).strip()
    if raw == "":
        return default
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, not {raw!r}")
    return raw == "1"


def _inside(path: Path, parent: Path) -> bool:
    """Return True if ``path`` is ``parent`` or lies below it."""
    path, parent = path.resolve(), parent.resolve()
    return path == parent or parent in path.parents


def _refuse_repository_outputs(root: Path, targets: Mapping[str, Path]) -> None:
    """Raise ``ValueError`` if a simulated run would write into the data, results or figures of the repository."""
    protected = [root / "data", root / "results", root / "figures"]
    for label, path in targets.items():
        clash = next((p for p in protected if _inside(path, p)), None)
        if clash is None and _inside(root, path):
            clash = root
        if clash is not None:
            raise ValueError(f"A simulated run must not use {path} as its {label} directory because it overlaps {clash}.")


def find_root(start: str | Path | None = None) -> Path:
    """Return the repository root, the nearest parent directory that holds ``src/dtt``.

    Parameters
    ----------
    start : str or pathlib.Path, optional
        Directory where the search starts; the current directory by default.

    Returns
    -------
    pathlib.Path
        The repository root.

    Raises
    ------
    FileNotFoundError
        If no parent directory contains ``src/dtt/__init__.py``.
    """
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "dtt" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError(f"repository root not found from {here}")


def resolve_mode(env: Mapping[str, str] | None = None) -> str:
    """Return ``"real"`` or ``"sim"`` from the environment variables.

    Parameters
    ----------
    env : mapping, optional
        Environment to read; ``os.environ`` by default.

    Returns
    -------
    str
        ``"sim"`` when ``DTT_WORLD=sim``; ``"real"`` when ``DTT_WORLD`` is
        ``real`` or unset and ``DTT_RUN_REAL`` is ``1``.

    Raises
    ------
    RuntimeError
        If a real run is requested without ``DTT_RUN_REAL=1``.
    ValueError
        If ``DTT_WORLD`` has another value.
    """
    env = os.environ if env is None else env
    world = str(env.get(WORLD_VARIABLE, "")).strip().lower()
    if world == "sim":
        return "sim"
    if world not in ("", "real"):
        raise ValueError(f"{WORLD_VARIABLE} must be 'real' or 'sim', not {world!r}")
    if str(env.get(REAL_VARIABLE, "")).strip() == "1":
        return "real"
    raise RuntimeError(
        f"Set {REAL_VARIABLE}=1 to run on the real data, or {WORLD_VARIABLE}=sim to run on a simulated world."
    )


class _ThrottledGuard:
    """Call a guard at most once per ``min_interval`` seconds."""

    def __init__(self, guard: Callable[[], Any], min_interval: float = 10.0, clock: Callable[[], float] = time.monotonic):
        self._guard = guard
        self._min_interval = float(min_interval)
        self._clock = clock
        self._last: float | None = None
        self.calls = 0
        self.passed = 0

    def __call__(self) -> None:
        self.calls += 1
        now = self._clock()
        if self._last is not None and now - self._last < self._min_interval:
            return
        self._guard()
        self.passed += 1
        self._last = self._clock()


@dataclass
class RunContext:
    """Paths, settings and helpers of one notebook run.

    Attributes
    ----------
    stage : str
        Name of the notebook.
    mode : str
        ``"real"`` or ``"sim"``.
    root : pathlib.Path
        Repository root.
    raw_dir : pathlib.Path
        Directory with the twenty World Bank indicator files and ``country_metadata.csv``.
    catalogue_path : pathlib.Path
        Cases catalogue.
    investment_fx_path : pathlib.Path or None
        File with derived dollar costs, or None.
    baseline_path : pathlib.Path
        Long-format baseline of the target economy.
    proposal_path : pathlib.Path
        JSON file with the proposal scenarios of the target economy.
    processed_dir, results_dir, figures_dir : pathlib.Path
        Directories for processed data, tables and figures.
    target_iso3, target_name : str
        Code and name of the target economy.
    proposal : dict
        Content of the proposal file: ``status`` text and ``scenarios``.
    guard : callable or None
        Zero-argument function that returns when the machine is within thermal
        limits; None when the guard is off.
    seeds : dict of str to int
        Named seeds.
    """

    stage: str
    mode: str
    root: Path
    raw_dir: Path
    catalogue_path: Path
    investment_fx_path: Path | None
    baseline_path: Path
    proposal_path: Path
    processed_dir: Path
    results_dir: Path
    figures_dir: Path
    target_iso3: str
    target_name: str
    proposal: dict[str, Any]
    guard: Callable[[], None] | None
    seeds: dict[str, int] = field(default_factory=lambda: dict(SEEDS))
    figure_log: list[dict[str, Any]] = field(default_factory=list)
    guard_object: Any = None

    @property
    def is_real(self) -> bool:
        """True for a run on the real files."""
        return self.mode == "real"

    def primary_scenario(self) -> dict[str, Any]:
        """Return the proposal scenario that is flagged as primary."""
        for scenario in self.proposal["scenarios"]:
            if scenario.get("primary"):
                return dict(scenario)
        return dict(self.proposal["scenarios"][0])

    def table_path(self, name: str) -> Path:
        """Return the path of a table below the results directory."""
        return self.results_dir / (name if Path(name).suffix else f"{name}.csv")

    def save_table(self, df: pd.DataFrame, name: str, index: bool = False) -> Path:
        """Write a table to the results directory and return its path."""
        path = self.table_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)
        print(f"Table saved: {path}")
        return path

    def read_table(self, name: str, **kwargs: Any) -> pd.DataFrame:
        """Read a table from the results directory.

        Raises
        ------
        FileNotFoundError
            If the table has not been written by an earlier notebook.
        """
        path = self.table_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run the notebook that writes it first")
        return pd.read_csv(path, **kwargs)

    def require(self, *names: str) -> None:
        """Check that earlier notebooks have written the named tables.

        Raises
        ------
        FileNotFoundError
            Naming every missing file.
        """
        missing = [str(self.table_path(n)) for n in names if not self.table_path(n).is_file()]
        if missing:
            raise FileNotFoundError("missing results of earlier notebooks: " + ", ".join(missing))

    def save_json(self, obj: Any, name: str) -> Path:
        """Write a JSON file to the results directory and return its path."""
        path = self.results_dir / (name if Path(name).suffix else f"{name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_clean_for_json(obj), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        print(f"JSON saved: {path}")
        return path

    def read_json(self, name: str) -> Any:
        """Read a JSON file from the results directory."""
        path = self.results_dir / (name if Path(name).suffix else f"{name}.json")
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run the notebook that writes it first")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_figure(self, fig: Any, name: str, show: bool = True) -> Path:
        """Save a matplotlib figure to the figures directory.

        The figure inventory of the stage (panels, axis labels, legend entries)
        is rewritten to ``figure_inventory_<stage>.csv`` in the results directory.
        """
        import matplotlib.pyplot as plt

        path = self.figures_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        axes = [a for a in fig.axes if a.get_visible()]
        self.figure_log.append(
            {
                "file": name,
                "panels": len(axes),
                "x labels": " | ".join(dict.fromkeys(a.get_xlabel() for a in axes if a.get_xlabel())),
                "y labels": " | ".join(dict.fromkeys(a.get_ylabel() for a in axes if a.get_ylabel())),
                "legend entries": sum(len(a.get_legend().get_texts()) for a in axes if a.get_legend() is not None)
                + sum(len(legend.get_texts()) for legend in fig.legends),
            }
        )
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
        inventory = self.results_dir / f"figure_inventory_{self.stage}.csv"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.figure_log).to_csv(inventory, index=False)
        print(f"Figure saved: {path}")
        return path

    def describe(self) -> None:
        """Print the mode, paths, package versions and the machine headroom."""
        import importlib.metadata as md

        print(f"Stage: {self.stage}   Mode: {self.mode}")
        print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
        versions = []
        for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib"):
            try:
                versions.append(f"{pkg} {md.version(pkg)}")
            except md.PackageNotFoundError:
                versions.append(f"{pkg} missing")
        print(", ".join(versions))
        print(f"Raw files:   {self.raw_dir}")
        print(f"Catalogue:   {self.catalogue_path}")
        print(f"Results:     {self.results_dir}")
        print(f"Figures:     {self.figures_dir}")
        print(f"Target:      {self.target_name} ({self.target_iso3}); thermal guard {'on' if self.guard else 'off'}")
        try:
            from dtt.thermal import headroom

            h = headroom()
            procs = h.get("claude_processes")
            count = procs.get("count") if isinstance(procs, Mapping) else procs

            def show(value: Any, template: str) -> str:
                return "unavailable" if value is None else template.format(value)

            print(
                f"Headroom: CPU {show(h.get('cpu_percent'), '{:.0f}%')}, "
                f"memory available {show(h.get('mem_available_gb'), '{:.1f} GB')}, "
                f"temperature {show(h.get('temperature_c'), '{:.0f} C')}, "
                f"Claude-related processes {show(count, '{}')}"
            )
        except Exception as exc:  # pragma: no cover - reporting only
            print(f"Headroom: not available ({type(exc).__name__})")


def _clean_for_json(obj: Any) -> Any:
    """Return a copy of ``obj`` that standard JSON can hold (NaN and infinities become null)."""
    if isinstance(obj, Mapping):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    if isinstance(obj, pd.Series):
        return _clean_for_json(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _clean_for_json(obj.to_dict(orient="records"))
    if isinstance(obj, np.generic):
        return _clean_for_json(obj.item())
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def apply_style() -> None:
    """Set the grayscale matplotlib style used by every figure."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.15",
            "axes.labelcolor": "0.0",
            "axes.prop_cycle": cycler(color=["0.0", "0.35", "0.6"], linestyle=["-", "--", ":"]),
            "text.color": "0.0",
            "xtick.color": "0.15",
            "ytick.color": "0.15",
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": True,
            "legend.edgecolor": "0.4",
            "legend.framealpha": 1.0,
            "image.cmap": "gray",
        }
    )


def _write_sim_target(sim_dir: Path) -> str:
    """Create the baseline, proposal and marker files of the simulated target economy.

    Returns
    -------
    str
        Code of the economy chosen as target: the first economy, in alphabetical
        order, without a case whose receipts, GDP and arrivals are present in
        every year from 2015 to 2024.
    """
    from dtt import panel as panel_mod

    cat = pd.read_csv(sim_dir / "cases_catalogue.csv", encoding="utf-8-sig")
    case_economies = set(cat["iso3"].astype(str))
    panel = panel_mod.build_global_panel(sim_dir)
    years = range(2015, 2025)
    needed = ["receipts_usd", "gdp_usd", "arrivals", "receipts_pct_gdp"]
    complete = panel.loc[panel["year"].isin(years), ["iso3", "year", *needed]].dropna()
    counts = complete.groupby("iso3")["year"].nunique()
    candidates = sorted(i for i in counts.index if i not in case_economies and counts[i] == len(years))
    if not candidates:
        raise RuntimeError("the simulated world has no economy that can serve as the target")
    iso3 = candidates[0]
    sub = panel[(panel["iso3"] == iso3) & (panel["year"] >= 2015)].sort_values("year")
    rows = []
    for _, r in sub.iterrows():
        year = int(r["year"])
        values = {
            "intl_tourism_receipts_usd_bn": r["receipts_usd"] / 1e9,
            "gdp_current_usd_bn": r["gdp_usd"] / 1e9,
            "intl_arrivals_million": r["arrivals"] / 1e6,
            "fx_thb_per_usd": SIM_TARGET_FX,
            "receipts_pct_gdp": r["receipts_pct_gdp"],
        }
        for var, val in values.items():
            rows.append({"year": year, "variable": var, "value": val, "unit": "", "source": "simulated", "url": "", "note": ""})
    pd.DataFrame(rows).to_csv(sim_dir / "target_baseline.csv", index=False)
    last = sub.dropna(subset=["gdp_usd"]).iloc[-1]
    capex_usd_bn = SIM_TARGET_CAPEX_SHARE * float(last["gdp_usd"]) / 1e9
    proposal = {
        "status": "Simulated target economy with a simulated investment scenario.",
        "fx_thb_per_usd_reference": SIM_TARGET_FX,
        "scenarios": [
            {
                "name": "complex_total",
                "capex_thb_bn": round(capex_usd_bn * SIM_TARGET_FX, 3),
                "scope": "simulated scenario",
                "primary": True,
            },
            {
                "name": "half_scale",
                "capex_thb_bn": round(0.5 * capex_usd_bn * SIM_TARGET_FX, 3),
                "scope": "simulated scenario at half scale",
                "primary": False,
            },
        ],
    }
    (sim_dir / "target_proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    (sim_dir / "target_iso3.txt").write_text(iso3, encoding="utf-8")
    return iso3


def setup_run(
    stage: str,
    mode: str | None = None,
    root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RunContext:
    """Create the run context of a notebook.

    Parameters
    ----------
    stage : str
        Name of the notebook, used in file names of the figure inventory.
    mode : {"real", "sim"}, optional
        Overrides the mode from the environment; a real run still needs
        ``DTT_RUN_REAL=1`` and is refused while ``DTT_WORLD=sim`` is set.
    root : str or pathlib.Path, optional
        Repository root; found from the current directory by default.
    env : mapping, optional
        Environment to read; ``os.environ`` by default.

    Returns
    -------
    RunContext
        The context.  A real run reads only the proposal file here and checks
        that the other input files exist; a simulated run builds its world on
        first use.

    Raises
    ------
    RuntimeError
        If the mode cannot be resolved from the environment or a real run is
        requested without ``DTT_RUN_REAL=1``.
    ValueError
        If ``mode`` or a switch variable has an invalid value, or a simulated
        run is given an output directory that overlaps the repository data,
        results or figures.
    FileNotFoundError
        If an input file of a real run is missing.
    """
    env = os.environ if env is None else env
    if mode is None:
        mode = resolve_mode(env)
    elif mode not in ("real", "sim"):
        raise ValueError("mode must be 'real' or 'sim'")
    elif mode == "real" and resolve_mode(env) != "real":
        raise RuntimeError(f"A real run is refused while {WORLD_VARIABLE}=sim is set.")
    seeds = dict(SEEDS)
    root_path = find_root(root)
    if str(root_path / "src") not in sys.path:
        sys.path.insert(0, str(root_path / "src"))

    if mode == "real":
        raw_dir = root_path / "data" / "raw" / "wdi"
        processed_dir = root_path / "data" / "processed"
        catalogue_path = processed_dir / "cases_catalogue.csv"
        fx = processed_dir / "cases_investment_fx.csv"
        investment_fx_path: Path | None = fx if fx.is_file() else None
        baseline_path = processed_dir / "thailand_baseline.csv"
        proposal_path = processed_dir / "thailand_proposal.json"
        results_dir = Path(env.get(RESULTS_VARIABLE) or root_path / "results")
        figures_dir = Path(env.get(FIGURES_VARIABLE) or root_path / "figures")
        target_iso3, target_name = "THA", "Thailand"
        required = [raw_dir / "country_metadata.csv", catalogue_path, baseline_path, proposal_path]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("input files missing: " + ", ".join(missing))
        guard_on = _flag(env, GUARD_VARIABLE, True)
    else:
        sim_dir = Path(env.get(SIM_DIR_VARIABLE) or Path(tempfile.gettempdir()) / "dtt_sim_world")
        results_dir = Path(env.get(RESULTS_VARIABLE) or sim_dir / "results")
        figures_dir = Path(env.get(FIGURES_VARIABLE) or sim_dir / "figures")
        _refuse_repository_outputs(root_path, {"world": sim_dir, "results": results_dir, "figures": figures_dir})
        guard_on = _flag(env, GUARD_VARIABLE, False)
        if not (sim_dir / "cases_catalogue.csv").is_file():
            from dtt.simulate_global import simulate_global_world

            simulate_global_world(sim_dir, n_econ=60, n_cases=16, seed=seeds["simulated world"], cluster_economies=True)
        if not (sim_dir / "target_iso3.txt").is_file():
            _write_sim_target(sim_dir)
        target_iso3 = (sim_dir / "target_iso3.txt").read_text(encoding="utf-8").strip()
        target_name = f"{target_iso3} (simulated)"
        raw_dir = sim_dir
        processed_dir = sim_dir / "processed"
        catalogue_path = sim_dir / "cases_catalogue.csv"
        investment_fx_path = None
        baseline_path = sim_dir / "target_baseline.csv"
        proposal_path = sim_dir / "target_proposal.json"

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    guard: Callable[[], None] | None = None
    guard_object = None
    if guard_on:
        from dtt.thermal import ThermalGuard

        guard_object = ThermalGuard(log_path=results_dir / "thermal_log.csv")
        guard = _ThrottledGuard(guard_object, min_interval=10.0)

    apply_style()
    return RunContext(
        stage=stage,
        mode=mode,
        root=root_path,
        raw_dir=raw_dir,
        catalogue_path=catalogue_path,
        investment_fx_path=investment_fx_path,
        baseline_path=baseline_path,
        proposal_path=proposal_path,
        processed_dir=processed_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
        target_iso3=target_iso3,
        target_name=target_name,
        proposal=proposal,
        guard=guard,
        seeds=seeds,
        guard_object=guard_object,
    )
