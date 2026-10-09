"""Run context shared by the analysis notebooks.

A notebook opens with ``ctx = setup_run("02_case_effects")``.  The context holds
the paths of the inputs and outputs, the target economy, the scenarios of the
Thailand proposal, the thermal guard and the seeds, and it saves tables, JSON
files and figures below the results and figures directories.

The context always reads the dataset of the repository: the World Bank files in
``data/raw/wdi``, the cases catalogue and the other files in ``data/processed``.

Environment variables
---------------------
DTT_RESULTS_DIR, DTT_FIGURE_DIR
    Override the directories that receive tables and figures; they default to
    ``results`` and ``figures`` of the repository.
DTT_GUARD
    ``1`` switches the thermal guard on and ``0`` switches it off; the default is
    on.

Printed paths
-------------
Everything the context prints names a file by a label that does not depend on
the machine: ``results/<file>`` and ``figures/<file>`` for the output
directories and a path relative to the repository for other files of the
repository.  No printed line holds an absolute path, so the output stored in an executed
notebook does not reveal the folders of the computer that ran it.

Result stamps
-------------
Every file written by :meth:`RunContext.save_table`, :meth:`RunContext.save_json`
and :meth:`RunContext.save_figure` receives a stamp in the JSON file
``file_stamps.json`` of the results directory.  The stamp holds the SHA-256
digest of the file, the name of the stage that wrote it and the digests of the
files that the stage had registered with :meth:`RunContext.require` when it
wrote the file.  Files are named in the stamp file by the labels above, and the
stamp file holds no time and no absolute path, so equal runs give equal stamp
files wherever the directories are.

:meth:`RunContext.require` checks every file that it is given against its stamp
and then, through the recorded inputs, the files from which that file was
computed.  It raises :class:`StaleResultError` when a stamped file differs from
the digest recorded by its producer, or when a recorded input differs from the
digest recorded when the file was computed.  A file without a stamp, for
example a raw data file, is accepted, but its digest is still compared where
another file recorded it as an input.  Only files that a stage registered with
``require`` are covered.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "SEEDS",
    "STAMP_FILE",
    "RunContext",
    "StaleResultError",
    "apply_style",
    "check_requirements",
    "find_root",
    "setup_run",
]

RESULTS_VARIABLE = "DTT_RESULTS_DIR"
FIGURES_VARIABLE = "DTT_FIGURE_DIR"
GUARD_VARIABLE = "DTT_GUARD"

#: Name of the stamp file in the results directory.
STAMP_FILE = "file_stamps.json"

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
    }
)

#: Packages whose minimum versions in ``requirements.txt`` are enforced by :func:`setup_run`.
CHECKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "psutil")

#: Seconds between two calls that reach the thermal guard.
GUARD_MIN_INTERVAL_SECONDS = 10.0
#: Longest wait of the thermal guard for CPU load alone, in seconds.
GUARD_CPU_WAIT_SECONDS = 300.0

_STAMP_FORMAT = 1
_DIGEST_BLOCK = 1 << 20
_REPLACE_ATTEMPTS = 5
_LOAD_READER_NAMES = {"_OtherProcessesLoad": "system load without this program and its child processes"}


class StaleResultError(RuntimeError):
    """A result file is not what its producer wrote, or was computed from files that have changed.

    The message names every file concerned together with the stage that
    produced it and lists the stages to run again.
    """


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Read a switch that is ``0`` or ``1``; an unset or empty variable gives ``default``."""
    raw = str(env.get(name, "")).strip()
    if raw == "":
        return default
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, not {raw!r}")
    return raw == "1"


def _relative_to(path: Path, base: Path) -> str | None:
    """Return the POSIX path of ``path`` relative to ``base``, ``""`` for ``base`` itself, or None if it lies elsewhere."""
    try:
        relative = Path(path).resolve().relative_to(Path(base).resolve())
    except ValueError:
        return None
    text = relative.as_posix()
    return "" if text == "." else text


def _number_text(value: Any) -> str:
    """Return a number in the shortest decimal form, or ``unknown`` for anything that is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unknown"
    return f"{value:g}"


def _under_root(path: Path, root: Path) -> str:
    """Return ``path`` relative to the repository root, or only its file name if it lies outside."""
    relative = _relative_to(path, root)
    return path.name if relative is None else (relative or ".")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the content of a file as 64 lowercase hexadecimal digits."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_DIGEST_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with line feeds as the only line ending, whatever the operating system."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_text_atomic(path: Path, text: str) -> None:
    """Write a text file in one step: the content goes to a temporary file that then replaces ``path``."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        _write_text(temporary, text)
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _load_stamps(path: Path, label: str) -> dict[str, dict[str, Any]]:
    """Read the stamp file.

    Parameters
    ----------
    path : pathlib.Path
        The stamp file; a missing file gives an empty record.
    label : str
        Name of the file in messages.

    Returns
    -------
    dict
        File label to a dict with ``sha256`` (str), ``stage`` (str) and
        ``inputs`` (dict of file label to digest).

    Raises
    ------
    ValueError
        If the file cannot be read or does not have the layout that
        :func:`_store_stamp` writes.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files = data["files"]
        if data["format"] != _STAMP_FORMAT or not isinstance(files, dict):
            raise ValueError("unknown layout")
        for entry in files.values():
            inputs = entry["inputs"]
            valid = isinstance(entry["sha256"], str) and isinstance(entry["stage"], str) and isinstance(inputs, dict)
            if not valid or not all(isinstance(k, str) and isinstance(v, str) for k, v in inputs.items()):
                raise ValueError("unknown layout")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"the stamp file {label} cannot be used ({type(exc).__name__}); delete it to continue without stamps") from exc
    return files


def _store_stamp(path: Path, label: str, key: str, entry: Mapping[str, Any]) -> None:
    """Add or replace the stamp of one file in the stamp file.

    The file is written with sorted keys and without any time, so equal stamps
    give equal bytes.
    """
    files = _load_stamps(path, label)
    files[key] = dict(entry)
    text = json.dumps({"format": _STAMP_FORMAT, "files": files}, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, text)


def _release(version: str) -> tuple[int, ...]:
    """Return the leading numeric release of a version string, for example ``(2, 5, 3)`` for ``"2.5.3"``."""
    match = re.match(r"\d+(?:\.\d+)*", version.strip())
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def check_requirements(root: str | Path, packages: tuple[str, ...] = CHECKED_PACKAGES) -> None:
    """Check that the installed packages meet the minimum versions of ``requirements.txt``.

    Parameters
    ----------
    root : str or pathlib.Path
        Repository root.  Nothing is checked when it holds no ``requirements.txt``.
    packages : tuple of str, optional
        Names of the packages to check; a package without a line of the form
        ``name>=version`` in the file is not checked.

    Raises
    ------
    RuntimeError
        Naming every package that is missing or older than the file requires,
        the Python interpreter in use and the way to select the environment of
        the project for a notebook run.
    """
    path = Path(root) / "requirements.txt"
    if not path.is_file():
        return
    import importlib.metadata as md

    minimum: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*>=\s*(\d[0-9A-Za-z.]*)", line)
        if match and match.group(1).lower() in packages:
            minimum[match.group(1).lower()] = match.group(2)
    problems = []
    for name in packages:
        wanted = minimum.get(name)
        if wanted is None:
            continue
        try:
            found = md.version(name)
        except md.PackageNotFoundError:
            problems.append(f"{name} is not installed (requirements.txt asks for {wanted} or later)")
            continue
        if _release(found) < _release(wanted):
            problems.append(f"{name} {found} is older than the {wanted} that requirements.txt asks for")
    if problems:
        raise RuntimeError(
            "The Python environment does not meet requirements.txt: "
            + "; ".join(problems)
            + f". Interpreter in use: {sys.executable}. Install the packages of requirements.txt in the environment of the project "
            "and run the notebook with the kernel of that environment, for example "
            "python -m ipykernel install --user --name dtt and then jupyter nbconvert --execute "
            "--ExecutePreprocessor.kernel_name=dtt."
        )


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


def _build_guard(results_dir: Path) -> tuple[Any, _ThrottledGuard]:
    """Create the thermal guard of a run and the throttled callable that the notebooks use.

    Parameters
    ----------
    results_dir : pathlib.Path
        Directory that receives ``thermal_log.csv``.

    Returns
    -------
    guard : dtt.thermal.ThermalGuard
        The guard.  It reads the temperature through the probes of
        :mod:`dtt.thermal`, reads the load of the machine without this program
        and its child processes, waits for CPU load for at most
        ``GUARD_CPU_WAIT_SECONDS`` and prints a line at the start and at the end
        of every wait.
    throttled : callable
        Calls the guard at most once per ``GUARD_MIN_INTERVAL_SECONDS``.
    """
    from dtt.thermal import ThermalGuard

    guard = ThermalGuard(
        log_path=results_dir / "thermal_log.csv",
        max_cpu_wait_seconds=GUARD_CPU_WAIT_SECONDS,
        verbose=True,
    )
    return guard, _ThrottledGuard(guard, min_interval=GUARD_MIN_INTERVAL_SECONDS)


@dataclass(repr=False)
class RunContext:
    """Paths, settings and helpers of one notebook run.

    Attributes
    ----------
    stage : str
        Name of the notebook.
    root : pathlib.Path
        Repository root.
    raw_dir : pathlib.Path
        Directory with the twenty World Bank indicator files and ``country_metadata.csv``.
    catalogue_path : pathlib.Path
        Cases catalogue.
    investment_fx_path : pathlib.Path
        File with derived dollar costs.
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
    figure_log : list of dict
        One row per figure saved by this context, the content of the figure
        inventory; saving a figure again replaces its row.
    guard_object : dtt.thermal.ThermalGuard or None
        The guard behind ``guard``; None when the guard is off.
    inputs_registered : dict of str to str
        Label and SHA-256 digest of every file passed to :meth:`require`; these
        are the inputs recorded in the stamp of each file written afterwards.
    """

    stage: str
    root: Path
    raw_dir: Path
    catalogue_path: Path
    investment_fx_path: Path
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
    inputs_registered: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __repr__(self) -> str:
        """Return the stage; the paths of the context are left out because they are absolute."""
        return f"RunContext(stage={self.stage!r})"

    def primary_scenario(self) -> dict[str, Any]:
        """Return the proposal scenario that is flagged as primary."""
        for scenario in self.proposal["scenarios"]:
            if scenario.get("primary"):
                return dict(scenario)
        return dict(self.proposal["scenarios"][0])

    # ------------------------------------------------------------------
    # Labels of files
    # ------------------------------------------------------------------
    def _label_bases(self) -> list[tuple[str | None, Path]]:
        """Directories that give files their labels, in the order in which they are tried."""
        return [("results", self.results_dir), ("figures", self.figures_dir), (None, self.root)]

    def _label(self, path: str | Path) -> str | None:
        """Return the machine-independent label of a file, or None if it lies in none of the known directories."""
        for prefix, base in self._label_bases():
            relative = _relative_to(Path(path), base)
            if relative is None:
                continue
            if prefix is None:
                return relative or "."
            return f"{prefix}/{relative}" if relative else prefix
        return None

    def _display(self, path: str | Path) -> str:
        """Return the text that names a file in printed lines: its label, or its file name if it has none."""
        label = self._label(path)
        return Path(path).name if label is None else label

    def _path_of(self, label: str) -> Path:
        """Return the path that a label stands for."""
        head, _, rest = label.partition("/")
        if head == "results":
            base = self.results_dir
        elif head == "figures":
            base = self.figures_dir
        else:
            return self.root / label
        return base / rest if rest else base

    # ------------------------------------------------------------------
    # Stamps
    # ------------------------------------------------------------------
    def _stamp_path(self) -> Path:
        """Return the path of the stamp file."""
        return self.results_dir / STAMP_FILE

    def _check_stamp_file(self) -> None:
        """Raise ``ValueError`` if the stamp file exists and cannot be used, before a file is written."""
        _load_stamps(self._stamp_path(), self._display(self._stamp_path()))

    def _stamp(self, path: Path) -> None:
        """Record the stamp of a file that this stage has just written."""
        label = self._label(path)
        if label is None:
            return
        entry = {
            "sha256": _sha256_file(path),
            "stage": self.stage,
            "inputs": dict(sorted(self.inputs_registered.items())),
        }
        stamp_path = self._stamp_path()
        _store_stamp(stamp_path, self._display(stamp_path), label, entry)

    def stamp_file(self, path: str | Path) -> Path:
        """Stamp a file that the stage wrote without :meth:`save_table`, :meth:`save_json` or :meth:`save_figure`.

        Parameters
        ----------
        path : str or pathlib.Path
            An existing file below the results or figures directory or below the
            repository.

        Returns
        -------
        pathlib.Path
            The path of the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file lies outside the directories above.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{self._display(path)} not found")
        if self._label(path) is None:
            raise ValueError(f"{path.name} lies outside the results, figures and repository directories and cannot be stamped")
        self._stamp(path)
        return path

    def _digest_of(self, label: str, cache: dict[str, str | None]) -> str | None:
        """Return the digest of the file with this label now, or None if it does not exist."""
        if label not in cache:
            path = self._path_of(label)
            cache[label] = _sha256_file(path) if path.is_file() else None
        return cache[label]

    def _verify(
        self,
        label: str,
        stamps: Mapping[str, Mapping[str, Any]],
        digests: dict[str, str | None],
        done: set[str],
        problems: list[tuple[str, str]],
    ) -> None:
        """Check one file against its stamp and, through the recorded inputs, the files it was computed from.

        Each problem found is appended to ``problems`` as a message and the
        name of the stage that has to run again: the producer of a file that
        was changed or removed, and the producer of a file that was computed
        from an input that has changed since.  A file is checked once.
        """
        if label in done:
            return
        done.add(label)
        entry = stamps.get(label)
        if entry is None:
            return
        stage = entry["stage"]
        current = self._digest_of(label, digests)
        if current is None:
            problems.append((f"{label} was written by stage {stage} and no longer exists", stage))
        elif current != entry["sha256"]:
            problems.append((f"{label} was written by stage {stage} and its content has changed since", stage))
        for input_label, recorded in entry["inputs"].items():
            self._verify(input_label, stamps, digests, done, problems)
            now = self._digest_of(input_label, digests)
            if now == recorded:
                continue
            source = stamps.get(input_label)
            if source is not None and (now is None or now != source["sha256"]):
                continue
            head = f"{label} (stage {stage}) was computed from"
            if now is None:
                problems.append((f"{head} {input_label}, which no longer exists", stage))
            elif source is None:
                problems.append((f"{head} {input_label}, which has changed since", stage))
            else:
                problems.append((f"{head} an earlier version of {input_label}, which stage {source['stage']} has written again since", stage))

    # ------------------------------------------------------------------
    # Tables, JSON files and figures
    # ------------------------------------------------------------------
    def table_path(self, name: str) -> Path:
        """Return the path of a table below the results directory."""
        return self.results_dir / (name if Path(name).suffix else f"{name}.csv")

    def _json_path(self, name: str) -> Path:
        """Return the path of a JSON file below the results directory."""
        return self.results_dir / (name if Path(name).suffix else f"{name}.json")

    def save_table(self, df: pd.DataFrame, name: str, index: bool = False) -> Path:
        """Write a table to the results directory, stamp it and return its path.

        The file is a CSV file with line feeds as line endings.  The returned
        path is absolute; the printed line names the file by its label.
        """
        self._check_stamp_file()
        path = self.table_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index, lineterminator="\n")
        self._stamp(path)
        print(f"Table saved: {self._display(path)}")
        return path

    def read_table(self, name: str, **kwargs: Any) -> pd.DataFrame:
        """Read a table from the results directory.

        Parameters
        ----------
        name : str
            Table name, with ``.csv`` added when it has no suffix.
        **kwargs
            Passed to :func:`pandas.read_csv`.  Unless the caller sets
            ``float_precision``, or selects an engine other than the C engine,
            numbers are parsed with ``float_precision="round_trip"``, which
            returns exactly the floating point numbers that were written.

        Raises
        ------
        FileNotFoundError
            If the table has not been written by an earlier notebook.
        """
        path = self.table_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{self._display(path)} not found; run the notebook that writes it first")
        if kwargs.get("engine") in (None, "c"):
            kwargs.setdefault("float_precision", "round_trip")
        return pd.read_csv(path, **kwargs)

    def require(self, *names: str | Path) -> None:
        """Check that earlier notebooks have written the named files and that they are unchanged.

        Each file must exist.  A file with a stamp must have the digest that its
        producer recorded, and every input that the stamp records must have the
        digest that it had when the file was computed; the check continues
        through the stamps of those inputs.  A file without a stamp is accepted.
        The digests of the files are registered as the inputs of every file that
        this stage writes afterwards.

        Parameters
        ----------
        *names : str or pathlib.Path
            Names of files below the results directory, written as for
            :meth:`table_path` (``.csv`` is added when there is no suffix), or
            paths of files elsewhere, for example in the processed data
            directory.

        Raises
        ------
        FileNotFoundError
            Naming every missing file.
        StaleResultError
            Naming every file that was changed after its producer wrote it or
            that was computed from a file that has changed since, with the stage
            of each, and the stages to run again.
        ValueError
            If the stamp file cannot be read.
        """
        paths = [self.table_path(n) for n in names]
        missing = [self._display(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing results of earlier notebooks: " + ", ".join(missing))
        stamps = _load_stamps(self._stamp_path(), self._display(self._stamp_path()))
        labels = [label for label in (self._label(p) for p in paths) if label is not None]
        digests: dict[str, str | None] = {}
        done: set[str] = set()
        problems: list[tuple[str, str]] = []
        for label in labels:
            self._verify(label, stamps, digests, done, problems)
        if problems:
            messages = list(dict.fromkeys(m for m, _ in problems))
            stages = sorted({s for _, s in problems})
            raise StaleResultError(
                "results of earlier notebooks are not what their producers wrote or are out of date:\n"
                + "\n".join(f"  - {m}" for m in messages)
                + "\nRun the notebooks again, in order, starting with the first of these: "
                + ", ".join(stages)
                + "."
            )
        for label in labels:
            self.inputs_registered[label] = str(self._digest_of(label, digests))

    def save_json(self, obj: Any, name: str) -> Path:
        """Write a JSON file with line feeds as line endings to the results directory, stamp it and return its path."""
        self._check_stamp_file()
        path = self._json_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(path, json.dumps(_clean_for_json(obj), indent=2, sort_keys=True, allow_nan=False))
        self._stamp(path)
        print(f"JSON saved: {self._display(path)}")
        return path

    def read_json(self, name: str) -> Any:
        """Read a JSON file from the results directory."""
        path = self._json_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{self._display(path)} not found; run the notebook that writes it first")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_figure(self, fig: Any, name: str, show: bool = True) -> Path:
        """Save a matplotlib figure to the figures directory and stamp it.

        The figure inventory of the stage (panels, axis labels, legend entries)
        is rewritten to ``figure_inventory_<stage>.csv`` in the results
        directory.  It holds one row per figure file: saving a figure under a
        name that is already in the inventory replaces its row.
        """
        import matplotlib.pyplot as plt

        self._check_stamp_file()
        path = self.figures_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        axes = [a for a in fig.axes if a.get_visible()]
        row = {
            "file": name,
            "panels": len(axes),
            "x labels": " | ".join(dict.fromkeys(a.get_xlabel() for a in axes if a.get_xlabel())),
            "y labels": " | ".join(dict.fromkeys(a.get_ylabel() for a in axes if a.get_ylabel())),
            "legend entries": sum(len(a.get_legend().get_texts()) for a in axes if a.get_legend() is not None)
            + sum(len(legend.get_texts()) for legend in fig.legends),
        }
        for position, previous in enumerate(self.figure_log):
            if previous["file"] == name:
                self.figure_log[position] = row
                break
        else:
            self.figure_log.append(row)
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        self._stamp(path)
        if show:
            plt.show()
        plt.close(fig)
        inventory = self.results_dir / f"figure_inventory_{self.stage}.csv"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.figure_log).to_csv(inventory, index=False, lineterminator="\n")
        self._stamp(inventory)
        print(f"Figure saved: {self._display(path)}")
        return path

    # ------------------------------------------------------------------
    # Report of the run
    # ------------------------------------------------------------------
    def guard_summary(self, show: bool = False) -> dict[str, Any]:
        """Return the totals of the thermal guard over the run so far.

        Parameters
        ----------
        show : bool, default False
            Print one line with the totals.

        Returns
        -------
        dict
            ``active`` (whether a guard exists), ``calls`` (calls of ``ctx.guard``)
            and ``passed`` (calls that reached the thermal guard), and the
            totals of :meth:`dtt.thermal.ThermalGuard.summary`: ``checks``,
            ``waits``, ``waited_seconds``, ``cpu_timeouts`` (waits for CPU load
            that ended at the longest allowed time), ``max_temperature_c`` and
            ``temperature_source``.  Without a guard the counts are zero and
            the temperature entries are None.
        """
        summary: dict[str, Any] = {
            "active": self.guard_object is not None,
            "calls": 0,
            "passed": 0,
            "checks": 0,
            "waits": 0,
            "waited_seconds": 0.0,
            "cpu_timeouts": 0,
            "max_temperature_c": None,
            "temperature_source": None,
        }
        if self.guard_object is not None:
            summary.update(self.guard_object.summary())
        if isinstance(self.guard, _ThrottledGuard):
            summary["calls"] = self.guard.calls
            summary["passed"] = self.guard.passed
        if show:
            if not summary["active"]:
                print("Thermal guard: off.")
            else:
                peak = summary["max_temperature_c"]
                peak_text = "not read" if peak is None else f"{peak:.1f} C ({summary['temperature_source']})"
                print(
                    f"Thermal guard: {summary['checks']} checks, {summary['waits']} waits "
                    f"({summary['waited_seconds']:.1f} seconds waited), {summary['cpu_timeouts']} CPU timeouts, "
                    f"highest temperature {peak_text}."
                )
        return summary

    def _thermal_line(self, readings: Mapping[str, Any] | None) -> str:
        """Return the line that tells whether a temperature reading exists and which load reader the guard uses."""
        if readings is None:
            reading = "not checked"
            available: bool | None = None
        elif readings.get("temperature_c") is None:
            reading = "not available on this machine"
            available = False
        else:
            reading = "available"
            available = True
        guard = self.guard_object
        if guard is None:
            return f"Thermal guard: off; temperature reading: {reading}; load reader: none."
        reader = getattr(guard, "_read_cpu", None)
        kind = type(reader).__name__
        name = _LOAD_READER_NAMES.get(kind) or f"custom function {getattr(reader, '__name__', kind)}"
        max_temp = _number_text(getattr(guard, "max_temp_c", None))
        max_cpu = _number_text(getattr(guard, "max_cpu_percent", None))
        cpu_wait = _number_text(getattr(guard, "max_cpu_wait_seconds", None))
        if available:
            policy = f", the guard waits while it exceeds {max_temp} C"
        elif available is False:
            policy = f", the guard waits while the load exceeds {max_cpu} percent, for at most {cpu_wait} seconds per wait"
        else:
            policy = ""
        return f"Thermal guard: on; temperature reading: {reading}{policy}; load reader: {name}."

    def describe(self) -> None:
        """Print the stage, package versions, the dataset files, the output directories, the thermal guard and the machine headroom.

        Paths are printed as labels, never as absolute paths.  One line states
        whether a temperature reading is available on this machine and which
        load reader the thermal guard uses.
        """
        import importlib.metadata as md

        print(f"Stage: {self.stage}")
        print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
        versions = []
        for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib"):
            try:
                versions.append(f"{pkg} {md.version(pkg)}")
            except md.PackageNotFoundError:
                versions.append(f"{pkg} missing")
        print(", ".join(versions))
        print(f"Raw files:   {self._display(self.raw_dir)}")
        print(f"Catalogue:   {self._display(self.catalogue_path)}")
        print(f"Results:     {self._display(self.results_dir)}")
        print(f"Figures:     {self._display(self.figures_dir)}")
        target = self.target_name if self.target_iso3 in self.target_name else f"{self.target_name} ({self.target_iso3})"
        print(f"Target:      {target}")
        print("Seeds:       " + ", ".join(f"{name} {value}" for name, value in self.seeds.items()))
        readings: Mapping[str, Any] | None = None
        try:
            from dtt.thermal import headroom

            readings = headroom()

            def show(value: Any, template: str) -> str:
                return "unavailable" if value is None else template.format(value)

            print(
                f"Headroom: CPU {show(readings.get('cpu_percent'), '{:.0f}%')}, "
                f"memory available {show(readings.get('mem_available_gb'), '{:.1f} GB')}, "
                f"temperature {show(readings.get('temperature_c'), '{:.0f} C')}"
            )
        except Exception as exc:  # pragma: no cover - reporting only
            readings = None
            print(f"Headroom: not available ({type(exc).__name__})")
        print(self._thermal_line(readings))


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


def setup_run(
    stage: str,
    root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RunContext:
    """Create the run context of a notebook.

    Parameters
    ----------
    stage : str
        Name of the notebook, used in file names of the figure inventory and
        as the producer named in the stamps of the files it writes.
    root : str or pathlib.Path, optional
        Repository root; found from the current directory by default.
    env : mapping, optional
        Environment to read; ``os.environ`` by default.

    Returns
    -------
    RunContext
        The context.  It reads the proposal file and checks that the other input
        files of the dataset exist.

    Raises
    ------
    RuntimeError
        If an installed package is older than ``requirements.txt`` asks for (see
        :func:`check_requirements`).
    ValueError
        If ``DTT_GUARD`` has an invalid value.
    FileNotFoundError
        If an input file of the dataset is missing.
    """
    env = os.environ if env is None else env
    seeds = dict(SEEDS)
    root_path = find_root(root)
    check_requirements(root_path)
    if str(root_path / "src") not in sys.path:
        sys.path.insert(0, str(root_path / "src"))

    raw_dir = root_path / "data" / "raw" / "wdi"
    processed_dir = root_path / "data" / "processed"
    catalogue_path = processed_dir / "cases_catalogue.csv"
    investment_fx_path = processed_dir / "cases_investment_fx.csv"
    baseline_path = processed_dir / "thailand_baseline.csv"
    proposal_path = processed_dir / "thailand_proposal.json"
    results_dir = Path(env.get(RESULTS_VARIABLE) or root_path / "results")
    figures_dir = Path(env.get(FIGURES_VARIABLE) or root_path / "figures")
    target_iso3, target_name = "THA", "Thailand"
    required = [raw_dir / "country_metadata.csv", catalogue_path, investment_fx_path, baseline_path, proposal_path]
    missing = [p for p in required if not p.is_file()]
    if missing:
        message = "input files missing: " + ", ".join(_under_root(p, root_path) for p in missing)
        if investment_fx_path in missing:
            message += f". The file {investment_fx_path.name} holds the derived dollar costs of the cases"
        raise FileNotFoundError(message)
    guard_on = _flag(env, GUARD_VARIABLE, True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    guard: Callable[[], None] | None = None
    guard_object = None
    if guard_on:
        guard_object, guard = _build_guard(results_dir)

    apply_style()
    return RunContext(
        stage=stage,
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
