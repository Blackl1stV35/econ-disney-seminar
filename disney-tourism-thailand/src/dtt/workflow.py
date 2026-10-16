"""Run context shared by the analysis notebooks.

A notebook opens with ``ctx = setup_run("02_case_effects")``.  The context holds
the paths of the inputs and outputs, the target economy, the scenarios of the
Thailand proposal, the thermal guard and the seeds.  It saves tables, JSON
files and figures below the results and figures directories, and it reads them
back, checking them against the stamps described below.

The run reads the raw files of the repository and needs the environment variable
``DTT_RUN_REAL=1``.  Tables, JSON files and figures are written below the results
and figures directories of the repository, unless the environment variables
``DTT_RESULTS_DIR`` and ``DTT_FIGURE_DIR`` name other directories.

Environment variables
---------------------
DTT_RUN_REAL
    ``1`` allows a run on the dataset; any other value does not.
DTT_RESULTS_DIR, DTT_FIGURE_DIR
    Override the directories that receive tables and figures.
DTT_GUARD
    ``1`` switches the thermal guard on and ``0`` switches it off; the default is on.

Printed paths
-------------
Messages name a file by its path from the repository root.  A file that lies
outside the repository, which is the case when ``DTT_RESULTS_DIR`` or
``DTT_FIGURE_DIR`` names another directory, is named by its path from the
results, figures or raw-data directory, and a file outside all of them by its
name alone.  No message holds
an absolute path, so the stored output of an executed notebook carries no
directory of the machine.

Input stamps
------------
A stage that records a file writes a stamp, ``<results directory>/stamps/<stage>.json``.
The stamp lists the files that the stage read (``inputs``) and the files that it
wrote (``outputs``), each with the SHA-256 digest of its bytes.  An input also
names the stage that produced the file, or ``null`` for a file that no stage
produced, such as a raw data file.  Paths in a stamp are relative: the
directories of the run are named ``results/``, ``figures/``, ``data/processed/``
and ``data/raw/wdi/`` whether they lie in the repository or outside it, and every
other path is relative to the repository root.  A stamp
holds no absolute path and no time, so two runs that read and write the same
bytes write the same stamp.

Before a stage reads a file that an earlier stage produced, the context checks
the file against the stamp of its producer.  The check continues through the
inputs that the producer recorded and through their producers, so that a change
at the start of a chain is found by a stage at its end, and so is a stage that
read a version of a file that its producer no longer records.  A file
in the results directory that no stamp lists as an output has no known
producer, and it cannot be read.  A file that no stage produced is recorded
with its digest; a later check requires that digest, but no producer is checked.

Notebooks write with ``RunContext.save_table``, ``save_json`` and
``save_figure``, which record their outputs.  They read with ``read_table``,
``read_json`` and ``require``, which record their inputs.  A file written in
another way is recorded with ``RunContext.record_output``, a file read in
another way with ``RunContext.record_input``, and the raw World Bank files of
:mod:`dtt.panel` with ``RunContext.record_raw_inputs``.  The stamp is written
each time a file is recorded.

A failed check raises :class:`StampError`, a ``ValueError``.  Its message names
the file, the producer stage and the two short digests (the first twelve
hexadecimal digits), or the stamp that is missing.

Public helpers are :func:`file_digest`, :func:`short_digest`, :func:`stamp_path`,
:func:`read_stamp` and :class:`StampError`.  Methods of :class:`RunContext` are
``record_input(*paths, producer=None)``, ``record_output(*paths)``,
``record_raw_inputs()``, ``write_stamp()``, ``stamp_key(path)`` and
``display_path(path)``; ``require(*names, producer=None)`` also checks the
stamps.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "REAL_VARIABLE",
    "SEEDS",
    "STAMP_DIRECTORY",
    "RunContext",
    "StampError",
    "apply_style",
    "file_digest",
    "find_root",
    "read_stamp",
    "resolve_mode",
    "setup_run",
    "short_digest",
    "stamp_path",
]

REAL_VARIABLE = "DTT_RUN_REAL"
RESULTS_VARIABLE = "DTT_RESULTS_DIR"
FIGURES_VARIABLE = "DTT_FIGURE_DIR"
GUARD_VARIABLE = "DTT_GUARD"

#: Name of the directory below the results directory that holds the stamps.
STAMP_DIRECTORY = "stamps"
#: Version of the layout of a stamp file.
STAMP_FORMAT = 1
#: Number of hexadecimal digits of a digest that messages show.
SHORT_DIGEST_LENGTH = 12

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

#: Episodes of the dataset catalogue with a known dollar cost, with and without the derived costs file.
_KNOWN_COSTS_WITH_FILE = 28
_KNOWN_COSTS_WITHOUT_FILE = 12
_EPISODE_COUNT = 44

_STAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RESULTS_AREA = "results"


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
    """Return the run mode, which is always ``"real"`` when the run is allowed.

    Parameters
    ----------
    env : mapping, optional
        Environment to read; ``os.environ`` by default.

    Returns
    -------
    str
        ``"real"``, returned when ``DTT_RUN_REAL`` is ``1``.

    Raises
    ------
    RuntimeError
        If the run is requested without ``DTT_RUN_REAL=1``.
    """
    env = os.environ if env is None else env
    if str(env.get(REAL_VARIABLE, "")).strip() == "1":
        return "real"
    raise RuntimeError(f"Set {REAL_VARIABLE}=1 to run the notebooks on the dataset.")


# ----------------------------------------------------------------------------
# Input stamps
# ----------------------------------------------------------------------------
def file_digest(path: str | Path) -> str:
    """Return the SHA-256 digest of a file as 64 hexadecimal digits.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read; it is read in blocks of one mebibyte.

    Returns
    -------
    str
        Lower-case hexadecimal digest of the bytes of the file.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def short_digest(digest: str | None) -> str:
    """Return the first twelve hexadecimal digits of a digest, or ``"none"`` when there is none."""
    return "none" if not digest else str(digest)[:SHORT_DIGEST_LENGTH]


def stamp_path(results_dir: str | Path, stage: str) -> Path:
    """Return the path of the stamp file of a stage: ``<results_dir>/stamps/<stage>.json``."""
    return Path(results_dir) / STAMP_DIRECTORY / f"{stage}.json"


def _parse_stamp(file: Path) -> dict[str, Any] | None:
    """Read a stamp file; return None if it cannot be read or does not have the layout of a stamp."""
    try:
        stamp = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stamp, dict) or stamp.get("format") != STAMP_FORMAT or stamp.get("stage") != file.stem:
        return None
    for table in ("inputs", "outputs"):
        records = stamp.get(table)
        if not isinstance(records, dict):
            return None
        for record in records.values():
            if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                return None
    return stamp


def read_stamp(results_dir: str | Path, stage: str) -> dict[str, Any] | None:
    """Read the stamp of a stage.

    Parameters
    ----------
    results_dir : str or pathlib.Path
        Results directory of the run.
    stage : str
        Name of the stage.

    Returns
    -------
    dict or None
        The stamp with the keys ``format``, ``stage``, ``inputs`` and
        ``outputs``, or None when the stage has no stamp file.

    Raises
    ------
    ValueError
        If the stamp file exists but does not have the layout of a stamp.
    """
    file = stamp_path(results_dir, stage)
    if not file.is_file():
        return None
    stamp = _parse_stamp(file)
    if stamp is None:
        raise ValueError(f"{STAMP_DIRECTORY}/{file.name} is not a valid stamp file")
    return stamp


def _stamp_message(
    kind: str, path: str, stage: str | None, consumer: str | None, expected: str | None, found: str | None
) -> str:
    """Return the text of a :class:`StampError`; see that class for the meaning of the fields."""
    if kind == "changed":
        return (
            f"{path} was changed after the stage {stage} wrote it: the stamp of {stage} records digest "
            f"{short_digest(expected)}, but the file has digest {short_digest(found)}. Run {stage} again."
        )
    if kind == "stale":
        return (
            f"The stage {consumer} read {path} with digest {short_digest(found)}, but the stage {stage}, which "
            f"produced it, records digest {short_digest(expected)}. Run {consumer} again."
        )
    if kind == "input_changed":
        return (
            f"{path} was changed after the stage {consumer} read it: the stamp of {consumer} records digest "
            f"{short_digest(expected)}, but the file has digest {short_digest(found)}. Run {consumer} again."
        )
    if kind == "missing_stamp":
        return (
            f"The stamp {STAMP_DIRECTORY}/{stage}.json of the stage {stage}, which produced {path}, is missing or "
            f"unreadable, so the file cannot be checked (file digest {short_digest(found)}). Run {stage} again."
        )
    if kind == "unlisted":
        return (
            f"The stamp of the stage {stage} does not list {path} as an output (file digest {short_digest(found)}). "
            f"Run {stage} again."
        )
    if kind == "unknown_producer":
        return (
            f"{path} lies in the results directory, but no stamp in {STAMP_DIRECTORY}/ lists it as an output "
            f"(file digest {short_digest(found)}). Run the stage that writes it again."
        )
    if kind == "missing_file":
        return f"{path} is listed in the stamp of the stage {stage or consumer} but does not exist. Run that stage again."
    raise ValueError(f"unknown stamp error kind {kind!r}")


class StampError(ValueError):
    """A file does not match the stamp of the stage that produced it, or that stamp is missing.

    Attributes
    ----------
    kind : str
        ``"changed"``: the file differs from the digest its producer recorded.
        ``"stale"``: a stage read a version of the file that its producer no
        longer records.  ``"input_changed"``: a file that no stage produced
        differs from the digest a stage recorded when it read it.
        ``"missing_stamp"``: the stamp of the producer named for the file is
        missing or unreadable.  ``"unlisted"``: the stamp of the named producer
        does not list the file as an output.  ``"unknown_producer"``: the file
        lies in the results directory and no stamp lists it.  ``"missing_file"``:
        a file that a stamp lists does not exist.
    path : str
        Relative path of the file, as written in the stamps.
    stage : str or None
        The producer stage, when one is known.
    consumer : str or None
        The stage that read the file.
    expected, found : str or None
        Full digests.  For ``"changed"`` and ``"input_changed"``, the digest in
        the stamp and the digest of the file.  For ``"stale"``, the digest that
        the producer records and the digest that the consumer read.  For the
        other kinds, the digest the consumer recorded, if any, and the digest of
        the file, if it exists.
    """

    def __init__(
        self,
        kind: str,
        path: str,
        stage: str | None = None,
        consumer: str | None = None,
        expected: str | None = None,
        found: str | None = None,
    ) -> None:
        super().__init__(_stamp_message(kind, path, stage, consumer, expected, found))
        self.kind = kind
        self.path = path
        self.stage = stage
        self.consumer = consumer
        self.expected = expected
        self.found = found


class _StampChecker:
    """Check files against the stamps of the stages that produced them.

    A checker reads the stamps of a results directory once, except the stamp of
    the running stage, and hashes every file at most once, so it serves one
    round of checks.
    """

    def __init__(self, stamp_dir: Path, locate: Callable[[str], Path], exclude: str) -> None:
        self._locate = locate
        self._stamps: dict[str, dict[str, Any]] = {}
        self._digests: dict[str, str | None] = {}
        self._done: set[str] = set()
        if stamp_dir.is_dir():
            for file in sorted(stamp_dir.glob("*.json")):
                if file.stem == exclude:
                    continue
                stamp = _parse_stamp(file)
                if stamp is not None:
                    self._stamps[file.stem] = stamp

    def digest(self, key: str) -> str | None:
        """Return the digest of the file with this stamp path, or None if the file does not exist."""
        if key not in self._digests:
            path = self._locate(key)
            self._digests[key] = file_digest(path) if path.is_file() else None
        return self._digests[key]

    def verify(
        self,
        key: str,
        *,
        recorded: str | None = None,
        hint: str | None = None,
        declared: str | None = None,
        consumer: str | None = None,
    ) -> str | None:
        """Check one file and, through the inputs of its producer, the chain behind it.

        Parameters
        ----------
        key : str
            Stamp path of the file.
        recorded : str, optional
            Digest that the consumer recorded when it read the file.
        hint : str, optional
            Producer that an earlier stamp recorded for the file.
        declared : str, optional
            Producer that the caller requires; it takes precedence over ``hint``.
        consumer : str, optional
            Stage that reads the file, for the messages.

        Returns
        -------
        str or None
            The producer stage, or None for a file that no stage produced.

        Raises
        ------
        StampError
            If the file or a file behind it does not match the stamps.
        """
        found = self.digest(key)
        named = declared or hint
        if found is None:
            raise StampError("missing_file", key, stage=named, consumer=consumer, expected=recorded)
        owners = [
            (stage, self._stamps[stage]["outputs"][key]["sha256"])
            for stage in sorted(self._stamps)
            if key in self._stamps[stage]["outputs"]
        ]
        if named is not None:
            if named not in self._stamps:
                raise StampError("missing_stamp", key, stage=named, consumer=consumer, expected=recorded, found=found)
            owners = [owner for owner in owners if owner[0] == named]
            if not owners:
                raise StampError("unlisted", key, stage=named, consumer=consumer, expected=recorded, found=found)
        if not owners:
            if key.startswith(_RESULTS_AREA + "/"):
                raise StampError("unknown_producer", key, consumer=consumer, expected=recorded, found=found)
            if recorded is not None and recorded != found:
                raise StampError("input_changed", key, consumer=consumer, expected=recorded, found=found)
            return None
        matching = [owner for owner in owners if owner[1] == found]
        stage, expected = (matching or owners)[0]
        if found != expected:
            raise StampError("changed", key, stage=stage, consumer=consumer, expected=expected, found=found)
        if recorded is not None and recorded != expected:
            raise StampError("stale", key, stage=stage, consumer=consumer, expected=expected, found=recorded)
        self._verify_inputs_of(stage)
        return stage

    def _verify_inputs_of(self, stage: str) -> None:
        """Check every input recorded in the stamp of ``stage``, once per checker."""
        if stage in self._done:
            return
        self._done.add(stage)
        for key, record in sorted(self._stamps[stage]["inputs"].items()):
            self.verify(key, recorded=record["sha256"], hint=record.get("producer"), consumer=stage)


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
        started = self._clock()
        if self._last is not None and started - self._last < self._min_interval:
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
        Always ``"real"``, the mode of the run on the dataset.
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
        One row per figure saved by this run; saving a figure again under the same
        name replaces its row.
    guard_object : object or None
        The thermal guard that ``guard`` calls.
    """

    stage: str
    mode: str
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
    _inputs: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _outputs: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def is_real(self) -> bool:
        """True for a run on the dataset."""
        return self.mode == "real"

    def primary_scenario(self) -> dict[str, Any]:
        """Return the proposal scenario that is flagged as primary."""
        for scenario in self.proposal["scenarios"]:
            if scenario.get("primary"):
                return dict(scenario)
        return dict(self.proposal["scenarios"][0])

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def display_path(self, path: str | Path) -> str:
        """Return the path as the messages of the run print it.

        Parameters
        ----------
        path : str or pathlib.Path
            Any path.

        Returns
        -------
        str
            The path from the repository root when the file lies in the
            repository; otherwise the path from the results, figures or raw-data
            directory of the run, whichever contains it first in that order; and
            the file name alone when no directory of the run contains it.  The
            separator is always ``/``.
        """
        resolved = Path(path).resolve()
        for base in (self.root, self.results_dir, self.figures_dir, self.raw_dir):
            if _inside(resolved, base):
                return resolved.relative_to(base.resolve()).as_posix()
        return resolved.name

    def _where(self, path: Path, outside: str) -> str:
        """Return a location for ``describe``: the repository-relative path, or ``outside`` for other locations."""
        resolved = Path(path).resolve()
        if _inside(resolved, self.root):
            return resolved.relative_to(self.root.resolve()).as_posix() or "."
        return outside

    def _areas(self) -> list[tuple[str, Path]]:
        """Directories of the run with the prefix that names them in a stamp."""
        return [
            (_RESULTS_AREA, self.results_dir),
            ("figures", self.figures_dir),
            ("data/processed", self.processed_dir),
            ("data/raw/wdi", self.raw_dir),
        ]

    def stamp_key(self, path: str | Path) -> str:
        """Return the relative path under which a stamp names a file.

        Parameters
        ----------
        path : str or pathlib.Path
            File of the run.

        Returns
        -------
        str
            ``results/...``, ``figures/...``, ``data/processed/...`` or
            ``data/raw/wdi/...`` for a file below the corresponding directory of
            the run (the innermost directory wins), and otherwise the path from the
            repository root.  The separator is always ``/``.

        Raises
        ------
        ValueError
            If the file lies outside the repository and the directories of the
            run, or lies in the repository at a place that the name of one of the
            directories of the run would also describe.
        """
        resolved = Path(path).resolve()
        best: tuple[str, Path] | None = None
        for prefix, base in self._areas():
            base = base.resolve()
            if _inside(resolved, base) and (best is None or len(base.parts) > len(best[1].parts)):
                best = (prefix, base)
        if best is not None:
            return f"{best[0]}/{resolved.relative_to(best[1]).as_posix()}"
        if _inside(resolved, self.root):
            key = resolved.relative_to(self.root.resolve()).as_posix()
            if any(key.startswith(prefix + "/") for prefix, _ in self._areas()):
                raise ValueError(
                    f"{key} cannot be told apart from a file in the {key.split('/')[0]} directory of this run, "
                    "which lies elsewhere"
                )
            return key
        raise ValueError(f"{resolved.name} lies outside the repository and the directories of this run")

    def _locate(self, key: str) -> Path:
        """Return the path of the file that a stamp names ``key``."""
        for prefix, base in sorted(self._areas(), key=lambda area: -len(area[0])):
            if key.startswith(prefix + "/"):
                return base / key[len(prefix) + 1 :]
        return self.root / key

    def table_path(self, name: str) -> Path:
        """Return the path of a table below the results directory."""
        return self.results_dir / (name if Path(name).suffix else f"{name}.csv")

    # ------------------------------------------------------------------
    # Stamps
    # ------------------------------------------------------------------
    def _checker(self) -> _StampChecker:
        """Return a checker of the stamps of the other stages of this run."""
        return _StampChecker(self.results_dir / STAMP_DIRECTORY, self._locate, exclude=self.stage)

    def record_input(self, *paths: str | Path | None, producer: str | None = None) -> list[str]:
        """Check files against their producers' stamps and record them as inputs of this stage.

        Parameters
        ----------
        *paths : str, pathlib.Path or None
            Files that the stage reads.  None is skipped, and so is a file that
            this stage has already written in this run.
        producer : str, optional
            Stage that must have written the files.

        Returns
        -------
        list of str
            The stamp paths of the files that were recorded.

        Raises
        ------
        FileNotFoundError
            If a file does not exist.
        StampError
            If a file does not match the stamp of its producer, directly or
            through the inputs the producer recorded, or the stamp of the producer
            is missing.  Nothing is recorded for the files after the failing one.
        ValueError
            If a file lies outside the repository and the directories of the run.
        """
        checker: _StampChecker | None = None
        recorded: list[str] = []
        for item in paths:
            if item is None:
                continue
            path = Path(item)
            key = self.stamp_key(path)
            if key in self._outputs:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"{self.display_path(path)} not found")
            checker = checker or self._checker()
            origin = checker.verify(key, declared=producer, consumer=self.stage)
            self._inputs[key] = {"producer": origin, "sha256": checker.digest(key)}
            recorded.append(key)
        if recorded:
            self.write_stamp()
        return recorded

    def record_output(self, *paths: str | Path) -> list[str]:
        """Record files that this stage wrote and write the stamp.

        Parameters
        ----------
        *paths : str or pathlib.Path
            Files that the stage has written; their digests are computed when the call is made.  A
            file that the stage had read is recorded as an output only.

        Returns
        -------
        list of str
            The stamp paths of the files.

        Raises
        ------
        FileNotFoundError
            If a file does not exist.
        ValueError
            If a file lies outside the repository and the directories of the run.
        """
        keys = []
        for item in paths:
            path = Path(item)
            key = self.stamp_key(path)
            if not path.is_file():
                raise FileNotFoundError(f"{self.display_path(path)} not found")
            self._inputs.pop(key, None)
            self._outputs[key] = file_digest(path)
            keys.append(key)
        if keys:
            self.write_stamp()
        return keys

    def record_raw_inputs(self) -> list[str]:
        """Record the twenty indicator files and the metadata file that :mod:`dtt.panel` reads.

        Returns
        -------
        list of str
            The stamp paths of the files that were recorded.
        """
        from dtt import panel as panel_mod

        names = [panel_mod.wdi_filename(group, code) for group in panel_mod.GROUPS for code in panel_mod.INDICATORS]
        return self.record_input(*(self.raw_dir / name for name in [*names, "country_metadata.csv"]))

    def write_stamp(self) -> Path:
        """Write the stamp of this stage and return its path.

        The stamp lists the inputs and outputs recorded in this run.  It is
        written to a temporary file that then replaces the old stamp, so that a
        stamp is never half written.

        Raises
        ------
        ValueError
            If a recorded path is absolute or leaves the directory of the run.
        """
        inputs = {key: dict(record) for key, record in sorted(self._inputs.items())}
        outputs = {key: {"sha256": digest} for key, digest in sorted(self._outputs.items())}
        for key in (*inputs, *outputs):
            if key.startswith("/") or ".." in key.split("/") or re.match(r"^[A-Za-z]:", key):
                raise ValueError(f"a stamp holds relative paths only, not {key!r}")
        payload = {"format": STAMP_FORMAT, "stage": self.stage, "inputs": inputs, "outputs": outputs}
        path = stamp_path(self.results_dir, self.stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.replace(temporary, path)
        return path

    # ------------------------------------------------------------------
    # Tables, JSON files and figures
    # ------------------------------------------------------------------
    def save_table(self, df: pd.DataFrame, name: str, index: bool = False) -> Path:
        """Write a table to the results directory, record it in the stamp and return its path."""
        path = self.table_path(name)
        self.stamp_key(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)
        self.record_output(path)
        print(f"Table saved: {self.display_path(path)}")
        return path

    def read_table(self, name: str, **kwargs: Any) -> pd.DataFrame:
        """Read a table from the results directory after checking it against its producer's stamp.

        Floating-point numbers are read with ``float_precision="round_trip"``, so
        that a table written by one notebook is read back bit for bit; the
        keyword arguments can override it.

        Parameters
        ----------
        name : str
            Table name; ``.csv`` is added when the name has no extension.
        **kwargs
            Passed to :func:`pandas.read_csv`.

        Raises
        ------
        FileNotFoundError
            If the table has not been written by an earlier notebook.
        StampError
            If the table, or a file behind it, does not match the stamps.
        """
        path = self.table_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{self.display_path(path)} not found; run the notebook that writes it first")
        self.record_input(path)
        if "float_precision" not in kwargs and kwargs.get("engine") in (None, "c"):
            kwargs["float_precision"] = "round_trip"
        return pd.read_csv(path, **kwargs)

    def require(self, *names: str, producer: str | None = None) -> None:
        """Check that earlier notebooks have written the named tables and that their stamps match.

        Parameters
        ----------
        *names : str
            Table or JSON file names in the results directory.
        producer : str, optional
            Stage that must have written the files.

        Raises
        ------
        FileNotFoundError
            Naming every missing file.
        StampError
            If a file does not match the stamp of its producer (see
            :meth:`record_input`).
        """
        paths = [self.table_path(n) for n in names]
        missing = [self.display_path(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing results of earlier notebooks: " + ", ".join(missing))
        self.record_input(*paths, producer=producer)

    def save_json(self, obj: Any, name: str) -> Path:
        """Write a JSON file to the results directory, record it in the stamp and return its path."""
        path = self.results_dir / (name if Path(name).suffix else f"{name}.json")
        self.stamp_key(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_clean_for_json(obj), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        self.record_output(path)
        print(f"JSON saved: {self.display_path(path)}")
        return path

    def read_json(self, name: str) -> Any:
        """Read a JSON file from the results directory after checking it against its producer's stamp."""
        path = self.results_dir / (name if Path(name).suffix else f"{name}.json")
        if not path.is_file():
            raise FileNotFoundError(f"{self.display_path(path)} not found; run the notebook that writes it first")
        self.record_input(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def save_figure(self, fig: Any, name: str, show: bool = True) -> Path:
        """Save a matplotlib figure to the figures directory.

        The figure inventory of the stage (panels, axis labels, legend entries)
        is rewritten to ``figure_inventory_<stage>.csv`` in the results
        directory.  Saving a figure under a name that the inventory already holds
        replaces its row, so that running a cell again leaves one row per figure.
        """
        import matplotlib.pyplot as plt

        path = self.figures_dir / name
        self.stamp_key(path)
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
        for position, existing in enumerate(self.figure_log):
            if existing["file"] == name:
                self.figure_log[position] = row
                break
        else:
            self.figure_log.append(row)
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
        inventory = self.results_dir / f"figure_inventory_{self.stage}.csv"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.figure_log).to_csv(inventory, index=False)
        self.record_output(path, inventory)
        print(f"Figure saved: {self.display_path(path)}")
        return path

    def describe(self) -> None:
        """Print the stage, locations, package versions and the machine headroom.

        Locations inside the repository are printed from the repository root;
        locations outside it are described in words.
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
        print(f"Raw files:   {self._where(self.raw_dir, 'the raw-data directory, outside the repository')}")
        print(f"Catalogue:   {self._where(self.catalogue_path, 'the cases catalogue, outside the repository')}")
        print(f"Results:     {self._where(self.results_dir, 'the results directory of the run, outside the repository')}")
        print(f"Figures:     {self._where(self.figures_dir, 'the figures directory of the run, outside the repository')}")
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


def setup_run(
    stage: str,
    mode: str | None = None,
    root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    guard_options: Mapping[str, Any] | None = None,
) -> RunContext:
    """Create the run context of a notebook.

    Parameters
    ----------
    stage : str
        Name of the notebook, used in the file names of the figure inventory and
        of the stamp.  It starts with a letter or digit and holds only letters,
        digits, dots, underscores and hyphens.
    mode : {"real"}, optional
        The only mode of the notebooks.  The run still needs ``DTT_RUN_REAL=1``.
    root : str or pathlib.Path, optional
        Repository root; found from the current directory by default.
    env : mapping, optional
        Environment to read; ``os.environ`` by default.
    guard_options : mapping, optional
        Keyword arguments of :class:`dtt.thermal.ThermalGuard`, for example
        readers that replace the sensors.  The log file is always
        ``results/thermal_log.csv`` unless ``log_path`` is given here.

    Returns
    -------
    RunContext
        The context.  The run reads only the proposal file here and checks that
        the other input files exist.

    Raises
    ------
    RuntimeError
        If the run is requested without ``DTT_RUN_REAL=1``.
    ValueError
        If ``stage``, ``mode`` or a switch variable has an invalid value.
    FileNotFoundError
        If an input file of the run is missing.  The file of derived dollar
        costs ``data/processed/cases_investment_fx.csv`` is required: without it
        the episodes with a known cost fall from 28 to 12 of 44.
    """
    if not isinstance(stage, str) or _STAGE_PATTERN.fullmatch(stage) is None:
        raise ValueError(
            f"stage must start with a letter or digit and hold only letters, digits, dots, underscores and hyphens, not {stage!r}"
        )
    env = os.environ if env is None else env
    if mode is not None and mode != "real":
        raise ValueError("mode must be 'real', the only mode of the notebooks")
    mode = resolve_mode(env)
    seeds = dict(SEEDS)
    root_path = find_root(root)
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
    missing = [p.relative_to(root_path).as_posix() for p in required if not p.is_file()]
    if missing:
        message = "input files missing: " + ", ".join(missing)
        if investment_fx_path.relative_to(root_path).as_posix() in missing:
            message += (
                f". The file {investment_fx_path.relative_to(root_path).as_posix()} holds the derived dollar costs "
                f"of the cases; without it the episodes with a known cost fall from {_KNOWN_COSTS_WITH_FILE} to "
                f"{_KNOWN_COSTS_WITHOUT_FILE} of {_EPISODE_COUNT}"
            )
        raise FileNotFoundError(message)
    guard_on = _flag(env, GUARD_VARIABLE, True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    guard: Callable[[], None] | None = None
    guard_object = None
    if guard_on:
        from dtt.thermal import ThermalGuard

        options = dict(guard_options or {})
        options.setdefault("log_path", results_dir / "thermal_log.csv")
        guard_object = ThermalGuard(**options)
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
