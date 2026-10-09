"""Tests for dtt.thermal.

Guard tests use fake temperature and CPU readers, a fake sleep and a fake clock,
so that no test waits in real time; the memory share in the log rows is read from
the host.  The load that does not belong to the program is tested with substitute
``psutil`` load readings and substitute process objects.  Probe tests use
substitutes for ``psutil``, for sysfs files in a temporary directory, for
``subprocess.run`` and for the probe functions, and one test runs a child
process of the current interpreter that prints undecodable bytes.  Four tests
run against the host: the default readers of a guard, the reader of the load of
other processes, ``read_temperature_c`` and ``headroom``.
"""
import ast
import csv
import errno
import inspect
import math
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dtt import thermal  # noqa: E402
from dtt.thermal import TemperatureReader, ThermalGuard, ThermalTimeout  # noqa: E402

PROBE_FUNCTIONS = {
    "psutil": "_probe_psutil",
    "sysfs": "_probe_sysfs",
    "windows_acpi": "_probe_windows_acpi",
    "windows_hardware_monitor": "_probe_windows_hardware_monitor",
}


class FakeTime:
    """Fake clock and sleep that advance together."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def sequence(values):
    """Reader that returns the given values in turn and then repeats the last one."""
    state = {"i": 0}

    def read():
        i = min(state["i"], len(values) - 1)
        state["i"] += 1
        value = values[i]
        if isinstance(value, Exception):
            raise value
        return value

    return read


def make_guard(temps, cpus=(10.0,), fake=None, **kwargs):
    fake = fake or FakeTime()
    guard = ThermalGuard(
        read_temp=sequence(list(temps)),
        read_cpu=sequence(list(cpus)),
        sleep=fake.sleep,
        clock=fake.clock,
        **kwargs,
    )
    return guard, fake


def read_log(path):
    """Rows of a CSV log as dictionaries, with the file closed afterwards."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def header_count(path):
    """Number of header rows in a CSV log."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.startswith("timestamp,"))


class StubProbes:
    """Substitutes for the four temperature probes with settable results.

    A result is a number, ``None`` or an exception that the probe raises.  The
    names of the probes that ran are recorded in ``order``.
    """

    def __init__(self, monkeypatch, **results):
        self.results = {name: None for name in PROBE_FUNCTIONS}
        self.results.update(results)
        self.order: list[str] = []
        for name, attribute in PROBE_FUNCTIONS.items():
            monkeypatch.setattr(thermal, attribute, self._probe(name))

    def _probe(self, name):
        def run():
            self.order.append(name)
            value = self.results[name]
            if isinstance(value, Exception):
                raise value
            return value

        return run


class RunRecorder:
    """Substitute for subprocess.run that records its calls."""

    def __init__(self, outputs=None, exception=None, returncode=0):
        self.outputs = outputs or {}
        self.exception = exception
        self.returncode = returncode
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exception is not None:
            raise self.exception
        command = args[-1]
        for key, text in self.outputs.items():
            if key in command:
                return subprocess.CompletedProcess(args, self.returncode, stdout=text, stderr="")
        return subprocess.CompletedProcess(args, self.returncode, stdout="", stderr="")


def write_zone(root, name, zone_type, millidegrees):
    """Create a sysfs-style thermal zone directory; a type of None writes no type file."""
    zone = root / name
    zone.mkdir()
    (zone / "temp").write_text(f"{millidegrees}\n", encoding="ascii")
    if zone_type is not None:
        (zone / "type").write_text(f"{zone_type}\n", encoding="ascii")


def zone_pattern(root):
    return str(root / "thermal_zone*" / "temp")


def entries(*values):
    return [SimpleNamespace(label="x", current=v, high=None, critical=None) for v in values]


# ----------------------------------------------------------------------------
# Guard: temperature branch
# ----------------------------------------------------------------------------
def test_no_wait_when_the_temperature_is_within_limits():
    guard, fake = make_guard([60.0])
    guard()
    guard.check()
    assert fake.sleeps == []
    summary = guard.summary()
    assert summary["checks"] == 2 and summary["waits"] == 0 and summary["waited_seconds"] == 0.0
    assert summary["max_temperature_c"] == 60.0
    assert summary["temperature_source"] == "custom"


@pytest.mark.parametrize("temperature", [85.0, 80.0, 78.0, 0.5])
def test_values_up_to_the_maximum_do_not_wait(temperature):
    guard, fake = make_guard([temperature])
    guard()
    assert fake.sleeps == []


def test_waits_in_poll_steps_until_the_resume_temperature():
    guard, fake = make_guard([90.0, 88.0, 80.0, 78.0], poll_seconds=5.0)
    guard()
    assert fake.sleeps == [5.0, 5.0, 5.0]
    summary = guard.summary()
    assert summary["checks"] == 1 and summary["waits"] == 1
    assert summary["waited_seconds"] == pytest.approx(15.0)
    assert summary["max_temperature_c"] == 90.0


def test_the_wait_continues_while_just_above_the_resume_temperature():
    guard, fake = make_guard([86.0, 78.01, 78.01, 77.9], poll_seconds=2.0)
    guard()
    assert fake.sleeps == [2.0, 2.0, 2.0]


def test_a_custom_resume_and_maximum_are_respected():
    guard, fake = make_guard([70.0, 66.0, 59.9], max_temp_c=65.0, resume_temp_c=60.0, poll_seconds=1.0)
    guard()
    assert fake.sleeps == [1.0, 1.0]


def test_timeout_raises_after_the_maximum_wait_and_logs_the_row(tmp_path):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([95.0], poll_seconds=5.0, max_wait_seconds=20.0, log_path=log)
    with pytest.raises(ThermalTimeout):
        guard()
    assert fake.sleeps == [5.0] * 4
    rows = read_log(log)
    assert len(rows) == 1
    assert rows[0]["action"] == "timeout" and float(rows[0]["waited_seconds"]) == pytest.approx(20.0)
    summary = guard.summary()
    assert summary["checks"] == 1 and summary["waits"] == 1 and summary["waited_seconds"] == pytest.approx(20.0)


def test_timeout_exception_is_a_runtime_error():
    assert issubclass(ThermalTimeout, RuntimeError)


def test_zero_maximum_wait_times_out_without_sleeping():
    guard, fake = make_guard([95.0], max_wait_seconds=0.0)
    with pytest.raises(ThermalTimeout):
        guard()
    assert fake.sleeps == []


def test_a_fake_sleep_with_a_real_clock_still_times_out():
    slept = []
    guard = ThermalGuard(
        read_temp=lambda: 99.0,
        read_cpu=lambda: 10.0,
        sleep=slept.append,
        clock=time.monotonic,
        poll_seconds=5.0,
        max_wait_seconds=30.0,
    )
    with pytest.raises(ThermalTimeout):
        guard()
    assert slept == [5.0] * 6


def test_the_guard_works_again_after_a_timeout():
    guard, fake = make_guard([95.0, 95.0, 95.0, 95.0, 50.0], poll_seconds=5.0, max_wait_seconds=10.0)
    with pytest.raises(ThermalTimeout):
        guard()
    guard()
    assert guard.summary()["checks"] == 2


@pytest.mark.parametrize(
    "max_wait, poll, expected",
    [
        (7.0, 5.0, [5.0, 2.0]),
        (3.0, 5.0, [3.0]),
        (10.0, 5.0, [5.0, 5.0]),
        (30.0, 4.0, [4.0] * 7 + [2.0]),
        (1.0, 5.0, [1.0]),
    ],
)
def test_the_last_sleep_is_clipped_to_the_remaining_time(tmp_path, max_wait, poll, expected):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([95.0], poll_seconds=poll, max_wait_seconds=max_wait, log_path=log)
    with pytest.raises(ThermalTimeout) as caught:
        guard()
    assert fake.sleeps == expected
    assert sum(fake.sleeps) == pytest.approx(max_wait)
    assert guard.summary()["waited_seconds"] == pytest.approx(max_wait)
    assert float(read_log(log)[0]["waited_seconds"]) == pytest.approx(max_wait)
    assert f"for {max_wait:g} seconds" in str(caught.value)


def test_a_wait_that_ends_before_the_limit_is_not_clipped():
    guard, fake = make_guard([95.0, 90.0, 70.0], poll_seconds=5.0, max_wait_seconds=12.0)
    guard()
    assert fake.sleeps == [5.0, 5.0]


# ----------------------------------------------------------------------------
# Guard: missing temperature and CPU branch
# ----------------------------------------------------------------------------
def test_a_missing_reading_during_the_wait_does_not_end_it():
    guard, fake = make_guard([95.0, None, None, 70.0], cpus=[3.0], poll_seconds=5.0)
    guard()
    assert fake.sleeps == [5.0, 5.0, 5.0]
    assert guard.summary()["waited_seconds"] == pytest.approx(15.0)


def test_a_missing_reading_between_two_hot_readings_does_not_end_the_wait():
    guard, fake = make_guard([95.0, None, 95.0, 77.0], cpus=[3.0], poll_seconds=5.0)
    guard()
    assert fake.sleeps == [5.0, 5.0, 5.0]


def test_a_temperature_that_does_not_return_during_the_wait_ends_in_a_timeout(tmp_path):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([95.0, None], cpus=[3.0], poll_seconds=5.0, max_wait_seconds=20.0, log_path=log)
    with pytest.raises(ThermalTimeout) as caught:
        guard()
    assert fake.sleeps == [5.0] * 4
    assert "temperature" in str(caught.value) and "CPU" not in str(caught.value)
    row = read_log(log)[0]
    assert row["action"] == "timeout" and row["temperature_c"] == "95.00"


def test_cpu_fallback_sleeps_one_step_when_the_load_drops_after_it():
    guard, fake = make_guard([None], cpus=[95.0, 50.0], poll_seconds=7.0)
    guard()
    assert fake.sleeps == [7.0]
    summary = guard.summary()
    assert summary["waits"] == 1 and summary["waited_seconds"] == pytest.approx(7.0)
    assert summary["temperature_source"] == "unavailable" and summary["max_temperature_c"] is None


def test_a_missing_reading_at_the_first_check_waits_while_the_load_is_above_the_limit():
    guard, fake = make_guard([None], cpus=[99.0, 99.0, 99.0, 50.0], poll_seconds=4.0)
    guard()
    assert fake.sleeps == [4.0, 4.0, 4.0]
    summary = guard.summary()
    assert summary["checks"] == 1 and summary["waits"] == 1
    assert summary["waited_seconds"] == pytest.approx(12.0)


def test_a_machine_that_stays_busy_without_a_temperature_does_not_raise_whatever_the_heat_limit(tmp_path):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([None], cpus=[99.0], poll_seconds=5.0, max_wait_seconds=20.0, log_path=log)
    guard()
    assert fake.sleeps == [5.0] * 60
    row = read_log(log)[0]
    assert row["action"] == "cpu_timeout" and row["temperature_c"] == "" and row["cpu_percent"] == "99.0"
    assert float(row["waited_seconds"]) == pytest.approx(300.0)


@pytest.mark.parametrize("cpu", [92.0, 50.0, 0.0])
def test_cpu_fallback_does_not_wait_at_or_below_the_limit(cpu):
    guard, fake = make_guard([None], cpus=[cpu])
    guard()
    assert fake.sleeps == []


def test_the_cpu_limit_follows_the_configuration():
    guard, fake = make_guard([None], cpus=[60.0, 40.0], max_cpu_percent=50.0, poll_seconds=3.0)
    guard()
    assert fake.sleeps == [3.0]


def test_the_cpu_load_is_ignored_when_a_temperature_is_available():
    guard, fake = make_guard([60.0], cpus=[99.9])
    guard()
    assert fake.sleeps == []


def test_an_unreadable_cpu_load_counts_as_within_the_limit():
    guard, fake = make_guard([None], cpus=[RuntimeError("load")])
    guard()
    assert fake.sleeps == []
    guard, fake = make_guard([None], cpus=[99.0, RuntimeError("load")], poll_seconds=2.0)
    guard()
    assert fake.sleeps == [2.0]


def test_a_temperature_that_appears_during_the_cpu_wait_ends_it_when_within_the_maximum():
    guard, fake = make_guard([None, None, 60.0], cpus=[99.0], poll_seconds=2.0)
    guard()
    assert fake.sleeps == [2.0, 2.0]
    assert guard.summary()["max_temperature_c"] == 60.0


def test_a_hot_temperature_that_appears_during_the_cpu_wait_needs_the_resume_temperature():
    guard, fake = make_guard([None, 90.0, 80.0, 77.0], cpus=[99.0], poll_seconds=2.0)
    guard()
    assert fake.sleeps == [2.0, 2.0, 2.0]
    guard, fake = make_guard([None, 90.0], cpus=[99.0], poll_seconds=5.0, max_wait_seconds=10.0)
    with pytest.raises(ThermalTimeout) as caught:
        guard()
    assert "temperature" in str(caught.value) and "CPU" not in str(caught.value)


# ----------------------------------------------------------------------------
# Guard: a wait for CPU load alone ends without an exception
# ----------------------------------------------------------------------------
def test_a_wait_for_cpu_load_alone_returns_at_the_cap_and_logs_cpu_timeout(tmp_path):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([None], cpus=[99.0], poll_seconds=5.0, max_cpu_wait_seconds=20.0, log_path=log)
    guard()
    assert fake.sleeps == [5.0] * 4
    rows = read_log(log)
    assert len(rows) == 1 and rows[0]["action"] == "cpu_timeout"
    assert float(rows[0]["waited_seconds"]) == pytest.approx(20.0)
    assert rows[0]["temperature_c"] == "" and rows[0]["cpu_percent"] == "99.0"
    summary = guard.summary()
    assert summary["checks"] == 1 and summary["waits"] == 1 and summary["cpu_timeouts"] == 1
    assert summary["waited_seconds"] == pytest.approx(20.0)


def test_the_default_cap_of_a_cpu_wait_is_300_seconds_and_the_wait_does_not_raise():
    parameters = inspect.signature(ThermalGuard).parameters
    assert parameters["max_cpu_wait_seconds"].default == 300.0
    assert parameters["max_wait_seconds"].default == 1800.0
    guard, fake = make_guard([None], cpus=[99.0], poll_seconds=5.0)
    guard()
    assert fake.sleeps == [5.0] * 60 and guard.summary()["cpu_timeouts"] == 1


def test_the_next_call_after_a_cpu_timeout_does_not_wait_for_cpu_load(tmp_path):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([None], cpus=[99.0], poll_seconds=5.0, max_cpu_wait_seconds=10.0, log_path=log)
    guard()
    assert fake.sleeps == [5.0, 5.0]
    guard()
    guard()
    assert fake.sleeps == [5.0, 5.0]
    rows = read_log(log)
    assert [row["action"] for row in rows] == ["cpu_timeout", "ok", "ok"]
    assert [row["cpu_percent"] for row in rows] == ["99.0", "99.0", "99.0"]
    assert [float(row["waited_seconds"]) for row in rows] == [pytest.approx(10.0), 0.0, 0.0]
    summary = guard.summary()
    assert summary["checks"] == 3 and summary["waits"] == 1 and summary["cpu_timeouts"] == 1


def test_a_wait_for_cpu_load_that_ends_before_the_cap_does_not_stop_later_waits():
    guard, fake = make_guard([None], cpus=[99.0, 50.0, 99.0, 50.0], poll_seconds=3.0, max_cpu_wait_seconds=30.0)
    guard()
    guard()
    assert fake.sleeps == [3.0, 3.0]
    summary = guard.summary()
    assert summary["waits"] == 2 and summary["cpu_timeouts"] == 0


def test_the_guard_still_waits_for_heat_after_a_cpu_timeout():
    guard, fake = make_guard([None] * 3 + [95.0], cpus=[99.0], poll_seconds=5.0, max_wait_seconds=30.0,
                             max_cpu_wait_seconds=10.0)
    guard()
    assert fake.sleeps == [5.0, 5.0]
    with pytest.raises(ThermalTimeout) as caught:
        guard()
    assert fake.sleeps == [5.0, 5.0] + [5.0] * 6
    assert "temperature" in str(caught.value) and "CPU" not in str(caught.value)
    assert guard.summary()["cpu_timeouts"] == 1


def test_a_wait_for_heat_keeps_its_own_longer_limit(tmp_path):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([95.0], poll_seconds=5.0, max_wait_seconds=40.0, max_cpu_wait_seconds=10.0, log_path=log)
    with pytest.raises(ThermalTimeout):
        guard()
    assert fake.sleeps == [5.0] * 8
    assert read_log(log)[0]["action"] == "timeout"
    assert guard.summary()["cpu_timeouts"] == 0


def test_a_cpu_wait_that_turns_hot_is_limited_by_the_maximum_wait_and_raises():
    guard, fake = make_guard([None, None, 90.0], cpus=[99.0], poll_seconds=5.0, max_wait_seconds=40.0,
                             max_cpu_wait_seconds=10.0)
    with pytest.raises(ThermalTimeout) as caught:
        guard()
    assert fake.sleeps == [5.0] * 8
    assert "temperature" in str(caught.value) and "CPU" not in str(caught.value)
    assert guard.summary()["cpu_timeouts"] == 0


def test_the_last_sleep_of_a_cpu_wait_is_clipped_to_the_remaining_time():
    guard, fake = make_guard([None], cpus=[99.0], poll_seconds=5.0, max_cpu_wait_seconds=12.0)
    guard()
    assert fake.sleeps == [5.0, 5.0, 2.0]
    assert guard.summary()["waited_seconds"] == pytest.approx(12.0)


def test_the_maximum_wait_for_heat_does_not_limit_a_wait_for_cpu_load():
    guard, fake = make_guard([None], cpus=[99.0], poll_seconds=5.0, max_wait_seconds=10.0, max_cpu_wait_seconds=25.0)
    guard()
    assert fake.sleeps == [5.0] * 5 and guard.summary()["cpu_timeouts"] == 1


def test_a_cpu_cap_of_zero_ends_a_cpu_wait_without_sleeping():
    guard, fake = make_guard([None], cpus=[99.0], max_cpu_wait_seconds=0.0)
    guard()
    assert fake.sleeps == [] and guard.summary()["cpu_timeouts"] == 1


def test_a_cpu_wait_with_a_real_clock_and_a_fake_sleep_ends_at_the_cap():
    slept = []
    guard = ThermalGuard(
        read_temp=lambda: None,
        read_cpu=lambda: 99.0,
        sleep=slept.append,
        clock=time.monotonic,
        poll_seconds=5.0,
        max_cpu_wait_seconds=30.0,
        verbose=False,
    )
    guard()
    assert slept == [5.0] * 6 and guard.summary()["cpu_timeouts"] == 1


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_an_invalid_cpu_wait_cap_raises(bad):
    with pytest.raises(ValueError, match="max_cpu_wait_seconds"):
        ThermalGuard(max_cpu_wait_seconds=bad)


def test_the_new_options_are_keyword_only_with_their_documented_defaults():
    parameters = inspect.signature(ThermalGuard).parameters
    for name in ("max_cpu_wait_seconds", "verbose"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["verbose"].default is True
    assert list(parameters)[:10] == [
        "max_temp_c", "resume_temp_c", "max_cpu_percent", "poll_seconds", "max_wait_seconds", "log_path", "read_temp",
        "read_cpu", "sleep", "clock",
    ]
    assert ThermalGuard(max_cpu_wait_seconds=0).max_cpu_wait_seconds == 0.0


def test_the_summary_reports_the_cpu_timeouts_next_to_the_other_totals():
    guard, _ = make_guard([60.0])
    assert guard.summary() == {
        "checks": 0, "waits": 0, "waited_seconds": 0.0, "cpu_timeouts": 0, "max_temperature_c": None,
        "temperature_source": "unavailable",
    }


# ----------------------------------------------------------------------------
# Guard: progress lines
# ----------------------------------------------------------------------------
def lines_of(capsys):
    """Lines printed to standard output since the last call."""
    return capsys.readouterr().out.splitlines()


def test_nothing_is_printed_when_the_machine_is_within_its_limits(capsys):
    guard, _ = make_guard([60.0, None], cpus=[10.0])
    guard()
    guard()
    assert lines_of(capsys) == []


def test_a_wait_for_heat_prints_one_line_at_the_start_and_one_at_the_end(capsys):
    guard, _ = make_guard([90.0, 88.0, 70.0], poll_seconds=5.0)
    guard()
    assert lines_of(capsys) == [
        "thermal guard: waiting, the CPU temperature is 90.0 C, above the limit of 85 C",
        "thermal guard: resumed after waiting 10.0 seconds",
    ]


def test_a_wait_for_cpu_load_prints_the_load_and_the_waiting_time(capsys):
    guard, _ = make_guard([None], cpus=[97.4, 97.4, 20.0], poll_seconds=4.0)
    guard()
    assert lines_of(capsys) == [
        "thermal guard: waiting, the CPU load is 97 percent, above the limit of 92 percent (no temperature available)",
        "thermal guard: resumed after waiting 8.0 seconds",
    ]


def test_a_cpu_timeout_prints_the_start_and_one_closing_line_and_says_that_it_stops_waiting(capsys):
    guard, _ = make_guard([None], cpus=[99.0], poll_seconds=5.0, max_cpu_wait_seconds=20.0)
    guard()
    first = lines_of(capsys)
    assert len(first) == 2 and first[0].startswith("thermal guard: waiting, the CPU load is 99 percent")
    assert "for 20.0 seconds" in first[1] and "not waiting for the CPU load again" in first[1]
    guard()
    assert lines_of(capsys) == []


def test_a_timeout_after_a_wait_for_heat_prints_its_closing_line_before_the_exception(capsys):
    guard, _ = make_guard([95.0], poll_seconds=5.0, max_wait_seconds=10.0)
    with pytest.raises(ThermalTimeout):
        guard()
    printed = lines_of(capsys)
    assert len(printed) == 2 and printed[0].startswith("thermal guard: waiting, the CPU temperature is 95.0 C")
    assert "gave up after waiting 10.0 seconds" in printed[1]


def test_a_silent_guard_prints_nothing_and_behaves_in_the_same_way(capsys, tmp_path):
    log = tmp_path / "log.csv"
    guard, fake = make_guard([90.0, 70.0], poll_seconds=5.0, verbose=False, log_path=log)
    guard()
    guard, fake_cpu = make_guard([None], cpus=[99.0], poll_seconds=5.0, max_cpu_wait_seconds=10.0, verbose=False)
    guard()
    assert lines_of(capsys) == []
    assert fake.sleeps == [5.0] and fake_cpu.sleeps == [5.0, 5.0]
    assert read_log(log)[0]["action"] == "waited"


def test_a_failing_output_stream_does_not_stop_the_guard(monkeypatch):
    class Broken:
        def write(self, text):
            raise OSError("closed")

        def flush(self):
            raise OSError("closed")

    monkeypatch.setattr(sys, "stdout", Broken())
    guard, fake = make_guard([90.0, 70.0], poll_seconds=5.0)
    guard()
    assert fake.sleeps == [5.0] and guard.summary()["waits"] == 1


# ----------------------------------------------------------------------------
# Guard: the load that does not belong to this program
# ----------------------------------------------------------------------------
class FakeProcess:
    """Substitute for a ``psutil.Process`` with a script of load readings.

    The first reading of every object is 0.0, like the first reading of a
    ``psutil`` process; the next ones follow ``loads`` and the last one repeats.
    From the reading number ``fail_from`` on, ``failure`` is raised instead when given.
    ``children`` is a list of process descriptions that the object returns as
    new ``FakeProcess`` objects at every call, as ``psutil`` does.
    """

    def __init__(self, pid, loads=(0.0,), children=(), failure=None, children_failure=None, fail_from=1):
        self.pid = pid
        self._loads = list(loads)
        self._children = list(children)
        self._failure = failure
        self._fail_from = fail_from
        self._children_failure = children_failure
        self._calls = 0

    def cpu_percent(self, interval=None):
        assert interval is None
        self._calls += 1
        if self._failure is not None and self._calls >= self._fail_from:
            raise self._failure
        if self._calls == 1:
            return 0.0
        return self._loads[min(self._calls - 2, len(self._loads) - 1)]

    def children(self, recursive=False):
        assert recursive is True
        if self._children_failure is not None:
            raise self._children_failure
        return [FakeProcess(**description) for description in self._children]


def patch_psutil(monkeypatch, system=80.0, cpus=4, process=None, process_failure=None):
    """Substitute the load readings of ``psutil`` and the process object it creates."""
    loads = system if isinstance(system, list) else [system]
    state = {"i": 0}

    def cpu_percent(interval=None):
        assert interval is None
        value = loads[min(state["i"], len(loads) - 1)]
        state["i"] += 1
        return value

    def make_process():
        if process_failure is not None:
            raise process_failure
        return process

    monkeypatch.setattr(thermal.psutil, "cpu_percent", cpu_percent)
    monkeypatch.setattr(thermal.psutil, "cpu_count", lambda logical=True: cpus)
    monkeypatch.setattr(thermal.psutil, "Process", make_process)


def test_the_load_of_the_process_tree_is_subtracted_from_the_load_of_the_system(monkeypatch):
    child = {"pid": 2, "loads": [60.0, 20.0]}
    process = FakeProcess(1, loads=[100.0], children=[child])
    patch_psutil(monkeypatch, system=[0.0, 80.0, 80.0, 80.0], cpus=4, process=process)
    read = thermal._default_cpu_reader()
    assert read() == pytest.approx(80.0 - 100.0 / 4.0)
    assert read() == pytest.approx(80.0 - (100.0 + 60.0) / 4.0)
    assert read() == pytest.approx(80.0 - (100.0 + 20.0) / 4.0)


def test_a_child_that_is_seen_for_the_first_time_adds_nothing_and_a_known_child_is_not_forgotten(monkeypatch):
    process = FakeProcess(1, loads=[40.0], children=[{"pid": 2, "loads": [80.0]}, {"pid": 3, "loads": [120.0]}])
    patch_psutil(monkeypatch, system=[0.0, 90.0, 90.0], cpus=8, process=process)
    read = thermal._default_cpu_reader()
    assert read() == pytest.approx(90.0 - 40.0 / 8.0)
    assert read() == pytest.approx(90.0 - (40.0 + 80.0 + 120.0) / 8.0)
    assert sorted(read._children) == [2, 3]


def test_the_load_of_other_processes_is_never_negative(monkeypatch):
    process = FakeProcess(1, loads=[400.0])
    patch_psutil(monkeypatch, system=[0.0, 30.0], cpus=4, process=process)
    assert thermal._default_cpu_reader()() == 0.0


def test_a_child_that_cannot_be_read_is_skipped_and_forgotten(monkeypatch):
    gone = {"pid": 5, "failure": psutil.NoSuchProcess(5)}
    denied = {"pid": 6, "failure": psutil.AccessDenied(6)}
    alive = {"pid": 7, "loads": [60.0]}
    process = FakeProcess(1, loads=[20.0], children=[gone, denied, alive])
    patch_psutil(monkeypatch, system=[0.0, 70.0, 70.0], cpus=2, process=process)
    read = thermal._default_cpu_reader()
    assert read() == pytest.approx(70.0 - 20.0 / 2.0)
    assert read() == pytest.approx(70.0 - (20.0 + 60.0) / 2.0)
    assert sorted(read._children) == [7]


@pytest.mark.parametrize(
    "kind", ["no process", "process load", "children", "no cpu count"],
)
def test_the_load_of_the_system_is_used_when_the_process_cannot_be_read(monkeypatch, kind):
    process = FakeProcess(1, loads=[100.0], children=[{"pid": 2, "loads": [100.0]}])
    cpus = 4
    process_failure = None
    if kind == "no process":
        process_failure = psutil.AccessDenied(1)
    elif kind == "process load":
        process = FakeProcess(1, loads=[100.0], failure=OSError("denied"), fail_from=2)
    elif kind == "children":
        process = FakeProcess(1, loads=[100.0], children_failure=psutil.AccessDenied(1))
    else:
        cpus = None
    patch_psutil(monkeypatch, system=[0.0, 66.0, 66.0], cpus=cpus, process=process, process_failure=process_failure)
    read = thermal._default_cpu_reader()
    assert read() == 66.0
    assert read() == 66.0


def test_a_guard_does_not_wait_for_the_load_that_it_causes_itself(monkeypatch, tmp_path):
    log = tmp_path / "log.csv"
    process = FakeProcess(1, loads=[390.0])
    patch_psutil(monkeypatch, system=[0.0, 99.0], cpus=4, process=process)
    fake = FakeTime()
    guard = ThermalGuard(read_temp=lambda: None, sleep=fake.sleep, clock=fake.clock, log_path=log, verbose=False)
    guard()
    assert fake.sleeps == []
    row = read_log(log)[0]
    assert row["action"] == "ok" and float(row["cpu_percent"]) == pytest.approx(99.0 - 390.0 / 4.0, abs=0.05)


def test_a_guard_waits_for_the_load_of_other_programs(monkeypatch, tmp_path):
    log = tmp_path / "log.csv"
    process = FakeProcess(1, loads=[8.0])
    patch_psutil(monkeypatch, system=[0.0, 99.0, 99.0, 60.0], cpus=4, process=process)
    fake = FakeTime()
    guard = ThermalGuard(
        read_temp=lambda: None, sleep=fake.sleep, clock=fake.clock, log_path=log, poll_seconds=5.0, verbose=False
    )
    guard()
    assert fake.sleeps == [5.0, 5.0]
    row = read_log(log)[0]
    assert row["action"] == "waited" and float(row["cpu_percent"]) == pytest.approx(99.0 - 8.0 / 4.0, abs=0.05)


def test_the_default_cpu_reader_is_the_process_tree_reader_and_an_explicit_reader_is_used_as_given():
    assert isinstance(ThermalGuard()._read_cpu, thermal._OtherProcessesLoad)
    reader = lambda: 12.0  # noqa: E731
    assert ThermalGuard(read_cpu=reader)._read_cpu is reader


def test_the_real_process_tree_reader_returns_a_load_between_zero_and_one_hundred():
    read = thermal._default_cpu_reader()
    for _ in range(3):
        value = read()
        assert isinstance(value, float) and 0.0 <= value <= 100.0
    own = read.own_load()
    assert own is None or own >= 0.0


# ----------------------------------------------------------------------------
# Guard: robustness, validation, iteration
# ----------------------------------------------------------------------------
def test_failing_readers_are_treated_as_unavailable(tmp_path):
    log = tmp_path / "log.csv"
    guard = ThermalGuard(
        read_temp=sequence([RuntimeError("sensor")]),
        read_cpu=sequence([ValueError("load")]),
        sleep=FakeTime().sleep,
        clock=FakeTime().clock,
        log_path=log,
    )
    guard()
    row = read_log(log)[0]
    assert row["temperature_c"] == "" and row["cpu_percent"] == "" and row["action"] == "ok"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_temperatures_count_as_unavailable(bad):
    guard, fake = make_guard([bad], cpus=[10.0])
    guard()
    assert guard.summary()["max_temperature_c"] is None and fake.sleeps == []


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        ThermalGuard(max_temp_c=70.0, resume_temp_c=75.0)
    with pytest.raises(ValueError):
        ThermalGuard(poll_seconds=0.0)
    with pytest.raises(ValueError):
        ThermalGuard(max_wait_seconds=-1.0)


def test_guarded_calls_the_guard_before_each_item():
    events = []

    def reader():
        events.append("check")
        return 50.0

    guard = ThermalGuard(read_temp=reader, read_cpu=lambda: 1.0, sleep=FakeTime().sleep, clock=FakeTime().clock)
    for item in guard.guarded(range(3)):
        events.append(f"use {item}")
    assert events == ["check", "use 0", "check", "use 1", "check", "use 2"]
    assert list(guard.guarded([])) == []
    assert guard.summary()["checks"] == 3


def test_guarded_propagates_a_timeout_and_stops_the_loop():
    temps = [50.0, 50.0, 99.0, 99.0, 99.0, 99.0]
    guard, _ = make_guard(temps, poll_seconds=5.0, max_wait_seconds=5.0)
    seen = []
    with pytest.raises(ThermalTimeout):
        for item in guard.guarded(range(10)):
            seen.append(item)
    assert seen == [0, 1]


@pytest.mark.parametrize("explicit", [False, True])
def test_default_reader_reports_the_probe_name(monkeypatch, explicit):
    StubProbes(monkeypatch, sysfs=61.0)
    fake = FakeTime()
    extra = {"read_temp": thermal.read_temperature_c} if explicit else {}
    guard = ThermalGuard(read_cpu=lambda: 5.0, sleep=fake.sleep, clock=fake.clock, **extra)
    guard()
    summary = guard.summary()
    assert summary["temperature_source"] == "sysfs" and summary["max_temperature_c"] == 61.0


def test_the_default_readers_of_a_guard_run_on_the_host():
    fake = FakeTime()
    guard = ThermalGuard(sleep=fake.sleep, clock=fake.clock, max_wait_seconds=10.0)
    guard()
    summary = guard.summary()
    assert summary["checks"] == 1
    sources = {"unavailable", "psutil", "sysfs", "windows_acpi", "windows_hardware_monitor"}
    assert summary["temperature_source"] in sources


# ----------------------------------------------------------------------------
# Guard: CSV log
# ----------------------------------------------------------------------------
def test_log_has_one_header_and_one_row_per_call(tmp_path):
    log = tmp_path / "nested" / "dir" / "thermal.csv"
    guard, _ = make_guard([60.0, 90.0, 70.0, None], cpus=[12.5, 30.0, 40.0, 20.0], log_path=log, poll_seconds=5.0)
    guard()
    guard()
    guard()
    with log.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == ["timestamp", "temperature_c", "cpu_percent", "mem_percent", "action", "waited_seconds"]
    rows = read_log(log)
    assert [r["action"] for r in rows] == ["ok", "waited", "ok"]
    assert rows[0]["temperature_c"] == "60.00" and rows[0]["cpu_percent"] == "12.5"
    assert rows[1]["temperature_c"] == "90.00" and float(rows[1]["waited_seconds"]) == pytest.approx(5.0)
    assert rows[2]["temperature_c"] == "" and float(rows[2]["waited_seconds"]) == 0.0
    for r in rows:
        datetime.fromisoformat(r["timestamp"])
        assert 0.0 <= float(r["mem_percent"]) <= 100.0


def test_a_second_guard_appends_without_a_second_header(tmp_path):
    log = tmp_path / "thermal.csv"
    first, _ = make_guard([60.0], log_path=log)
    first()
    second, _ = make_guard([61.0], log_path=log)
    second()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("timestamp,")


def test_two_guards_that_both_find_no_log_write_one_header(tmp_path, monkeypatch):
    log = tmp_path / "log.csv"
    first, _ = make_guard([60.0], log_path=log)
    second, _ = make_guard([61.0], log_path=log)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self, *a, **k: False if self == log else real_exists(self, *a, **k))
    first()
    second()
    monkeypatch.undo()
    assert header_count(log) == 1
    assert [row["temperature_c"] for row in read_log(log)] == ["60.00", "61.00"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["log.csv"]


def test_a_guard_that_loses_the_race_to_create_the_log_writes_no_header(tmp_path, monkeypatch):
    log = tmp_path / "log.csv"
    first, _ = make_guard([60.0], log_path=log)
    second, _ = make_guard([61.0], log_path=log)
    real_link = os.link

    def link_after_the_other_guard(source, destination, **kwargs):
        monkeypatch.setattr(thermal.os, "link", real_link)
        second()
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(thermal.os, "link", link_after_the_other_guard)
    first()
    monkeypatch.undo()
    assert header_count(log) == 1
    assert sorted(row["temperature_c"] for row in read_log(log)) == ["60.00", "61.00"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["log.csv"]


def test_guards_of_one_process_never_collide_on_the_name_of_the_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr(thermal.time, "time_ns", lambda: 123456789)
    monkeypatch.setattr(thermal.os, "getpid", lambda: 4242)
    log = tmp_path / "log.csv"
    first, _ = make_guard([60.0], log_path=log)
    second, _ = make_guard([61.0], log_path=log)
    real_link = os.link

    def link_after_the_other_guard(source, destination, **kwargs):
        monkeypatch.setattr(thermal.os, "link", real_link)
        second()
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(thermal.os, "link", link_after_the_other_guard)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first()
    monkeypatch.undo()
    assert header_count(log) == 1
    assert sorted(row["temperature_c"] for row in read_log(log)) == ["60.00", "61.00"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["log.csv"]


def test_the_header_is_in_the_file_before_any_row_can_follow_it(tmp_path, monkeypatch):
    log = tmp_path / "log.csv"
    guard, _ = make_guard([60.0], log_path=log)
    real_link = os.link
    seen = []

    def link_and_inspect(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        seen.append(Path(destination).read_text(encoding="utf-8"))

    monkeypatch.setattr(thermal.os, "link", link_and_inspect)
    guard()
    assert seen == ["timestamp,temperature_c,cpu_percent,mem_percent,action,waited_seconds\n"]


def test_without_hard_links_the_header_is_still_written_once(tmp_path, monkeypatch):
    def no_links(*args, **kwargs):
        raise PermissionError(errno.EPERM, "hard links are not supported")

    monkeypatch.setattr(thermal.os, "link", no_links)
    log = tmp_path / "log.csv"
    first, _ = make_guard([60.0], log_path=log)
    second, _ = make_guard([61.0], log_path=log)
    first()
    second()
    assert header_count(log) == 1 and len(read_log(log)) == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["log.csv"]


def test_an_empty_log_file_receives_the_header(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("", encoding="utf-8")
    guard, _ = make_guard([60.0], log_path=log)
    guard()
    assert header_count(log) == 1 and len(read_log(log)) == 1


def test_a_log_file_that_is_removed_during_the_run_starts_again_with_a_header(tmp_path):
    log = tmp_path / "log.csv"
    guard, _ = make_guard([60.0, 61.0], log_path=log)
    guard()
    log.unlink()
    guard()
    assert header_count(log) == 1 and [row["temperature_c"] for row in read_log(log)] == ["61.00"]


def test_no_file_is_written_without_a_log_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    guard, _ = make_guard([60.0])
    guard()
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_log_warns_once_and_does_not_stop_the_run(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    guard, _ = make_guard([60.0], log_path=blocker / "log.csv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        guard()
        guard()
    assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 1
    assert guard.summary()["checks"] == 2


# ----------------------------------------------------------------------------
# Probes: psutil and sysfs
# ----------------------------------------------------------------------------
def test_psutil_probe_takes_the_hottest_cpu_sensor(monkeypatch):
    sensors = {
        "coretemp": entries(50.0, 62.5),
        "k10temp": entries(40.0),
        "acpitz": entries(35.0),
        "nvme": entries(99.0),
    }
    monkeypatch.setattr(thermal.psutil, "sensors_temperatures", lambda: sensors, raising=False)
    assert thermal._probe_psutil() == 62.5
    monkeypatch.setattr(thermal.psutil, "sensors_temperatures", lambda: {"cpu_thermal": entries(48.0)}, raising=False)
    assert thermal._probe_psutil() == 48.0


def test_psutil_probe_returns_none_without_usable_readings(monkeypatch):
    unusable = {"coretemp": entries(0.0, -3.0, 9.9, float("nan"), 400.0, None)}
    for sensors in ({}, {"nvme": entries(70.0)}, unusable):
        monkeypatch.setattr(thermal.psutil, "sensors_temperatures", lambda s=sensors: s, raising=False)
        assert thermal._probe_psutil() is None


def test_psutil_probe_never_raises(monkeypatch):
    def broken():
        raise OSError("no access")

    monkeypatch.setattr(thermal.psutil, "sensors_temperatures", broken, raising=False)
    assert thermal._probe_psutil() is None
    monkeypatch.delattr(thermal.psutil, "sensors_temperatures", raising=False)
    assert thermal._probe_psutil() is None


def test_sysfs_probe_reads_millidegrees_and_skips_bad_files(tmp_path):
    for index, value in enumerate((45000, 61500, "garbage", -5000)):
        write_zone(tmp_path, f"thermal_zone{index}", "x86_pkg_temp", value)
    (tmp_path / "thermal_zone4").mkdir()
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) == pytest.approx(61.5)


def test_sysfs_probe_returns_none_when_nothing_matches(tmp_path):
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) is None


def test_sysfs_probe_ignores_battery_and_wireless_zones(tmp_path):
    write_zone(tmp_path, "thermal_zone0", "x86_pkg_temp", 52000)
    write_zone(tmp_path, "thermal_zone1", "iwlwifi_1", 99000)
    write_zone(tmp_path, "thermal_zone2", "BAT0", 31000)
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) == 52.0


def test_sysfs_probe_takes_the_maximum_over_the_selected_zones(tmp_path):
    write_zone(tmp_path, "thermal_zone0", "x86_pkg_temp", 52000)
    write_zone(tmp_path, "thermal_zone1", "acpitz", 61500)
    write_zone(tmp_path, "thermal_zone2", "coretemp", 47000)
    write_zone(tmp_path, "thermal_zone3", "nvme", 99000)
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) == pytest.approx(61.5)


@pytest.mark.parametrize(
    "zone_type",
    [
        "x86_pkg_temp", "coretemp", "k10temp", "cpu", "cpu-thermal", "cpu_thermal", "soc_thermal", "acpitz", "TCPU",
        "Tctl",
    ],
)
def test_zone_types_that_name_a_processor_sensor_are_used(tmp_path, zone_type):
    write_zone(tmp_path, "thermal_zone0", zone_type, 47000)
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) == 47.0
    assert thermal._is_cpu_zone_type(zone_type)


@pytest.mark.parametrize(
    "zone_type",
    ["iwlwifi_1", "BAT0", "nvme", "pch_skylake", "INT3400 Thermal", "gpu-thermal", "ddr-thermal", "charger", ""],
)
def test_other_zone_types_are_ignored(tmp_path, zone_type):
    write_zone(tmp_path, "thermal_zone0", zone_type, 90000)
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) is None
    assert not thermal._is_cpu_zone_type(zone_type)


def test_a_zone_without_a_type_file_is_ignored(tmp_path):
    write_zone(tmp_path, "thermal_zone0", None, 90000)
    write_zone(tmp_path, "thermal_zone1", "x86_pkg_temp", 40000)
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) == 40.0


def test_a_selected_zone_with_an_implausible_value_does_not_hide_the_others(tmp_path):
    write_zone(tmp_path, "thermal_zone0", "x86_pkg_temp", 0)
    write_zone(tmp_path, "thermal_zone1", "acpitz", 9000)
    write_zone(tmp_path, "thermal_zone2", "coretemp", 44000)
    assert thermal._probe_sysfs(zone_pattern(tmp_path)) == 44.0


# ----------------------------------------------------------------------------
# Probes: Windows commands
# ----------------------------------------------------------------------------
def test_windows_acpi_probe_converts_tenths_of_kelvin(monkeypatch):
    run = RunRecorder(outputs={"MSAcpi_ThermalZoneTemperature": "3000\r\n3231\r\n"})
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._probe_windows_acpi() == pytest.approx(3231 / 10.0 - 273.15)
    args, kwargs = run.calls[0]
    command = args[-1]
    assert args[0].lower() == "powershell"
    assert "Get-CimInstance" in command and "root/wmi" in command and "MSAcpi_ThermalZoneTemperature" in command
    assert kwargs["timeout"] == 5


def test_windows_probes_do_nothing_off_windows(monkeypatch):
    run = RunRecorder(exception=AssertionError("subprocess must not be called"))
    monkeypatch.setattr(thermal, "_is_windows", lambda: False)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._probe_windows_acpi() is None
    assert thermal._probe_windows_hardware_monitor() is None
    assert run.calls == []


@pytest.mark.parametrize(
    "run",
    [
        RunRecorder(exception=subprocess.TimeoutExpired(cmd="powershell", timeout=5)),
        RunRecorder(exception=FileNotFoundError("powershell")),
        RunRecorder(exception=PermissionError("denied")),
        RunRecorder(outputs={"": "3000"}, returncode=1),
        RunRecorder(outputs={"": "not a number"}),
        RunRecorder(outputs={"": ""}),
    ],
)
def test_windows_probes_return_none_on_failure(monkeypatch, run):
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._probe_windows_acpi() is None
    assert thermal._probe_windows_hardware_monitor() is None


def test_hardware_monitor_probe_queries_both_namespaces_in_order(monkeypatch):
    run = RunRecorder(outputs={"OpenHardwareMonitor": "51.5\r\n63,0\r\n"})
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._probe_windows_hardware_monitor() == pytest.approx(63.0)
    commands = [call[0][-1] for call in run.calls]
    assert "root/LibreHardwareMonitor" in commands[0] and "root/OpenHardwareMonitor" in commands[1]
    for command in commands:
        assert "-ClassName Sensor" in command and "'Temperature'" in command and "*CPU*" in command
    assert all(call[1]["timeout"] == 5 for call in run.calls)


def test_hardware_monitor_probe_stops_at_the_first_namespace_with_values(monkeypatch):
    run = RunRecorder(outputs={"LibreHardwareMonitor": "44.0"})
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._probe_windows_hardware_monitor() == 44.0
    assert len(run.calls) == 1


# ----------------------------------------------------------------------------
# Plausibility window and command output
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "values, expected",
    [
        ([10.0], 10.0),
        ([9.99], None),
        ([150.0], 150.0),
        ([150.01], None),
        ([0.05, 62.0], 62.0),
        ([5.0, 9.0], None),
        ([float("nan"), float("inf"), "x", None], None),
    ],
)
def test_readings_are_plausible_from_10_to_150_degrees(values, expected):
    assert thermal._hottest(values) == expected


def test_an_acpi_value_of_zero_degrees_celsius_is_not_a_temperature(monkeypatch):
    run = RunRecorder(outputs={"MSAcpi_ThermalZoneTemperature": "2732\r\n"})
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._probe_windows_acpi() is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3231\r\n", [3231.0]),
        ("51.5\r\nWARNING: 128 sensors enumerated\r\n", [51.5]),
        ("45 46\n", []),
        ("1,5\n", [1.5]),
        ("  +42.5  \n", [42.5]),
        ("12:30\nnan\ninf\n1e2\n0x10\n1,234.5\n", []),
        ("40\n\n41\n", [40.0, 41.0]),
        ("", []),
    ],
)
def test_only_lines_that_consist_of_one_number_are_readings(text, expected):
    assert thermal._numbers(text) == expected


def test_a_stray_line_with_numbers_is_not_taken_as_a_sensor(monkeypatch):
    run = RunRecorder(outputs={"LibreHardwareMonitor": "51.5\r\nWARNING: 128 sensors enumerated\r\n"})
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._probe_windows_hardware_monitor() == 51.5


def test_a_stray_line_in_the_output_does_not_make_the_guard_wait(monkeypatch):
    run = RunRecorder(outputs={"LibreHardwareMonitor": "51.5\r\nWARNING: 128 sensors enumerated\r\n"})
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    monkeypatch.setattr(thermal.psutil, "sensors_temperatures", lambda: {}, raising=False)
    monkeypatch.setattr(thermal, "_probe_sysfs", lambda: None)
    fake = FakeTime()
    guard = ThermalGuard(read_cpu=lambda: 1.0, sleep=fake.sleep, clock=fake.clock, max_wait_seconds=60.0)
    guard()
    assert fake.sleeps == []
    assert guard.summary()["max_temperature_c"] == 51.5
    assert guard.summary()["temperature_source"] == "windows_hardware_monitor"


def test_the_command_output_is_decoded_with_replacement_characters(monkeypatch):
    run = RunRecorder(outputs={"": "3231"})
    monkeypatch.setattr(thermal.subprocess, "run", run)
    assert thermal._powershell_output("any command") == "3231"
    assert run.calls[0][1]["errors"] == "replace" and run.calls[0][1]["text"] is True


def test_undecodable_bytes_in_the_output_do_not_hide_a_reading(monkeypatch):
    real_run = subprocess.run
    code = "import sys; sys.stdout.buffer.write(b'3231\\r\\n\\xff\\xfe\\r\\n')"

    def run_python(args, **kwargs):
        return real_run([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run_python)
    text = thermal._powershell_output("any command")
    assert text is not None and text.splitlines()[0] == "3231"
    assert thermal._probe_windows_acpi() == pytest.approx(3231 / 10.0 - 273.15)


# ----------------------------------------------------------------------------
# Probe order, remembered probe and retry period
# ----------------------------------------------------------------------------
def test_probe_order_and_fall_through(monkeypatch):
    probes = StubProbes(monkeypatch, sysfs=OSError("boom"), windows_acpi=55.5, windows_hardware_monitor=99.0)
    assert thermal._read_with_source() == (55.5, "windows_acpi")
    assert thermal.read_temperature_c() == 55.5
    assert probes.order[:3] == ["psutil", "sysfs", "windows_acpi"]
    assert "windows_hardware_monitor" not in probes.order


def test_first_probe_wins_and_none_is_returned_when_all_fail(monkeypatch):
    monkeypatch.setattr(thermal, "_probe_psutil", lambda: 47.0)
    monkeypatch.setattr(thermal, "_probe_sysfs", lambda: pytest.fail("later probes must not run"))
    assert thermal._read_with_source() == (47.0, "psutil")
    for name in ("_probe_psutil", "_probe_sysfs", "_probe_windows_acpi", "_probe_windows_hardware_monitor"):
        monkeypatch.setattr(thermal, name, lambda: None)
    assert thermal.read_temperature_c() is None and thermal._read_with_source() is None


def test_read_temperature_c_runs_the_probes_at_every_call(monkeypatch):
    probes = StubProbes(monkeypatch, sysfs=50.0)
    assert thermal.read_temperature_c() == 50.0
    assert thermal.read_temperature_c() == 50.0
    assert probes.order == ["psutil", "sysfs", "psutil", "sysfs"]


def test_a_probe_that_failed_is_not_run_again_for_sixty_seconds(monkeypatch):
    probes = StubProbes(monkeypatch)
    fake = FakeTime()
    reader = TemperatureReader(clock=fake.clock)
    assert reader.read() is None
    assert probes.order == list(PROBE_FUNCTIONS)
    fake.now += 59.5
    assert reader.read() is None
    assert probes.order == list(PROBE_FUNCTIONS)
    fake.now += 0.5
    assert reader.read() is None
    assert probes.order == list(PROBE_FUNCTIONS) * 2


def test_a_probe_that_raises_counts_as_failed(monkeypatch):
    probes = StubProbes(monkeypatch, psutil=OSError("denied"), sysfs=RuntimeError("broken"))
    fake = FakeTime()
    reader = TemperatureReader(clock=fake.clock)
    assert reader() is None
    assert reader() is None
    assert probes.order == list(PROBE_FUNCTIONS)


def test_the_probe_that_worked_is_tried_first(monkeypatch):
    probes = StubProbes(monkeypatch, sysfs=50.0)
    fake = FakeTime()
    reader = TemperatureReader(clock=fake.clock)
    assert reader.read() == (50.0, "sysfs")
    assert probes.order == ["psutil", "sysfs"]
    fake.now += 61.0
    assert reader.read() == (50.0, "sysfs")
    assert reader.read() == (50.0, "sysfs")
    assert probes.order == ["psutil", "sysfs", "sysfs", "sysfs"]


def test_another_probe_takes_over_when_the_remembered_one_fails(monkeypatch):
    probes = StubProbes(monkeypatch, sysfs=50.0)
    fake = FakeTime()
    reader = TemperatureReader(clock=fake.clock)
    assert reader.read() == (50.0, "sysfs")
    fake.now += 61.0
    probes.results["sysfs"] = None
    probes.results["psutil"] = 41.0
    probes.order.clear()
    assert reader.read() == (41.0, "psutil")
    assert probes.order == ["sysfs", "psutil"]
    probes.order.clear()
    fake.now += 1.0
    assert reader.read() == (41.0, "psutil")
    assert probes.order == ["psutil"]


def test_each_failed_probe_waits_for_its_own_retry_period(monkeypatch):
    probes = StubProbes(monkeypatch, sysfs=50.0)
    fake = FakeTime()
    reader = TemperatureReader(clock=fake.clock)
    assert reader.read() == (50.0, "sysfs")
    fake.now += 10.0
    probes.results["sysfs"] = None
    assert reader.read() is None
    probes.results["psutil"] = 38.0
    probes.order.clear()
    fake.now += 49.5
    assert reader.read() is None
    assert probes.order == []
    fake.now += 0.5
    assert reader.read() == (38.0, "psutil")
    assert probes.order == ["psutil"]


def test_a_retry_period_of_zero_runs_every_probe_at_every_read(monkeypatch):
    probes = StubProbes(monkeypatch)
    reader = TemperatureReader(clock=FakeTime().clock, retry_seconds=0.0)
    reader.read()
    reader.read()
    assert probes.order == list(PROBE_FUNCTIONS) * 2


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_an_invalid_retry_period_raises(bad):
    with pytest.raises(ValueError):
        TemperatureReader(retry_seconds=bad)


def test_the_reader_returns_the_temperature_without_the_source(monkeypatch):
    StubProbes(monkeypatch, psutil=44.5)
    reader = TemperatureReader(clock=FakeTime().clock)
    assert reader() == 44.5 and isinstance(reader(), float)


def test_a_guard_without_a_temperature_does_not_start_powershell_at_every_check(monkeypatch):
    run = RunRecorder()
    monkeypatch.setattr(thermal, "_is_windows", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", run)
    monkeypatch.setattr(thermal.psutil, "sensors_temperatures", lambda: {}, raising=False)
    monkeypatch.setattr(thermal, "_probe_sysfs", lambda: None)
    fake = FakeTime()
    guard = ThermalGuard(read_cpu=lambda: 1.0, sleep=fake.sleep, clock=fake.clock)
    for _ in range(10):
        guard()
        fake.now += 5.0
    assert len(run.calls) == 3
    fake.now += 60.0
    guard()
    assert len(run.calls) == 6


def test_a_guard_keeps_using_the_probe_that_worked(monkeypatch):
    probes = StubProbes(monkeypatch, windows_acpi=60.0)
    fake = FakeTime()
    guard = ThermalGuard(read_cpu=lambda: 1.0, sleep=fake.sleep, clock=fake.clock)
    for _ in range(4):
        guard()
        fake.now += 61.0
    assert probes.order == ["psutil", "sysfs", "windows_acpi", "windows_acpi", "windows_acpi", "windows_acpi"]
    assert guard.summary()["temperature_source"] == "windows_acpi"


def test_real_temperature_read_never_raises():
    value = thermal.read_temperature_c()
    assert value is None or (isinstance(value, float) and 10.0 <= value <= 150.0)


# ----------------------------------------------------------------------------
# Head room
# ----------------------------------------------------------------------------
HEADROOM_KEYS = ["cpu_percent", "mem_available_gb", "mem_percent", "temperature_c", "claude_processes"]
PSUTIL_ERRORS = [
    psutil.AccessDenied(),
    psutil.NoSuchProcess(4242),
    OSError("denied"),
    PermissionError("denied"),
    RuntimeError("broken"),
]


def answer(value):
    """Function that returns ``value`` or, when it is an exception, raises it."""

    def call(*args, **kwargs):
        if isinstance(value, Exception):
            raise value
        return value

    return call


def stub_headroom(monkeypatch, cpu=12.5, memory=None, processes=(), temperature=61.0):
    """Substitute the readings of headroom; a value that is an exception is raised by its reading."""
    if memory is None:
        memory = SimpleNamespace(available=8 * 2**30, percent=33.3)
    monkeypatch.setattr(thermal.psutil, "cpu_percent", answer(cpu))
    monkeypatch.setattr(thermal.psutil, "virtual_memory", answer(memory))
    monkeypatch.setattr(thermal, "read_temperature_c", answer(temperature))
    if isinstance(processes, Exception):
        monkeypatch.setattr(thermal.psutil, "process_iter", answer(processes))
    else:
        monkeypatch.setattr(thermal.psutil, "process_iter", lambda attrs=None: iter(processes))


def test_headroom_collects_the_documented_fields(monkeypatch):
    intervals = []
    processes = [
        SimpleNamespace(info={"name": "Claude"}),
        SimpleNamespace(info={"name": "claude-code"}),
        SimpleNamespace(info={"name": "node"}),
        SimpleNamespace(info={"name": "CLAUDE.exe"}),
        SimpleNamespace(info={"name": None}),
    ]
    stub_headroom(monkeypatch, processes=processes)
    monkeypatch.setattr(thermal.psutil, "cpu_percent", lambda interval=None: intervals.append(interval) or 12.5)
    info = thermal.headroom()
    assert intervals == [0.5]
    assert info["cpu_percent"] == 12.5
    assert info["mem_available_gb"] == pytest.approx(8.0)
    assert info["mem_percent"] == pytest.approx(33.3)
    assert info["temperature_c"] == 61.0
    assert info["claude_processes"] == {"count": 3, "names": ["CLAUDE.exe", "Claude", "claude-code"]}
    assert list(info) == HEADROOM_KEYS


def test_headroom_without_temperature_or_claude_processes(monkeypatch):
    stub_headroom(monkeypatch, cpu=3.0, processes=[SimpleNamespace(info={"name": "python"})], temperature=None)
    info = thermal.headroom()
    assert info["temperature_c"] is None
    assert info["claude_processes"] == {"count": 0, "names": []}


@pytest.mark.parametrize("error", PSUTIL_ERRORS, ids=lambda error: type(error).__name__)
def test_headroom_reports_none_for_a_cpu_reading_that_raises(monkeypatch, error):
    stub_headroom(monkeypatch, cpu=error)
    info = thermal.headroom()
    assert info["cpu_percent"] is None
    assert info["mem_available_gb"] == pytest.approx(8.0) and info["mem_percent"] == pytest.approx(33.3)
    assert info["temperature_c"] == 61.0 and info["claude_processes"] == {"count": 0, "names": []}


@pytest.mark.parametrize("error", PSUTIL_ERRORS, ids=lambda error: type(error).__name__)
def test_headroom_reports_none_for_a_memory_reading_that_raises(monkeypatch, error):
    stub_headroom(monkeypatch, memory=error)
    info = thermal.headroom()
    assert info["mem_available_gb"] is None and info["mem_percent"] is None
    assert info["cpu_percent"] == 12.5 and info["temperature_c"] == 61.0


@pytest.mark.parametrize("error", PSUTIL_ERRORS, ids=lambda error: type(error).__name__)
def test_headroom_reports_none_when_the_process_list_cannot_be_read(monkeypatch, error):
    stub_headroom(monkeypatch, processes=error)
    info = thermal.headroom()
    assert info["claude_processes"] is None
    assert info["cpu_percent"] == 12.5 and info["mem_percent"] == pytest.approx(33.3)


@pytest.mark.parametrize("error", PSUTIL_ERRORS, ids=lambda error: type(error).__name__)
def test_headroom_reports_none_for_a_temperature_that_raises(monkeypatch, error):
    stub_headroom(monkeypatch, temperature=error)
    info = thermal.headroom()
    assert info["temperature_c"] is None and info["cpu_percent"] == 12.5


def test_headroom_skips_a_process_whose_name_cannot_be_read(monkeypatch):
    class Unreadable:
        @property
        def info(self):
            raise psutil.AccessDenied()

    processes = [Unreadable(), SimpleNamespace(info={"name": "claude"}), SimpleNamespace(info=None)]
    stub_headroom(monkeypatch, processes=processes)
    assert thermal.headroom()["claude_processes"] == {"count": 1, "names": ["claude"]}


def test_headroom_reports_none_when_the_process_iterator_fails_midway(monkeypatch):
    def processes(attrs=None):
        yield SimpleNamespace(info={"name": "claude"})
        raise psutil.AccessDenied()

    stub_headroom(monkeypatch)
    monkeypatch.setattr(thermal.psutil, "process_iter", processes)
    assert thermal.headroom()["claude_processes"] is None


def test_headroom_reports_none_for_every_entry_when_everything_fails(monkeypatch):
    error = psutil.AccessDenied()
    stub_headroom(monkeypatch, cpu=error, memory=error, processes=error, temperature=error)
    info = thermal.headroom()
    assert list(info) == HEADROOM_KEYS and all(value is None for value in info.values())


def test_real_headroom_has_sane_values(monkeypatch):
    monkeypatch.setattr(thermal.psutil, "cpu_percent", lambda interval=None: 7.0)
    info = thermal.headroom()
    assert info["cpu_percent"] == 7.0
    assert info["mem_available_gb"] > 0.0 and 0.0 <= info["mem_percent"] <= 100.0
    assert info["temperature_c"] is None or math.isfinite(info["temperature_c"])
    assert info["claude_processes"]["count"] == len(info["claude_processes"]["names"])
    assert all("claude" in name.lower() for name in info["claude_processes"]["names"])


# ----------------------------------------------------------------------------
# Hygiene of the module
# ----------------------------------------------------------------------------
def unused_parameters(module):
    """Parameters of the functions of a module that are never read, as ``(function, parameter)`` pairs."""
    tree = ast.parse(inspect.getsource(module))
    unused = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            names = [a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs]
            names += [a.arg for a in (arguments.vararg, arguments.kwarg) if a is not None]
            read = {
                n.id for statement in node.body for n in ast.walk(statement)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            unused += [(node.name, name) for name in names if name not in read and name not in ("self", "cls")]
    return unused


def test_every_parameter_of_every_function_of_the_module_is_read():
    assert unused_parameters(thermal) == []
