"""Host temperature probes and a throttle for long computations.

``read_temperature_c`` returns the hottest CPU temperature where the platform
exposes it.  ``TemperatureReader`` does the same and remembers which probe
worked.  ``headroom`` summarises the free capacity of the machine.
``ThermalGuard`` pauses a computation while the machine is too hot, or, when no
temperature is available, too busy, and records every check in a CSV file.

Temperature probes, tried in this order until one returns a value
-----------------------------------------------------------------
``psutil``
    ``psutil.sensors_temperatures()`` entries named ``coretemp``, ``k10temp``,
    ``cpu_thermal`` and ``acpitz``.
``sysfs``
    ``/sys/class/thermal/thermal_zone*/temp`` in millidegrees Celsius, for the
    zones whose ``type`` file names a processor, package or core sensor: a type
    that contains ``cpu``, ``core``, ``pkg``, ``package``, ``processor``, ``soc``,
    ``k10temp``, ``tctl``, ``tdie`` or ``acpitz`` (case-insensitive).  Zones of
    other types, such as batteries, wireless cards and storage devices, and zones
    without a readable type are ignored.  The hottest selected zone is used.
``windows_acpi``
    On Windows, the PowerShell command ``Get-CimInstance -Namespace root/wmi
    -ClassName MSAcpi_ThermalZoneTemperature``; ``CurrentTemperature`` is given in
    tenths of Kelvin.
``windows_hardware_monitor``
    On Windows, the temperature sensors whose name contains ``CPU`` in the WMI
    namespaces ``root/LibreHardwareMonitor`` and ``root/OpenHardwareMonitor``
    (class ``Sensor``), read through PowerShell.

A probe that raises, times out or returns nothing plausible yields ``None`` for
that probe.  Readings that are not finite or lie outside 10 to 150 degrees
Celsius are ignored.  The output of a command is read line by line, and a line
is a reading only when it consists of exactly one decimal number; lines with
text or with several numbers are ignored.  The output is decoded with
replacement characters for undecodable bytes.

A :class:`TemperatureReader` tries the probe that supplied its previous reading
first and does not run a probe again for 60 seconds after that probe failed.
"""
from __future__ import annotations

import csv
import glob
import io
import math
import os
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

import psutil

__all__ = [
    "ThermalTimeout",
    "ThermalGuard",
    "TemperatureReader",
    "read_temperature_c",
    "headroom",
]

T = TypeVar("T")

_PSUTIL_KEYS = ("coretemp", "k10temp", "cpu_thermal", "acpitz")
_SYSFS_PATTERN = "/sys/class/thermal/thermal_zone*/temp"
_SYSFS_CPU_MARKERS = ("cpu", "core", "pkg", "package", "processor", "soc", "k10temp", "tctl", "tdie", "acpitz")
_HARDWARE_MONITOR_NAMESPACES = ("root/LibreHardwareMonitor", "root/OpenHardwareMonitor")
_POWERSHELL_TIMEOUT_SECONDS = 5.0
_MIN_PLAUSIBLE_C = 10.0
_MAX_PLAUSIBLE_C = 150.0
_PROBE_RETRY_SECONDS = 60.0
_PROBE_NAMES = ("psutil", "sysfs", "windows_acpi", "windows_hardware_monitor")
_LOG_COLUMNS = ("timestamp", "temperature_c", "cpu_percent", "mem_percent", "action", "waited_seconds")
_NUMBER_LINE = re.compile(r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)")
_ACPI_COMMAND = (
    "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature "
    "| ForEach-Object { [string]$_.CurrentTemperature }"
)


class ThermalTimeout(RuntimeError):
    """The machine stayed too hot or too busy for longer than the allowed waiting time."""


# ----------------------------------------------------------------------------
# Temperature probes
# ----------------------------------------------------------------------------
def _hottest(values: Iterable[Any]) -> float | None:
    """Largest plausible temperature in a collection of readings.

    Parameters
    ----------
    values : iterable
        Readings in degrees Celsius; entries that are ``None``, not numeric,
        not finite or outside the closed range from 10 to 150 are ignored.

    Returns
    -------
    float or None
        The largest accepted reading, or ``None`` if there is none.
    """
    accepted = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and _MIN_PLAUSIBLE_C <= number <= _MAX_PLAUSIBLE_C:
            accepted.append(number)
    return max(accepted) if accepted else None


def _probe_psutil() -> float | None:
    """Hottest reading of the CPU sensors reported by ``psutil``.

    Returns
    -------
    float or None
        Temperature in degrees Celsius, or ``None`` when the platform has no
        ``sensors_temperatures`` or none of the CPU sensors report a value.
    """
    try:
        sensors = psutil.sensors_temperatures()
        readings = []
        for key in _PSUTIL_KEYS:
            for entry in sensors.get(key) or []:
                readings.append(getattr(entry, "current", None))
        return _hottest(readings)
    except Exception:
        return None


def _is_cpu_zone_type(zone_type: str) -> bool:
    """Whether a thermal zone type names a processor, package or core sensor.

    Parameters
    ----------
    zone_type : str
        Content of the ``type`` file of a thermal zone.

    Returns
    -------
    bool
        ``True`` when the lower-case type contains one of the markers ``cpu``,
        ``core``, ``pkg``, ``package``, ``processor``, ``soc``, ``k10temp``,
        ``tctl``, ``tdie`` or ``acpitz``.
    """
    text = zone_type.strip().lower()
    return any(marker in text for marker in _SYSFS_CPU_MARKERS)


def _probe_sysfs(pattern: str = _SYSFS_PATTERN) -> float | None:
    """Hottest processor thermal zone of the Linux sysfs interface.

    Parameters
    ----------
    pattern : str, default ``/sys/class/thermal/thermal_zone*/temp``
        Glob pattern of the files that hold millidegrees Celsius.  The ``type``
        file next to each matching file names the sensor of the zone.

    Returns
    -------
    float or None
        Largest plausible temperature in degrees Celsius over the zones whose
        type names a processor, package or core sensor (see
        :func:`_is_cpu_zone_type`), or ``None`` if no such zone can be read.
    """
    try:
        paths = sorted(glob.glob(pattern))
    except Exception:
        return None
    readings = []
    for path in paths:
        try:
            with open(os.path.join(os.path.dirname(path), "type"), "r", encoding="ascii", errors="replace") as handle:
                if not _is_cpu_zone_type(handle.read()):
                    continue
            with open(path, "r", encoding="ascii") as handle:
                readings.append(int(handle.read().strip()) / 1000.0)
        except Exception:
            continue
    return _hottest(readings)


def _is_windows() -> bool:
    """Whether the interpreter runs on Windows."""
    return sys.platform.startswith("win")


def _powershell_output(command: str) -> str | None:
    """Run a PowerShell command and return its standard output.

    Parameters
    ----------
    command : str
        PowerShell command text.

    Returns
    -------
    str or None
        Standard output decoded with replacement characters for undecodable
        bytes, or ``None`` if PowerShell is missing, the command fails or it
        does not finish within 5 seconds.
    """
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_POWERSHELL_TIMEOUT_SECONDS,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _numbers(text: str) -> list[float]:
    """Numbers found in the lines of a text.

    Parameters
    ----------
    text : str
        Output of a command.  A line counts when, after removing surrounding
        white space, it consists of exactly one decimal number with an optional
        sign and a decimal point or decimal comma; other lines are ignored.

    Returns
    -------
    list of float
        The numbers of the counted lines, in order.
    """
    values = []
    for line in text.splitlines():
        token = line.strip()
        if _NUMBER_LINE.fullmatch(token):
            values.append(float(token.replace(",", ".")))
    return values


def _probe_windows_acpi() -> float | None:
    """Hottest ACPI thermal zone on Windows.

    Returns
    -------
    float or None
        Temperature in degrees Celsius, converted from tenths of Kelvin, or
        ``None`` off Windows or when the query fails.
    """
    if not _is_windows():
        return None
    text = _powershell_output(_ACPI_COMMAND)
    if text is None:
        return None
    return _hottest(value / 10.0 - 273.15 for value in _numbers(text))


def _hardware_monitor_command(namespace: str) -> str:
    """PowerShell command that lists CPU temperature sensors of a WMI namespace.

    Parameters
    ----------
    namespace : str
        WMI namespace such as ``root/LibreHardwareMonitor``.

    Returns
    -------
    str
        Command text that prints the sensor values in degrees Celsius.
    """
    return (
        f"Get-CimInstance -Namespace {namespace} -ClassName Sensor "
        "| Where-Object { $_.SensorType -eq 'Temperature' -and $_.Name -like '*CPU*' } "
        "| ForEach-Object { [string]$_.Value }"
    )


def _probe_windows_hardware_monitor() -> float | None:
    """Hottest CPU sensor of LibreHardwareMonitor or OpenHardwareMonitor.

    Returns
    -------
    float or None
        Temperature in degrees Celsius from the first namespace that returns a
        plausible value, or ``None`` off Windows or when neither responds.
    """
    if not _is_windows():
        return None
    for namespace in _HARDWARE_MONITOR_NAMESPACES:
        text = _powershell_output(_hardware_monitor_command(namespace))
        if text is None:
            continue
        value = _hottest(_numbers(text))
        if value is not None:
            return value
    return None


class TemperatureReader:
    """Read the hottest CPU temperature through the probes and remember what worked.

    A reading tries the probe that supplied the previous reading first and then
    the remaining probes in the order ``psutil``, ``sysfs``, ``windows_acpi`` and
    ``windows_hardware_monitor``.  A probe that raises or returns nothing is not
    run again until ``retry_seconds`` have passed on the clock; a probe that
    succeeds is run at every reading.

    Parameters
    ----------
    clock : callable, default ``time.monotonic``
        Function without arguments that returns the current time in seconds.
    retry_seconds : float, default 60.0
        Seconds that pass after a failure of a probe before it runs again; not
        negative.

    Raises
    ------
    ValueError
        If ``retry_seconds`` is negative or not a number.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        retry_seconds: float = _PROBE_RETRY_SECONDS,
    ) -> None:
        if not retry_seconds >= 0:
            raise ValueError("retry_seconds must not be negative")
        self._clock = clock
        self._retry_seconds = float(retry_seconds)
        self._last: str | None = None
        self._failed_at: dict[str, float] = {}

    def read(self) -> tuple[float, str] | None:
        """Temperature and the name of the probe that supplied it.

        Returns
        -------
        tuple of (float, str) or None
            Reading in degrees Celsius and probe name (``psutil``, ``sysfs``,
            ``windows_acpi`` or ``windows_hardware_monitor``), or ``None`` when
            no probe that is allowed to run returns a reading.
        """
        probes = {
            "psutil": _probe_psutil,
            "sysfs": _probe_sysfs,
            "windows_acpi": _probe_windows_acpi,
            "windows_hardware_monitor": _probe_windows_hardware_monitor,
        }
        order = list(_PROBE_NAMES)
        if self._last in order:
            order.remove(self._last)
            order.insert(0, self._last)
        for name in order:
            failed_at = self._failed_at.get(name)
            if failed_at is not None and self._clock() - failed_at < self._retry_seconds:
                continue
            try:
                value = probes[name]()
            except Exception:
                value = None
            if value is not None:
                self._last = name
                self._failed_at.pop(name, None)
                return float(value), name
            self._failed_at[name] = self._clock()
        return None

    def __call__(self) -> float | None:
        """Hottest CPU temperature that the platform exposes.

        Returns
        -------
        float or None
            Temperature in degrees Celsius, or ``None`` when no probe succeeds.
        """
        found = self.read()
        return None if found is None else found[0]


def _read_with_source() -> tuple[float, str] | None:
    """Temperature and the name of the probe that supplied it, without memory.

    Returns
    -------
    tuple of (float, str) or None
        Reading in degrees Celsius and probe name from the first probe that
        succeeds, or ``None``.
    """
    return TemperatureReader(retry_seconds=0.0).read()


def read_temperature_c() -> float | None:
    """Hottest CPU temperature that the platform exposes.

    The probes ``psutil``, ``sysfs``, ``windows_acpi`` and
    ``windows_hardware_monitor`` are tried in this order and the first reading
    is returned.  A probe that raises or times out counts as no reading.  Every
    call runs the probes afresh; use :class:`TemperatureReader` to remember which
    probe worked.

    Returns
    -------
    float or None
        Temperature in degrees Celsius, or ``None`` when no probe succeeds.
    """
    found = _read_with_source()
    return None if found is None else found[0]


# ----------------------------------------------------------------------------
# Head room
# ----------------------------------------------------------------------------
def _claude_process_names() -> list[str] | None:
    """Names of the running processes whose name contains ``claude``.

    Returns
    -------
    list of str or None
        One name per matching process, case-insensitive match, sorted;
        processes whose name cannot be read are skipped.  ``None`` when the
        process list cannot be read.
    """
    names: list[str] = []
    try:
        for process in psutil.process_iter(["name"]):
            try:
                name = process.info.get("name")
            except Exception:
                continue
            if name and "claude" in str(name).lower():
                names.append(str(name))
    except Exception:
        return None
    return sorted(names)


def _attempt(function: Callable[[], T]) -> T | None:
    """Result of a function call, or ``None`` if the call raises.

    Parameters
    ----------
    function : callable
        Function without arguments.

    Returns
    -------
    object or None
        The return value of ``function``, or ``None`` if it raised an exception.
    """
    try:
        return function()
    except Exception:
        return None


def headroom() -> dict[str, Any]:
    """Current CPU, memory and temperature head room of the machine.

    Every entry that cannot be read, for example because ``psutil`` raises an
    access or operating system error, is ``None``; the function does not raise.

    Returns
    -------
    dict
        ``cpu_percent`` (``psutil.cpu_percent`` over 0.5 seconds),
        ``mem_available_gb`` (available memory in gibibytes), ``mem_percent``
        (share of memory in use), ``temperature_c`` (see
        :func:`read_temperature_c`) and ``claude_processes``, a dictionary with
        the ``count`` and the sorted ``names`` of the running processes whose
        name contains ``claude`` (case-insensitive).
    """
    cpu = _attempt(lambda: float(psutil.cpu_percent(interval=0.5)))
    memory = _attempt(psutil.virtual_memory)
    names = _claude_process_names()
    return {
        "cpu_percent": cpu,
        "mem_available_gb": _attempt(lambda: float(memory.available) / 2**30),
        "mem_percent": _attempt(lambda: float(memory.percent)),
        "temperature_c": _attempt(read_temperature_c),
        "claude_processes": None if names is None else {"count": len(names), "names": names},
    }


# ----------------------------------------------------------------------------
# Log file
# ----------------------------------------------------------------------------
def _header_line() -> str:
    """Header row of the CSV log as text.

    Returns
    -------
    str
        The column names written by :mod:`csv`, with the line terminator.
    """
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerow(_LOG_COLUMNS)
    return buffer.getvalue()


def _create_log(path: Path) -> None:
    """Make sure that the log file starts with exactly one header row.

    A temporary file in the same directory receives the header and is then
    linked to the final name, which succeeds for only one of several callers, so
    the log never exists without its header.  Where hard links are not available
    the file is created exclusively.  A file that exists but is empty receives
    the header.

    Parameters
    ----------
    path : Path
        Log file; its parent directory must exist.

    Raises
    ------
    OSError
        If the directory cannot be written.
    """
    header = _header_line().encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, flags, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(header)
        try:
            os.link(temporary, path)
            return
        except FileExistsError:
            pass
        except OSError:
            try:
                descriptor = os.open(path, flags, 0o666)
            except FileExistsError:
                pass
            else:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(header)
                return
        if os.path.getsize(path) == 0:
            with path.open("ab") as handle:
                handle.write(header)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------------
class ThermalGuard:
    """Pause a computation while the machine is too hot or too busy.

    A call of the guard (or of :meth:`check`) reads the temperature.  If a
    temperature is available and above ``max_temp_c``, the guard sleeps in steps
    of ``poll_seconds`` until a reading at or below ``resume_temp_c`` is
    obtained; a step without a reading does not end the wait.  If no temperature
    is available, the guard compares the CPU load with ``max_cpu_percent`` and,
    while the load is higher, sleeps in steps of ``poll_seconds`` and reads again;
    a temperature that appears during this wait ends it when it is at or below
    ``max_temp_c`` and starts the wait for ``resume_temp_c`` when it is above.  A
    CPU load that cannot be read counts as within the limit.  The last step of a
    wait is shortened so that the wait does not exceed ``max_wait_seconds``, and
    :class:`ThermalTimeout` is raised when that time has passed without the
    machine being within its limits.  Every call appends a row to the CSV file
    ``log_path`` when one is given.

    Parameters
    ----------
    max_temp_c : float, default 85.0
        Temperature above which the guard waits.
    resume_temp_c : float, default 78.0
        Temperature at or below which the wait ends; at most ``max_temp_c``.
    max_cpu_percent : float, default 92.0
        CPU load above which the guard waits when no temperature is available.
    poll_seconds : float, default 5.0
        Length of one sleeping step; positive.
    max_wait_seconds : float, default 1800.0
        Longest total wait of one call before :class:`ThermalTimeout`.
    log_path : str, Path or None, default None
        CSV file that receives one row per call; parent directories are created
        on the first write.  A file that is missing or empty is created together
        with its header row in a single step; guards that start together write
        one header.
    read_temp : callable or None, default None
        Function without arguments that returns degrees Celsius or ``None``;
        finite values are used as given, anything else counts as unavailable.
        ``None`` uses a :class:`TemperatureReader` driven by ``clock``.
    read_cpu : callable or None, default None
        Function without arguments that returns the CPU load in percent.  The
        default reads ``psutil.cpu_percent(interval=None)``, the load since the
        previous reading.
    sleep : callable, default ``time.sleep``
        Function that waits for the given number of seconds.
    clock : callable, default ``time.monotonic``
        Function that returns the current time in seconds.

    Raises
    ------
    ValueError
        If ``resume_temp_c`` exceeds ``max_temp_c``, ``poll_seconds`` is not
        positive or ``max_wait_seconds`` is negative.
    """

    def __init__(
        self,
        max_temp_c: float = 85.0,
        resume_temp_c: float = 78.0,
        max_cpu_percent: float = 92.0,
        poll_seconds: float = 5.0,
        max_wait_seconds: float = 1800.0,
        log_path: str | Path | None = None,
        read_temp: Callable[[], float | None] | None = None,
        read_cpu: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not resume_temp_c <= max_temp_c:
            raise ValueError("resume_temp_c must not exceed max_temp_c")
        if not poll_seconds > 0:
            raise ValueError("poll_seconds must be positive")
        if not max_wait_seconds >= 0:
            raise ValueError("max_wait_seconds must not be negative")
        self.max_temp_c = float(max_temp_c)
        self.resume_temp_c = float(resume_temp_c)
        self.max_cpu_percent = float(max_cpu_percent)
        self.poll_seconds = float(poll_seconds)
        self.max_wait_seconds = float(max_wait_seconds)
        self.log_path = None if log_path is None else Path(log_path)
        self._read_temp = read_temp
        self._with_source: Callable[[], tuple[float, str] | None] | None = None
        if read_temp is None or read_temp is read_temperature_c:
            self._with_source = TemperatureReader(clock=clock).read
        self._read_cpu = read_cpu if read_cpu is not None else _default_cpu_reader()
        self._sleep = sleep
        self._clock = clock
        self._checks = 0
        self._waits = 0
        self._waited_seconds = 0.0
        self._max_temperature: float | None = None
        self._source = "unavailable"
        self._log_warned = False

    def __call__(self) -> None:
        """Return once the machine is within its limits; see :meth:`check`."""
        self.check()

    def check(self) -> None:
        """Wait until the machine is within its limits.

        Raises
        ------
        ThermalTimeout
            If the machine is not within its limits after ``max_wait_seconds``
            of waiting: the temperature stays above ``resume_temp_c`` after it
            exceeded ``max_temp_c``, or the CPU load stays above
            ``max_cpu_percent`` while no temperature is available.  The row of
            the call is logged before the exception is raised.
        """
        started = self._clock()
        temperature = self._measure_temperature()
        cpu = self._measure_cpu()
        memory = _memory_percent()
        action = "ok"
        waited = 0.0
        hot = False
        if temperature is not None:
            if temperature > self.max_temp_c:
                action, waited, hot = self._wait(started, hot=True)
        elif cpu is not None and cpu > self.max_cpu_percent:
            action, waited, hot = self._wait(started, hot=False)
        self._checks += 1
        if action != "ok":
            self._waits += 1
            self._waited_seconds += waited
        self._append_log(temperature, cpu, memory, action, waited)
        if action == "timeout":
            if hot:
                reason = (
                    f"temperature stayed above {self.resume_temp_c:g} C for {waited:g} seconds "
                    f"after exceeding {self.max_temp_c:g} C"
                )
            else:
                reason = f"CPU load stayed above {self.max_cpu_percent:g} percent for {waited:g} seconds"
            raise ThermalTimeout(reason)

    def guarded(self, iterable: Iterable[T]) -> Iterator[T]:
        """Iterate while calling the guard before each item.

        Parameters
        ----------
        iterable : iterable
            Items to yield.

        Yields
        ------
        object
            The items of ``iterable``, each after a successful :meth:`check`.
        """
        for item in iterable:
            self.check()
            yield item

    def summary(self) -> dict[str, Any]:
        """Totals over all calls so far.

        Returns
        -------
        dict
            ``checks`` (number of calls), ``waits`` (calls that waited or timed
            out), ``waited_seconds`` (total waiting time),
            ``max_temperature_c`` (highest temperature read, ``None`` if none)
            and ``temperature_source`` (probe name, ``custom`` for a user
            supplied reader, or ``unavailable`` if no temperature was read).
        """
        return {
            "checks": self._checks,
            "waits": self._waits,
            "waited_seconds": self._waited_seconds,
            "max_temperature_c": self._max_temperature,
            "temperature_source": self._source,
        }

    def _measure_temperature(self) -> float | None:
        """Read the temperature, record it and return it.

        Returns
        -------
        float or None
            Temperature in degrees Celsius, or ``None`` if the reader fails or
            reports nothing usable.
        """
        try:
            if self._with_source is not None:
                found = self._with_source()
                if found is None:
                    return None
                value, source = found
            else:
                value, source = self._read_temp(), "custom"
            if value is None:
                return None
            value = float(value)
        except Exception:
            return None
        if not math.isfinite(value):
            return None
        self._source = source
        if self._max_temperature is None or value > self._max_temperature:
            self._max_temperature = value
        return value

    def _measure_cpu(self) -> float | None:
        """Read the CPU load.

        Returns
        -------
        float or None
            Load in percent, or ``None`` if the reader fails.
        """
        try:
            value = float(self._read_cpu())
        except Exception:
            return None
        return value if math.isfinite(value) else None

    def _wait(self, started: float, hot: bool) -> tuple[str, float, bool]:
        """Sleep in steps until the machine is within its limits or time runs out.

        Each step sleeps for ``poll_seconds`` or for the time that remains of
        ``max_wait_seconds`` if that is shorter, and is followed by a new
        temperature reading.  While the machine counts as hot, only a reading at
        or below ``resume_temp_c`` ends the wait.  Otherwise a temperature
        reading ends the wait, and without a reading the wait ends when the CPU
        load is within ``max_cpu_percent`` or cannot be read.

        Parameters
        ----------
        started : float
            Clock value at the start of the call.
        hot : bool
            ``True`` when the wait began with a temperature above ``max_temp_c``
            and ``False`` when it began with a CPU load above ``max_cpu_percent``
            and no temperature.

        Returns
        -------
        action : {"waited", "timeout"}
            Outcome of the wait.
        waited : float
            Seconds waited: the larger of the elapsed clock time and the total
            requested sleeping time.
        hot : bool
            Whether the machine counted as hot when the wait ended; a
            temperature above ``max_temp_c`` that appears during a wait that
            began without a temperature sets it.
        """
        slept = 0.0
        while True:
            waited = max(self._clock() - started, slept)
            remaining = self.max_wait_seconds - waited
            if remaining <= 0.0:
                return "timeout", waited, hot
            step = min(self.poll_seconds, remaining)
            self._sleep(step)
            slept += step
            temperature = self._measure_temperature()
            if temperature is not None and temperature > self.max_temp_c:
                hot = True
            if hot:
                ready = temperature is not None and temperature <= self.resume_temp_c
            elif temperature is not None:
                ready = True
            else:
                cpu = self._measure_cpu()
                ready = cpu is None or cpu <= self.max_cpu_percent
            if ready:
                return "waited", max(self._clock() - started, slept), hot

    def _append_log(
        self,
        temperature: float | None,
        cpu: float | None,
        memory: float | None,
        action: str,
        waited: float,
    ) -> None:
        """Append one row to the CSV log.

        Parameters
        ----------
        temperature, cpu, memory : float or None
            Readings at the start of the call; ``None`` is written as an empty field.
        action : {"ok", "waited", "timeout"}
            Outcome of the call.
        waited : float
            Seconds waited.
        """
        if self.log_path is None:
            return
        row = [
            datetime.now().astimezone().isoformat(timespec="seconds"),
            "" if temperature is None else f"{temperature:.2f}",
            "" if cpu is None else f"{cpu:.1f}",
            "" if memory is None else f"{memory:.1f}",
            action,
            f"{waited:.2f}",
        ]
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = self.log_path.stat().st_size
            except FileNotFoundError:
                size = 0
            if size == 0:
                _create_log(self.log_path)
            with self.log_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)
        except OSError as error:
            if not self._log_warned:
                self._log_warned = True
                warnings.warn(f"thermal log could not be written: {error}", RuntimeWarning, stacklevel=3)


def _default_cpu_reader() -> Callable[[], float]:
    """CPU load reader based on ``psutil.cpu_percent`` without blocking.

    One reading of ``psutil.cpu_percent(interval=None)`` is taken when the
    reader is created, which starts the interval of the first call.

    Returns
    -------
    callable
        Function without arguments that returns the load in percent since its
        previous call.
    """
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass

    def read() -> float:
        return float(psutil.cpu_percent(interval=None))

    return read


def _memory_percent() -> float | None:
    """Share of memory in use.

    Returns
    -------
    float or None
        Percent of memory in use, or ``None`` if it cannot be read.
    """
    try:
        return float(psutil.virtual_memory().percent)
    except Exception:
        return None
