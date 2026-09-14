"""Portable process and host metadata used by benchmark records."""

from __future__ import annotations

import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def cpu_model() -> str:
    """Return a best-effort human-readable CPU model."""

    if sys_platform_is_linux():
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except (OSError, IndexError):
            pass
    return platform.processor() or "unknown"


def sys_platform_is_linux() -> bool:
    return platform.system() == "Linux"


def _process_affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return sorted(int(cpu) for cpu in getter(0))
    except OSError:
        return None


def _linux_cpuinfo() -> list[str]:
    if not sys_platform_is_linux():
        return []
    try:
        return Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return []


def _cpu_flags() -> list[str] | None:
    for line in _linux_cpuinfo():
        if line.lower().startswith(("flags", "features")):
            return line.split(":", 1)[1].split()
    return None


def _physical_cpu_count() -> int | None:
    pairs: set[tuple[str, str]] = set()
    processor_info: dict[str, str] = {}
    for line in _linux_cpuinfo() + [""]:
        if line.strip():
            if ":" in line:
                key, value = line.split(":", 1)
                processor_info[key.strip().lower()] = value.strip()
            continue
        physical = processor_info.get("physical id")
        core = processor_info.get("core id")
        if physical is not None and core is not None:
            pairs.add((physical, core))
        processor_info = {}
    if pairs:
        return len(pairs)
    return None


def _power_management() -> dict[str, str] | None:
    if not sys_platform_is_linux():
        return None
    values: dict[str, str] = {}
    for name in ("scaling_driver", "scaling_governor", "energy_performance_preference"):
        path = Path("/sys/devices/system/cpu/cpu0/cpufreq") / name
        try:
            values[name] = path.read_text().strip()
        except OSError:
            continue
    return values or None


def collect_environment(*, include_cpu_controls: bool = False) -> dict[str, Any]:
    """Collect stable metadata without importing numerical backends.

    The default fields intentionally match the existing ``sht-bench`` record
    schema.  FFT records request the optional process/CPU-control fields.
    """

    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
    }
    if not include_cpu_controls:
        return result

    if result["cpu"] == "unknown":
        for line in _linux_cpuinfo():
            if line.lower().startswith(("hardware", "processor")) and ":" in line:
                result["cpu"] = line.split(":", 1)[1].strip()
                break

    affinity = _process_affinity()
    physical = _physical_cpu_count()
    logical = result["logical_cpus"]
    result.update(
        {
            "os": platform.system(),
            "architecture": platform.machine(),
            "process_affinity": affinity,
            "process_affinity_cpus": len(affinity) if affinity is not None else None,
            "physical_cpus": physical,
            "smt_status": (
                "enabled"
                if physical is not None and logical is not None and logical > physical
                else "not-detected"
            ),
            "cpu_instruction_set": _cpu_flags(),
            "power_management": _power_management(),
        }
    )
    return result
