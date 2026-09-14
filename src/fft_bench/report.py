"""Markdown and figure output for FFT benchmark results."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PRECISION_ORDER = ("float32", "float64", "longdouble")
PRECISION_COLORS = {
    "float32": "#0072B2",
    "float64": "#D55E00",
    "longdouble": "#009E73",
}


def load_payload(path: Path) -> dict[str, Any]:
    """Load one FFT payload, accepting a bare record list for convenience."""

    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"records": data, "skipped": [], "failures": []}
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain an FFT result payload")
    return {
        **data,
        "records": list(data.get("records", [])),
        "skipped": list(data.get("skipped", [])),
        "failures": list(data.get("failures", [])),
    }


def _successful_records(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        records.extend(
            record
            for record in payload.get("records", [])
            if record.get("status", "ok") == "ok"
        )
    return records


def _skipped_records(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for payload in payloads:
        skipped.extend(payload.get("skipped", []))
    return skipped


def _failure_records(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for payload in payloads:
        failures.extend(payload.get("failures", []))
    return failures


def format_milliseconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    milliseconds = seconds * 1e3
    if milliseconds < 0.001:
        return "<0.001"
    if milliseconds < 1:
        return f"{milliseconds:.3f}"
    if milliseconds < 100:
        return f"{milliseconds:.2f}"
    return f"{milliseconds:.1f}"


def format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "—"
    return f"{numerator / denominator:.3g}×"


def _environment_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    first = records[0]
    build = first.get("build_provenance", {})
    return {
        "OS": first.get("os", first.get("platform", "unknown")),
        "Architecture": first.get("architecture", first.get("machine", "unknown")),
        "CPU": first.get("cpu", "unknown"),
        "Python": first.get("python", "unknown"),
        "NumPy": first.get("numpy_version", "recorded by dtype runtime"),
        "DUCC": first.get("backend_version", "unknown"),
        "DUCC binding": first.get("ducc_wrapper", "unknown"),
        "Build mode": build.get("build_mode", "unknown"),
        "DUCC source": build.get("ducc_source", "unknown"),
        "DUCC optimization": build.get("ducc_optimization", "unknown"),
        "Long-double representation": first.get(
            "longdouble_classification", "not recorded"
        ),
    }


def _record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record["kind"]),
        str(record["requested_precision"]),
        int(record["fft_size"]),
        int(record["threads_requested"]),
    )


def _median_record_map(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    result: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for record in records:
        result[_record_key(record)] = record
    return result


def _table_for_kind(records: list[dict[str, Any]], kind: str) -> list[str]:
    rows = [record for record in records if record.get("kind") == kind]
    threads = sorted({int(record["threads_requested"]) for record in rows})
    lines = [
        "| FFT size | Threads | float32 (ms) | float64 (ms) | longdouble (ms) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for thread in threads:
        sizes = sorted(
            {
                int(record["fft_size"])
                for record in rows
                if int(record["threads_requested"]) == thread
            }
        )
        for size in sizes:
            cells = [str(size), str(thread)]
            for precision in PRECISION_ORDER:
                match = next(
                    (
                        record
                        for record in rows
                        if record["requested_precision"] == precision
                        and int(record["threads_requested"]) == thread
                        and int(record["fft_size"]) == size
                    ),
                    None,
                )
                cells.append(
                    format_milliseconds(match.get("median_s") if match else None)
                )
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _slowdown_table(records: list[dict[str, Any]], kind: str) -> list[str]:
    values = _median_record_map(records)
    combinations = sorted(
        {
            (int(record["fft_size"]), int(record["threads_requested"]))
            for record in records
            if record.get("kind") == kind
        }
    )
    lines = [
        "| FFT size | Threads | longdouble / float64 | float32 / float64 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for size, thread in combinations:
        f32 = values.get((kind, "float32", size, thread))
        f64 = values.get((kind, "float64", size, thread))
        ld = values.get((kind, "longdouble", size, thread))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(size),
                    str(thread),
                    format_ratio(
                        ld.get("median_s") if ld else None,
                        f64.get("median_s") if f64 else None,
                    ),
                    format_ratio(
                        f32.get("median_s") if f32 else None,
                        f64.get("median_s") if f64 else None,
                    ),
                ]
            )
            + " |"
        )
    return lines


def build_report(
    records: list[dict[str, Any]],
    skipped: list[dict[str, Any]] | None = None,
    failures: list[dict[str, Any]] | None = None,
    *,
    plot_files: list[str] | None = None,
) -> str:
    """Render the self-contained human-readable report as Markdown."""

    skipped = skipped or []
    failures = failures or []
    lines = [
        "# DUCC FFT benchmark",
        "",
        "This report contains forward one-dimensional DUCC FFT timings. Backend setup, input generation, output allocation, dtype validation, and warm-up are outside the measured samples; `out=` is preallocated for every timed transform.",
        "",
        "Timing alone is not a numerical-accuracy comparison.",
        "",
        "## Runtime environment",
        "",
    ]
    environment = _environment_summary(records)
    if environment:
        lines.extend(["| Field | Detected value |", "| --- | --- |"])
        for key, value in environment.items():
            lines.append(f"| {key} | {value} |")
        first = records[0]
        lines.extend(
            [
                "",
                f"**Detected longdouble:** `{first.get('longdouble_classification', 'not recorded')}`.",
                "",
                "NumPy `finfo.nmant` is reported separately below as the mantissa-bit count; for the usual normalized binary formats the effective significand precision is `nmant + 1`.",
                "",
                "| longdouble field | Value |",
                "| --- | ---: |",
                f"| itemsize (bytes) | {first.get('longdouble_itemsize', '—')} |",
                f"| finfo.bits | {first.get('longdouble_bits', '—')} |",
                f"| finfo.nmant | {first.get('longdouble_nmant', '—')} |",
                f"| finfo.eps | {first.get('longdouble_eps_text', first.get('longdouble_eps', '—'))} |",
            ]
        )
    else:
        lines.append("No successful timing records were produced.")

    lines.extend(["", "## Median transform time", ""])
    if records:
        for kind in ("r2c", "c2c"):
            if any(record.get("kind") == kind for record in records):
                lines.extend([f"### {kind}", "", *_table_for_kind(records, kind), ""])
    else:
        lines.append("No successful timing records were produced.")

    lines.extend(["## Long-double slowdown", ""])
    lines.append(
        "Ratios use matching FFT size and requested thread count. A missing value means that the corresponding precision was unsupported or not run."
    )
    lines.append("")
    for kind in ("r2c", "c2c"):
        if any(record.get("kind") == kind for record in records):
            lines.extend([f"### {kind}", "", *_slowdown_table(records, kind), ""])

    if plot_files:
        lines.extend(["## Figures", ""])
        lines.extend(f"- [{path}]({path})" for path in plot_files)
        lines.append("")

    if skipped:
        lines.extend(
            [
                "## Unsupported requested cases",
                "",
                "No timing result was fabricated for these cases.",
                "",
            ]
        )
        lines.extend(
            "- `{kind}/{requested_precision}`, N={fft_size}, threads={threads_requested}: {reason}".format(
                **item
            )
            for item in skipped
        )
        lines.append("")
    if failures:
        lines.extend(["## Setup failures", ""])
        for failure in failures:
            lines.append(f"- {failure.get('reason', failure)}")
        lines.append("")

    lines.extend(
        [
            "## Method notes",
            "",
            "DUCC is invoked through `ducc0.fft.r2c` or `ducc0.fft.c2c` with `forward=True`, `inorm=0`, explicit `nthreads`, and a preallocated output. Matrix thread-count cells are process-isolated. The longdouble label is portable; the detected ABI above is the scientific interpretation of that label.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_records(records: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    created: list[str] = []
    for kind in ("r2c", "c2c"):
        kind_records = [record for record in records if record.get("kind") == kind]
        for threads in sorted(
            {int(record["threads_requested"]) for record in kind_records}
        ):
            rows = [
                record
                for record in kind_records
                if int(record["threads_requested"]) == threads
            ]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(6.6, 4.3))
            for precision in PRECISION_ORDER:
                precision_rows = sorted(
                    [
                        record
                        for record in rows
                        if record.get("requested_precision") == precision
                    ],
                    key=lambda record: int(record["fft_size"]),
                )
                if not precision_rows:
                    continue
                ax.plot(
                    [int(record["fft_size"]) for record in precision_rows],
                    [float(record["median_s"]) * 1e3 for record in precision_rows],
                    marker="o",
                    color=PRECISION_COLORS[precision],
                    label=precision,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("1-D FFT size")
            ax.set_ylabel("median forward transform time (ms)")
            ax.set_title(f"DUCC {kind} — {threads} thread(s)")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = output / f"{kind}-threads-{threads}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            created.append(path.name)
    return created


def write_report(payloads: Iterable[dict[str, Any]], output: Path) -> list[str]:
    """Write only ``README.md`` and PNG figures into *output*."""

    payload_list = list(payloads)
    records = _successful_records(payload_list)
    skipped = _skipped_records(payload_list)
    failures = _failure_records(payload_list)
    output.mkdir(parents=True, exist_ok=True)
    plot_files = _plot_records(records, output) if records else []
    (output / "README.md").write_text(
        build_report(records, skipped, failures, plot_files=plot_files)
    )
    return plot_files
