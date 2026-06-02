#!/usr/bin/env python3
"""Aggregate reports/ncu_*.csv into ncu_summary.txt + ncu_summary.md + status table."""
from __future__ import annotations

import csv
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

METRIC_LABELS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM throughput",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 throughput",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM throughput",
    "smsp__warps_active.avg.pct_of_peak_sustained_active": "Warps active",
    "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active": "Tensor pipe active",
    "launch__registers_per_thread": "Registers / thread",
    "launch__shared_mem_per_block": "Shared memory / block",
    "gpu__time_duration.sum": "Kernel duration (us, avg)",
}

# PR007 kernel tags -> human title
KERNEL_CASES = [
    ("ncu_fwt_baseline.csv", "fwt_butterfly_batch_kernel", "Baseline FWT (14 butterfly stages, batch=1)"),
    ("ncu_phase_split.csv", "iqp_phase_split_kernel", "TC: phase split"),
    ("ncu_modulo_precompute.csv", "precompute_modulo_kernel_p26_implicit", "TC: modulo precompute"),
    ("ncu_ozaki_grid.csv", "implicit_hadamard_ozaki_grid_kernel_implicit", "Ozaki MMA grid (OZAKI_NCU_PROFILE=1)"),
    ("ncu_ozaki_persistent.csv", "implicit_hadamard_ozaki_persistent_kernel_implicit", "Ozaki MMA persistent (production)"),
]


def parse_csv(path: Path) -> tuple[dict[str, list[float]], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    status = "OK"
    if "LaunchFailed" in text:
        status = "LaunchFailed"
    if "(0, 0, 0)" in text and "Grid Size" in text:
        if status == "OK":
            status = "LaunchFailed (zero grid)"

    by_metric: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line.startswith('"') or "Metric Name" in line:
            continue
        try:
            row = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(row) < 15 or not row[0].isdigit():
            continue
        metric, val = row[12], row[14].replace(",", "")
        if val.lower() in ("nan", "n/a", ""):
            continue
        try:
            by_metric.setdefault(metric, []).append(float(val))
        except ValueError:
            pass

    if not by_metric and status == "OK":
        status = "No metrics"
    return {m: vs for m, vs in by_metric.items()}, status


def format_metric(name: str, values: list[float]) -> str:
    v = statistics.mean(values)
    if name == "gpu__time_duration.sum":
        return f"  {METRIC_LABELS.get(name, name)}: {v / 1000:.2f}"
    if "Registers" in METRIC_LABELS.get(name, "") or "Shared" in METRIC_LABELS.get(name, ""):
        return f"  {METRIC_LABELS.get(name, name)}: {v:.0f}"
    label = METRIC_LABELS.get(name, name)
    return f"  {label}: {v:.2f}%" if "throughput" in name or "warps" in name or "pipe_tensor" in name or "active" in name else f"  {label}: {v:.2f}"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines_txt = [
        f"PR007 NCU Summary — Colab bundle — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    md_rows = [
        "# PR007 NCU Summary (All Cases)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Case | NCU status | DRAM % | SM % | Tensor % | Warps % | Duration (us) |",
        "|------|------------|--------|------|----------|---------|---------------|",
    ]

    for filename, _kernel, title in KERNEL_CASES:
        path = REPORTS / filename
        lines_txt.append(title)
        lines_txt.append("-" * min(70, len(title)))
        if not path.exists():
            lines_txt.append("  (not run — missing CSV)")
            md_rows.append(f"| {title} | MISSING | — | — | — | — | — |")
            lines_txt.append("")
            continue

        metrics, status = parse_csv(path)
        lines_txt.append(f"  NCU status: {status}")
        order = list(METRIC_LABELS.keys())
        for k in order:
            if k in metrics:
                lines_txt.append(format_metric(k, metrics[k]))

        dram = sm = tensor = warps = dur = "—"
        if metrics:
            if "dram__throughput.avg.pct_of_peak_sustained_elapsed" in metrics:
                dram = f"{statistics.mean(metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed']):.1f}"
            if "sm__throughput.avg.pct_of_peak_sustained_elapsed" in metrics:
                sm = f"{statistics.mean(metrics['sm__throughput.avg.pct_of_peak_sustained_elapsed']):.1f}"
            if "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active" in metrics:
                tensor = f"{statistics.mean(metrics['sm__pipe_tensor_active.avg.pct_of_peak_sustained_active']):.1f}"
            if "smsp__warps_active.avg.pct_of_peak_sustained_active" in metrics:
                warps = f"{statistics.mean(metrics['smsp__warps_active.avg.pct_of_peak_sustained_active']):.1f}"
            if "gpu__time_duration.sum" in metrics:
                dur = f"{statistics.mean(metrics['gpu__time_duration.sum']) / 1000:.2f}"

        md_rows.append(f"| {title} | {status} | {dram} | {sm} | {tensor} | {warps} | {dur} |")
        lines_txt.append("")

    bench = REPORTS / "benchmark_results.txt"
    lines_txt.append("Benchmark (see reports/benchmark_results.txt)")
    if bench.exists():
        for line in bench.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Duration:" in line or "PASSED" in line or "Max Absolute" in line:
                lines_txt.append(f"  {line.strip()}")
    lines_txt.append("")
    lines_txt.append("Raw CSV: reports/ncu_*.csv | Logs: reports/ncu_*.log")

    txt_path = REPORTS / "ncu_summary.txt"
    md_path = REPORTS / "ncu_summary.md"
    txt_path.write_text("\n".join(lines_txt), encoding="utf-8")
    md_path.write_text("\n".join(md_rows) + "\n\n## Detail\n\nSee `ncu_summary.txt`.\n", encoding="utf-8")

    print(txt_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())