#!/usr/bin/env python3
"""Local benchmark for PR #1387 TC remediation.

This script is intentionally kept under dev_notes and should not be committed to
the Apache Mahout PR branch. It measures the pushed remediation commit only.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from qumat_qdp import QdpEngine


def iqp_param_count(num_qubits: int) -> int:
    return num_qubits + num_qubits * (num_qubits - 1) // 2


def make_data(batch_size: int, num_qubits: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    n_params = iqp_param_count(num_qubits)
    return torch.randn(batch_size, n_params, dtype=torch.float64).numpy()


def sync() -> None:
    torch.cuda.synchronize()


def time_call(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        out = fn()
        _ = torch.from_dlpack(out).shape
    sync()

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        tensor = torch.from_dlpack(out)
        sync()
        _ = tensor.shape
        samples.append((time.perf_counter() - start) * 1000.0)

    return statistics.median(samples), statistics.mean(samples)


def measure(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    engine = QdpEngine(device_id=0, precision="float64")
    rows: list[dict[str, float | int | str]] = []

    for num_qubits in args.qubits:
        batch_size = args.batch_size
        if num_qubits >= 18:
            batch_size = min(batch_size, args.large_batch_size)

        data = make_data(batch_size, num_qubits, seed=args.seed)
        state_len = 1 << num_qubits

        fwt_fn = lambda: engine.encode(data, num_qubits, "iqp")
        tc_fn = lambda: engine.encode_batch_tc(data, num_qubits)

        fwt_median, fwt_mean = time_call(fwt_fn, args.warmup, args.iters)
        tc_median, tc_mean = time_call(tc_fn, args.warmup, args.iters)

        fwt_state = torch.from_dlpack(engine.encode(data, num_qubits, "iqp")).clone()
        tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits)).clone()
        sync()
        max_err = (fwt_state - tc_state).abs().max().item()
        tc_norm_err = (
            tc_state.abs().pow(2).sum(dim=1) - torch.ones(batch_size, device=tc_state.device)
        ).abs().max().item()

        rows.append(
            {
                "num_qubits": num_qubits,
                "batch_size": batch_size,
                "state_len": state_len,
                "fwt_median_ms": fwt_median,
                "tc_median_ms": tc_median,
                "fwt_mean_ms": fwt_mean,
                "tc_mean_ms": tc_mean,
                "tc_vs_fwt_speedup": fwt_median / tc_median if tc_median else math.nan,
                "max_abs_err_vs_fwt": max_err,
                "tc_norm_max_err": tc_norm_err,
            }
        )
        print(
            f"N={num_qubits:2d} batch={batch_size:2d} "
            f"FWT={fwt_median:8.3f} ms TC={tc_median:8.3f} ms "
            f"speedup={fwt_median / tc_median:6.2f}x err={max_err:.3e}"
        )

    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, float | int | str]], path: Path, commit: str) -> None:
    lines = [
        "# PR1387 TC Benchmark",
        "",
        f"Commit: `{commit}`",
        "",
        "| N | batch | FWT median ms | TC median ms | TC speedup | max abs err vs FWT | TC norm max err |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {num_qubits} | {batch_size} | {fwt_median_ms:.3f} | "
            "{tc_median_ms:.3f} | {tc_vs_fwt_speedup:.2f}x | "
            "{max_abs_err_vs_fwt:.3e} | {tc_norm_max_err:.3e} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(rows: list[dict[str, float | int | str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    ns = [int(r["num_qubits"]) for r in rows]
    fwt = [float(r["fwt_median_ms"]) for r in rows]
    tc = [float(r["tc_median_ms"]) for r in rows]
    err = [max(float(r["max_abs_err_vs_fwt"]), 1e-18) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ns, fwt, marker="o", label="Default FWT encode")
    axes[0].plot(ns, tc, marker="o", label="Tensor Core encode_batch_tc")
    axes[0].set_xlabel("Number of qubits")
    axes[0].set_ylabel("Median latency (ms)")
    axes[0].set_title("IQP encode latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(ns, err, marker="o", color="#b04040")
    axes[1].set_xlabel("Number of qubits")
    axes[1].set_ylabel("max |TC - FWT| (zero shown as 1e-18)")
    axes[1].set_title("TC agreement vs default FWT")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="dev_notes/pr1387_benchmark")
    parser.add_argument("--qubits", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--large-batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    rows = measure(args)

    out_dir = Path(args.out_dir)
    write_csv(rows, out_dir / "pr1387_tc_benchmark.csv")
    write_markdown(rows, out_dir / "pr1387_tc_benchmark.md", commit)
    write_plot(rows, out_dir / "pr1387_tc_benchmark.png")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
