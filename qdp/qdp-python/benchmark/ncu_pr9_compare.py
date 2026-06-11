#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare PR8 vs PR9 native encode via nsys launch counts and NCU metrics."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
OUT_DIR = REPO_ROOT / "qdp" / "qdp-kernels" / "reports" / "pr9_ncu"
PROFILE_PY = OUT_DIR / "ncu_profile_native_encode.py"

KERNEL_PATTERN = re.compile(
    r"native_fp64|iqp_phase|transpose|recombine|extreme_iqp"
)


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = False) -> int:
    return subprocess.run(cmd, cwd=cwd, check=check).returncode


def _capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False
    )
    return result.stdout + result.stderr


def run_nsys(label: str, n: int) -> None:
    rep = OUT_DIR / f"{label}_N{n}"
    print(f"=== nsys {label} N={n} ===")
    _run(
        [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "-o",
            str(rep),
            sys.executable,
            str(PROFILE_PY),
            "--qubits",
            str(n),
            "--samples",
            "32",
            "--rounds",
            "10",
        ],
        cwd=REPO_ROOT,
    )
    kern_sum = OUT_DIR / f"{label}_N{n}_kern_sum.txt"
    kern_sum.write_text(
        _capture(
            [
                "nsys",
                "stats",
                "--force-overwrite=true",
                "--report",
                "cuda_gpu_kern_sum",
                f"{rep}.nsys-rep",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        line
        for line in kern_sum.read_text(encoding="utf-8").splitlines()
        if KERNEL_PATTERN.search(line)
    ]
    for line in lines[:40]:
        print(line)


def run_ncu(label: str, n: int) -> None:
    rep = OUT_DIR / f"{label}_N{n}_ncu"
    print(f"=== ncu {label} N={n} ===")
    log_path = OUT_DIR / f"{label}_N{n}_ncu.log"
    cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--set",
        "basic",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        "regex:native_fp64|iqp_phase|transpose|recombine",
        "-o",
        str(rep),
        sys.executable,
        str(PROFILE_PY),
        "--qubits",
        str(n),
        "--samples",
        "32",
        "--rounds",
        "5",
    ]
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False
        )
    if proc.returncode != 0:
        print(f"ncu exited {proc.returncode} (see {log_path})")

    ncu_rep = Path(f"{rep}.ncu-rep")
    if ncu_rep.exists():
        summary = OUT_DIR / f"{label}_N{n}_ncu_summary.txt"
        text = _capture(
            ["ncu", "--import", str(ncu_rep), "--page", "details", "--print-summary", "per-kernel"]
        )
        summary.write_text(text, encoding="utf-8")
        print(text)


def profile_ref(ref: str, label: str) -> None:
    wt = OUT_DIR / f"worktree_{label}"
    print(f"\n############################################")
    print(f"# Profiling {label} @ {ref}")
    print(f"############################################")
    if wt.exists():
        shutil.rmtree(wt)
    _run(["git", "worktree", "add", "--detach", str(wt), ref], cwd=REPO_ROOT, check=True)
    _run(
        ["maturin", "develop", "--release"],
        cwd=wt / "qdp" / "qdp-python",
        check=True,
    )
    for n in (12, 14, 16):
        run_nsys(label, n)
    run_ncu(label, 14)
    run_ncu(label, 16)


def print_launch_diff() -> None:
    print("\n=== Launch-count diff (grep native/transpose) ===")
    for n in (12, 14, 16):
        print(f"--- N={n} ---")
        for label in ("PR8", "PR9"):
            print(f"{label}:")
            path = OUT_DIR / f"{label}_N{n}_kern_sum.txt"
            if not path.exists():
                print("(no data)")
                continue
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if KERNEL_PATTERN.search(line)
            ]
            if not lines:
                print("(no data)")
            else:
                for line in lines[:15]:
                    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="NCU/nsys PR8 vs PR9 compare")
    parser.add_argument("pr8_ref", nargs="?", default="840f016f8")
    parser.add_argument("pr9_ref", nargs="?", default="HEAD")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BENCH_DIR / "ncu_profile_native_encode.py", PROFILE_PY)

    profile_ref(args.pr8_ref, "PR8")
    profile_ref(args.pr9_ref, "PR9")

    _run(
        ["maturin", "develop", "--release"],
        cwd=REPO_ROOT / "qdp" / "qdp-python",
        check=True,
    )
    print_launch_diff()
    print(f"\nReports written to {OUT_DIR}/")


if __name__ == "__main__":
    main()