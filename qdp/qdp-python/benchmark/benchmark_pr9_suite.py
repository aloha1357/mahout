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

"""PR9 benchmark suite: E2E latency, native encode timing, and fast pytest."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]

E2E_FILTER = re.compile(
    r"^(  |Mahout-|E2E|Samples:|={3,}|encode_batch|Total Time|Arrow read|"
    r"VERIFY|\[Mahout|Max Amplitude|>> SUCCESS)"
)


def _run(cmd: list[str], *, cwd: Path | None = None, filter_stdout: bool = False) -> int:
    if filter_stdout:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if E2E_FILTER.match(line):
                print(line, end="")
        return proc.wait()
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


def run_e2e(qubits: list[int], samples: int) -> None:
    for n in qubits:
        print(f"\n########## N={n} ##########")
        _run(
            [
                sys.executable,
                "benchmark_e2e.py",
                "--qubits",
                str(n),
                "--samples",
                str(samples),
                "--encoding-method",
                "iqp-z",
                "--frameworks",
                "mahout-arrow",
                "mahout-tc",
                "mahout-native",
            ],
            cwd=BENCH_DIR,
            filter_stdout=True,
        )


def run_encode(label: str, qubits: list[int], samples: int) -> None:
    print("\n=== fp32/fp64 encode GPU timing ===")
    _run(
        [
            sys.executable,
            "benchmark_iqp_native.py",
            "--label",
            label,
            "--qubits",
            *[str(n) for n in qubits],
            "--samples",
            str(samples),
        ],
        cwd=BENCH_DIR,
    )


def run_pytest() -> None:
    print("\n=== pytest (fast) ===")
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "testing/qdp/test_iqp_native_path.py",
            "testing/qdp/test_iqp_tc_path.py",
            "testing/qdp/test_iqp_native_e2e.py",
            "testing/qdp/test_iqp_native_fp32.py",
            "-m",
            "not slow",
            "-q",
            "--tb=no",
        ],
        cwd=REPO_ROOT,
    )


def git_head() -> str:
    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="PR9 full benchmark suite")
    parser.add_argument("--label", default="PR9")
    parser.add_argument("--qubits", type=int, nargs="+", default=[12, 14, 16])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    print(f"=== Benchmark suite: {args.label} ===")
    print(f"git: {git_head()}")

    if not args.skip_e2e:
        run_e2e(args.qubits, args.samples)
    if not args.skip_encode:
        run_encode(args.label, args.qubits, args.samples)
    if not args.skip_pytest:
        run_pytest()


if __name__ == "__main__":
    main()