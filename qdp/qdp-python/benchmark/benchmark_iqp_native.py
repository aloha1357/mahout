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

"""GPU encode_batch_native timing for IQP (fp32/fp64)."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from qumat_qdp import QdpEngine

from utils import generate_batch_data


def bench_encode(
    precision: str,
    num_qubits: int,
    num_samples: int,
    rounds: int,
    warmup: int,
) -> float:
    engine = QdpEngine(device_id=0, precision=precision)
    dtype = np.float32 if precision == "float32" else np.float64
    data = generate_batch_data(num_samples, num_qubits, "iqp-z", seed=42).astype(dtype)

    for _ in range(warmup):
        engine.encode_batch_native(data, num_qubits, "iqp-z")
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        engine.encode_batch_native(data, num_qubits, "iqp-z")
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="IQP native encode GPU benchmark")
    parser.add_argument("--qubits", type=int, nargs="+", default=[12, 14, 16])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--precision",
        choices=["float32", "float64", "both"],
        default="both",
    )
    parser.add_argument("--label", default="HEAD")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    precisions = (
        ["float32", "float64"] if args.precision == "both" else [args.precision]
    )

    print("=" * 72)
    print(f"IQP native encode benchmark  [{args.label}]")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"samples={args.samples}, rounds={args.rounds}, "
        f"warmup={args.warmup}, qubits={args.qubits}"
    )
    print("=" * 72)

    results: dict[tuple[str, int], float] = {}
    for prec in precisions:
        print(f"\n--- {prec} ---")
        for n in args.qubits:
            ms = bench_encode(prec, n, args.samples, args.rounds, args.warmup)
            per_us = ms / args.samples * 1000.0
            results[(prec, n)] = ms
            path = "fused IQP" if n <= 12 else "Kronecker+fused-transpose"
            print(
                f"  N={n:2d}  {ms:8.3f} ms/batch  ({per_us:6.1f} us/sample)  [{path}]"
            )

    if args.precision == "both" and len(args.qubits) > 0:
        print("\n--- fp32 / fp64 ---")
        for n in args.qubits:
            if ("float32", n) in results and ("float64", n) in results:
                ratio = results[("float32", n)] / results[("float64", n)]
                print(f"  N={n:2d}  {ratio:.2f}x")


if __name__ == "__main__":
    main()